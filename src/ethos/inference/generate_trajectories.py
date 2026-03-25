"""Trajectory generation for an arbitrary (subject_id, prediction_time) index.

Usage example::

    import polars as pl
    from ethos.inference.generate_trajectories import generate_trajectories

    index_df = pl.read_parquet("task_labels/tuning/labels.parquet")
    generate_trajectories(
        model_fp="checkpoints/model.pt",
        input_dir="data/tokenized",
        index_df=index_df,
        output_dir="generated_trajectories",
        n_trajectories=20,
        device="cuda",
    )

Output format
-------------
Writes ``output_dir/{0..n_trajectories-1}.parquet``, one file per trajectory
repetition, each containing one row per generated event across all samples.
Schema matches GeneratedTrajectorySchema from MEDS_trajectory_evaluation:

    subject_id      Int64
    time            Datetime[μs]
    prediction_time Datetime[μs]
    code            Utf8
    numeric_value   Float32  (always null — ethos encodes values as quantile tokens)
"""

from datetime import timedelta
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import torch as th
from loguru import logger
from tqdm import tqdm

from ..datasets.index import IndexDataset
from ..utils import load_model_checkpoint, setup_torch
from ..vocabulary import Vocabulary
from .utils import get_next_token

# Polars schema used when building intermediate DataFrames (times as Int64
# microseconds; cast to Datetime before writing).
_TRAJECTORY_SCHEMA = {
    "subject_id": pl.Int64,
    "time": pl.Int64,
    "prediction_time": pl.Int64,
    "code": pl.Utf8,
    "numeric_value": pl.Float32,
}

# PyArrow schema for the output parquet files.
_PA_SCHEMA = pa.schema(
    [
        ("subject_id", pa.int64()),
        ("time", pa.timestamp("us")),
        ("prediction_time", pa.timestamp("us")),
        ("code", pa.string()),
        ("numeric_value", pa.float32()),
    ]
)


def _generate_for_sample(
    model,
    x: th.Tensor | tuple[th.Tensor, th.Tensor],
    vocab: Vocabulary,
    n_positions: int,
    ctx_size: int,
    stop_token_strs: set[str],
    time_limit_us: float,
    n_trajectories: int,
    device: str,
    autocast_ctx,
    temperature: float,
) -> list[list[int]]:
    """Generate *n_trajectories* continuations for a single patient context.

    All repetitions share the same input context but are sampled independently.
    Generation stops for each repetition individually when it produces a stop
    token or exceeds *time_limit_us* of simulated time.

    Args:
        model: The ethos GPT model (or encoder-decoder model).
        x: Patient context tensor ``[seq_len]`` (decoder-only) or
            ``(ctx_tensor [ctx_len], timeline_tensor [seq_len])`` (enc-dec).
        vocab: Vocabulary for decoding token IDs.
        n_positions: Model's maximum sequence length.
        ctx_size: Length of the static patient context prefix (decoder-only).
        stop_token_strs: Token strings at which to halt generation.
        time_limit_us: Maximum simulated time in microseconds.
        n_trajectories: Number of independent repetitions to generate.
        device: Torch device string.
        autocast_ctx: ``torch.amp.autocast`` context (or ``nullcontext``).
        temperature: Sampling temperature.

    Returns:
        List of length *n_trajectories*, each element a list of generated token IDs.
    """
    is_encoder_decoder = isinstance(x, tuple)

    if is_encoder_decoder:
        ctx_tensor, timeline_tensor = x
        ctx_batch = ctx_tensor.unsqueeze(0).expand(n_trajectories, -1).contiguous().to(device)
        timeline = timeline_tensor.unsqueeze(0).expand(n_trajectories, -1).contiguous().to(device)
    else:
        ctx_batch = None
        timeline = x.unsqueeze(0).expand(n_trajectories, -1).contiguous().to(device)

    # active_idx[i] = original trajectory index for batch position i
    active_idx = list(range(n_trajectories))
    active_gen_times = th.zeros(len(active_idx), dtype=th.float64)
    generated: list[list[int]] = [[] for _ in range(n_trajectories)]
    offset = 0
    interval_means = vocab.interval_estimates["mean"]

    while active_idx:
        with autocast_ctx:
            next_tokens = get_next_token(model, timeline, ctx=ctx_batch, temperature=temperature)
        # next_tokens: [n_active, 1]

        # Once the sequence reaches the model's context limit, start sliding:
        # drop the oldest non-context token on each subsequent step.
        if not offset and timeline.size(1) == n_positions:
            offset = 1

        if ctx_batch is not None:
            # Encoder-decoder: ctx is fixed; slide the decoder timeline only.
            timeline = th.cat([timeline[:, offset:], next_tokens], dim=1)
        else:
            # Decoder-only: preserve the static context prefix, slide the rest.
            timeline = th.cat(
                [timeline[:, :ctx_size], timeline[:, ctx_size + offset:], next_tokens],
                dim=1,
            )

        new_tokens_cpu = next_tokens.cpu().view(-1).tolist()
        done: list[int] = []
        for local_i, tok_id in enumerate(new_tokens_cpu):
            traj_i = active_idx[local_i]
            generated[traj_i].append(tok_id)
            tok_str = vocab.decode(tok_id)
            active_gen_times[local_i] += interval_means.get(tok_str, 0)
            if tok_str in stop_token_strs or active_gen_times[local_i].item() > time_limit_us:
                done.append(local_i)

        if done:
            keep = th.ones(len(active_idx), dtype=th.bool)
            for i in done:
                keep[i] = False
            active_idx = [active_idx[i] for i in range(len(active_idx)) if keep[i]]
            timeline = timeline[keep]
            active_gen_times = active_gen_times[keep]
            if ctx_batch is not None:
                ctx_batch = ctx_batch[keep]

    return generated


def _format_trajectory(
    vocab: Vocabulary,
    metadata: dict,
    generated_token_ids: list[int],
) -> list[dict]:
    """Convert generated token IDs into rows for the GeneratedTrajectorySchema.

    The clock starts at ``metadata["window_last_observed"]`` and advances
    whenever a time-interval token (one present in ``vocab.interval_estimates``)
    is decoded.  All tokens — including time-interval tokens — are emitted as
    rows.  ``numeric_value`` is always null because ethos encodes numeric
    measurements as quantile tokens rather than separate float fields.

    Args:
        vocab: Vocabulary used for decoding.
        metadata: Dict from ``IndexDataset.__getitem__`` containing
            ``subject_id``, ``prediction_time``, and ``window_last_observed``
            (all as int64 microseconds).
        generated_token_ids: Sequence of token IDs produced by the model.

    Returns:
        List of row dicts with keys matching ``_TRAJECTORY_SCHEMA``.
    """
    interval_means = vocab.interval_estimates["mean"]
    current_time_us: int = metadata["window_last_observed"]
    rows = []
    for tok_id in generated_token_ids:
        tok_str = vocab.decode(tok_id)
        delta_us = interval_means.get(tok_str, 0)
        if delta_us:
            current_time_us = int(current_time_us + delta_us)
        rows.append(
            {
                "subject_id": metadata["subject_id"],
                "time": current_time_us,
                "prediction_time": metadata["prediction_time"],
                "code": tok_str,
                "numeric_value": None,
            }
        )
    return rows


def _flush_chunk(out_paths: list[Path | None], chunk_rows: list[list[dict]]) -> None:
    """Write accumulated rows to their parquet files and clear the lists.

    Writes accumulated rows to their parquet files and clears the lists.
    If the file already exists, reads it and concatenates before writing.
    """
    for traj_idx, (out_fp, rows) in enumerate(zip(out_paths, chunk_rows)):
        if out_fp is not None and rows:
            df = pl.DataFrame(rows, schema=_TRAJECTORY_SCHEMA).with_columns(
                pl.col("time").cast(pl.Datetime("us")),
                pl.col("prediction_time").cast(pl.Datetime("us")),
            )
            table = df.to_arrow().cast(_PA_SCHEMA)
            if out_fp.exists():
                existing = pq.read_table(out_fp)
                table = pa.concat_tables([existing, table])
            pq.write_table(table, out_fp)
        chunk_rows[traj_idx] = []


def generate_trajectories(
    model_fp: Path | str,
    input_dir: Path | str,
    index_df: pl.DataFrame,
    output_dir: Path | str,
    n_trajectories: int = 20,
    time_limit: timedelta = timedelta(days=365.25 * 2),
    device: str = "cuda",
    temperature: float = 1.0,
    no_compile: bool = False,
    do_overwrite: bool = False,
    chunk_size: int = 500,
) -> None:
    """Generate trajectories for every row in *index_df* using an ethos model.

    For each ``(subject_id, prediction_time)`` row, *n_trajectories* independent
    continuations are sampled from the model, starting from the last event in
    the patient's record that precedes ``prediction_time``.

    Results are written as::

        output_dir/
            0.parquet   ← all samples, trajectory repetition 0
            1.parquet   ← all samples, trajectory repetition 1
            ...
            {n_trajectories-1}.parquet

    Each file's schema matches ``GeneratedTrajectorySchema`` from
    ``MEDS_trajectory_evaluation``:
    ``subject_id`` (Int64), ``time`` (Datetime[μs]),
    ``prediction_time`` (Datetime[μs]), ``code`` (Utf8),
    ``numeric_value`` (Float32, always null).

    Rows are streamed to disk every *chunk_size* samples so that memory usage
    stays bounded regardless of dataset size.

    Args:
        model_fp: Path to the ethos model checkpoint (``.pt`` file).
        input_dir: Path to the ethos tokenized dataset directory.
        index_df: DataFrame with columns ``subject_id`` (Int64) and
            ``prediction_time`` (Datetime or Int64 microseconds).
        output_dir: Directory to write output parquet files (created if absent).
        n_trajectories: Number of trajectory repetitions per sample.
        time_limit: Maximum simulated time per trajectory.
        device: Torch device string (e.g. ``"cuda"``, ``"cuda:0"``, ``"cpu"``).
        temperature: Sampling temperature.
        no_compile: If True, skip ``torch.compile``.
        do_overwrite: If True, overwrite existing output files.
        chunk_size: Number of samples to process before flushing rows to disk.
    """
    model_fp = Path(model_fp)
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up dtype and autocast
    dtype = "bfloat16" if "cuda" in device else "float32"
    autocast_ctx = setup_torch(device, dtype=dtype)
    if "cuda" in device:
        th.set_float32_matmul_precision("high")

    # Load model
    logger.info(f"Loading model from {model_fp}")
    model, _ = load_model_checkpoint(model_fp, map_location=device)
    model.to(device)
    model.eval()
    model = th.compile(model, disable=no_compile)
    n_positions = model.config.n_positions

    # Build dataset from index
    logger.info(f"Loading dataset from {input_dir}")
    dataset = IndexDataset(
        input_dir=input_dir,
        index_df=index_df,
        n_positions=n_positions,
        time_limit=time_limit,
    )
    n_dropped = len(index_df) - len(dataset)
    logger.info(
        f"{len(dataset):,} valid samples ({n_dropped:,} dropped — subject absent or "
        "no events before prediction_time)"
    )

    vocab = dataset.vocab
    ctx_size = dataset.context_size
    stop_token_strs = set(dataset.stop_stokens)
    time_limit_us = time_limit.total_seconds() * 1e6

    # Build per-trajectory output paths; delete stale files when overwriting.
    out_paths: list[Path | None] = []
    for traj_idx in range(n_trajectories):
        out_fp = output_dir / f"{traj_idx}.parquet"
        if out_fp.exists() and not do_overwrite:
            logger.info(f"Skipping existing {out_fp} (set do_overwrite=True to regenerate)")
            out_paths.append(None)
        else:
            if out_fp.exists():  # do_overwrite=True; remove so first flush starts fresh
                out_fp.unlink()
            out_paths.append(out_fp)

    chunk_rows: list[list[dict]] = [[] for _ in range(n_trajectories)]

    for sample_idx in tqdm(range(len(dataset)), desc="Generating trajectories", unit="sample"):
        x, metadata = dataset[sample_idx]
        trajectories = _generate_for_sample(
            model=model,
            x=x,
            vocab=vocab,
            n_positions=n_positions,
            ctx_size=ctx_size,
            stop_token_strs=stop_token_strs,
            time_limit_us=time_limit_us,
            n_trajectories=n_trajectories,
            device=device,
            autocast_ctx=autocast_ctx,
            temperature=temperature,
        )
        for traj_idx, tok_ids in enumerate(trajectories):
            if out_paths[traj_idx] is not None:
                chunk_rows[traj_idx].extend(_format_trajectory(vocab, metadata, tok_ids))

        if (sample_idx + 1) % chunk_size == 0:
            _flush_chunk(out_paths, chunk_rows)
            logger.debug(f"Flushed chunk at sample {sample_idx + 1:,}")

    # Flush any remaining rows
    _flush_chunk(out_paths, chunk_rows)

    n_written = sum(1 for p in out_paths if p is not None)
    logger.info(f"Trajectory generation complete. Wrote {n_written} parquet files to {output_dir}")
