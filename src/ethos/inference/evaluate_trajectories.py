"""Evaluate Monte Carlo trajectory rollouts against ground truth labels.

For each (subject_id, prediction_time) and a set of target codes, computes
the probability that each code appears within a time window across all
trajectory repetitions.  Then computes per-code AUROC against labels.

Ground truth labels are derived automatically from the tokenized dataset on
disk: for each (subject_id, prediction_time) pair found in the trajectories,
we check whether each target code actually appears within ``duration_days``
in the real patient timeline.

Memory-safe: reads one trajectory parquet at a time for hit counting, and
processes tokenized shards one at a time for label generation.

Usage example::

    from ethos.inference.evaluate_trajectories import evaluate_trajectories

    evaluate_trajectories(
        trajectory_dir="trajectories/merged",
        input_dir="data/tokenized/held_out",
        codes=["MEDS_DEATH", "HOSPITAL_ADMISSION"],
        duration_days=30,
        output_fp="results.parquet",
    )
"""

from pathlib import Path

import polars as pl
import torch as th
from loguru import logger
from safetensors import safe_open
from sklearn.metrics import roc_auc_score

from ethos.vocabulary import Vocabulary


def _extract_prediction_pairs(trajectory_dir: Path) -> pl.DataFrame:
    """Extract unique (subject_id, prediction_time) pairs from trajectory parquets.

    Returns a DataFrame with columns ``subject_id`` (Int64) and
    ``prediction_time`` (Int64, microseconds since epoch).
    """
    pairs = (
        pl.scan_parquet(trajectory_dir / "*.parquet")
        .select("subject_id", "prediction_time")
        .unique()
        .with_columns(pl.col("prediction_time").cast(pl.Datetime("us")).cast(pl.Int64))
        .collect()
    )
    logger.info(f"Found {len(pairs):,} unique (subject_id, prediction_time) pairs")
    return pairs


def _generate_labels(
    input_dir: Path,
    prediction_pairs: pl.DataFrame,
    codes: list[str],
    duration_us: int,
) -> pl.DataFrame:
    """Generate ground truth labels by scanning tokenized safetensors shards.

    For each (subject_id, prediction_time) pair, checks whether each target
    code appears in the real patient timeline within the time window
    ``(prediction_time, prediction_time + duration_us]``.

    Processes one shard at a time for memory efficiency.

    Returns:
        DataFrame with columns ``(subject_id, prediction_time, code, boolean_value)``.
    """
    vocab = Vocabulary.from_path(input_dir)
    target_token_ids = {vocab.stoi[c]: c for c in codes if c in vocab.stoi}
    missing = set(codes) - set(target_token_ids.values())
    if missing:
        logger.warning(f"Codes not found in vocabulary (skipped): {missing}")

    target_id_set = set(target_token_ids.keys())

    # Build lookup: subject_id -> list of prediction_times (Int64 microseconds)
    subject_to_pred_times: dict[int, list[int]] = {}
    for row in prediction_pairs.iter_rows(named=True):
        sid = row["subject_id"]
        subject_to_pred_times.setdefault(sid, []).append(row["prediction_time"])

    shard_fps = sorted(input_dir.glob("[0-9]*.safetensors"))
    if not shard_fps:
        raise FileNotFoundError(f"No safetensors shards found in {input_dir}")

    rows: list[dict] = []
    patients_found = 0

    for shard_fp in shard_fps:
        with safe_open(shard_fp, framework="pt") as f:
            patient_ids = f.get_tensor("patient_ids")
            patient_offsets = f.get_tensor("patient_offsets")
            shard_len = f.get_slice("tokens").get_shape()[0]

            # Compute end offset for each patient in this shard
            end_offsets = th.cat([patient_offsets[1:], th.tensor([shard_len])])

            for i in range(len(patient_ids)):
                sid = patient_ids[i].item()
                if sid not in subject_to_pred_times:
                    continue

                patients_found += 1
                start = patient_offsets[i].item()
                end = end_offsets[i].item()

                # Load this patient's tokens and times
                tokens = f.get_slice("tokens")[start:end]
                times = f.get_slice("times")[start:end]

                for pred_time in subject_to_pred_times[sid]:
                    end_time = pred_time + duration_us

                    # Binary search for the window (pred_time, end_time]
                    pred_t = th.tensor(pred_time)
                    end_t = th.tensor(end_time)
                    window_start = th.searchsorted(times, pred_t, right=True).item()
                    window_end = th.searchsorted(times, end_t, right=True).item()

                    if window_start >= end - start:
                        # No future events after prediction_time
                        codes_present = set()
                    else:
                        window_tokens = tokens[window_start:window_end]
                        codes_present = target_id_set & set(window_tokens.tolist())

                    for tid, code_str in target_token_ids.items():
                        rows.append(
                            {
                                "subject_id": sid,
                                "prediction_time": pred_time,
                                "code": code_str,
                                "boolean_value": tid in codes_present,
                            }
                        )

    n_expected = len(subject_to_pred_times)
    if patients_found < n_expected:
        logger.warning(
            f"{n_expected - patients_found} subjects from trajectories "
            f"not found in tokenized data"
        )

    labels = pl.DataFrame(rows).with_columns(
        pl.col("boolean_value").cast(pl.Int32),
    )
    n_pos = labels["boolean_value"].sum()
    logger.info(f"Generated {len(labels):,} labels ({n_pos:,} positive)")
    return labels


def evaluate_trajectories(
    trajectory_dir: Path | str,
    input_dir: Path | str,
    codes: list[str],
    duration_days: int | float,
    output_fp: Path | str,
) -> pl.DataFrame:
    """Compute per-code trajectory probabilities and AUROC.

    Ground truth labels are derived from the tokenized dataset: for each
    (subject_id, prediction_time) found in the trajectory files, we check
    whether each code in *codes* actually appears within *duration_days*
    in the real patient timeline stored in *input_dir*.

    Args:
        trajectory_dir: Directory containing ``0.parquet`` through
            ``{n-1}.parquet`` trajectory files.
        input_dir: Directory containing the tokenized dataset
            (safetensors shards, vocabulary, etc.).
        codes: List of code strings to evaluate.
        duration_days: Time window in days from ``prediction_time``.
        output_fp: Path to write the results parquet.

    Returns:
        DataFrame with columns ``(code, auroc, prevalence, n)``.
    """
    trajectory_dir = Path(trajectory_dir)
    input_dir = Path(input_dir)
    output_fp = Path(output_fp)

    duration_us = int(duration_days * 86400 * 1_000_000)

    # Extract unique (subject_id, prediction_time) pairs from trajectories
    prediction_pairs = _extract_prediction_pairs(trajectory_dir)

    # Generate ground truth labels from the tokenized dataset
    labels = _generate_labels(input_dir, prediction_pairs, codes, duration_us)
    logger.info(f"Generated {len(labels):,} label rows for {len(codes)} codes")

    # Initialize accumulator from labels
    accumulator = (
        labels.select("subject_id", "prediction_time", "code")
        .unique()
        .with_columns(hit_count=pl.lit(0, dtype=pl.Int32))
    )

    # Auto-detect trajectory files
    traj_files = sorted(trajectory_dir.glob("*.parquet"), key=lambda p: int(p.stem))
    n_trajectories = len(traj_files)
    if n_trajectories == 0:
        raise FileNotFoundError(f"No parquet files found in {trajectory_dir}")
    logger.info(f"Found {n_trajectories} trajectory files in {trajectory_dir}")

    duration = pl.duration(days=duration_days)
    code_set = set(codes)

    for traj_fp in traj_files:
        logger.info(f"Processing {traj_fp.name}")

        traj = pl.read_parquet(traj_fp)

        # Filter to target codes within time window, cast prediction_time to
        # Int64 microseconds to match the accumulator's key type.
        hits = (
            traj.lazy()
            .filter(
                pl.col("code").is_in(code_set)
                & (pl.col("time") <= (pl.col("prediction_time") + duration))
            )
            .with_columns(pl.col("prediction_time").cast(pl.Datetime("us")).cast(pl.Int64))
            .select("subject_id", "prediction_time", "code")
            .unique()
            .collect()
        )

        if len(hits) == 0:
            continue

        accumulator = (
            accumulator.lazy()
            .join(
                hits.lazy().with_columns(hit=pl.lit(1, dtype=pl.Int32)),
                on=["subject_id", "prediction_time", "code"],
                how="left",
            )
            .with_columns(
                hit_count=(pl.col("hit_count") + pl.col("hit").fill_null(0))
            )
            .drop("hit")
            .collect()
        )

    # Compute probabilities and join with ground truth
    results = (
        accumulator.with_columns(
            probability=(pl.col("hit_count") / n_trajectories)
        )
        .drop("hit_count")
        .join(
            labels.select("subject_id", "prediction_time", "code", "boolean_value"),
            on=["subject_id", "prediction_time", "code"],
            how="left",
        )
    )

    # Per-code AUROC and prevalence
    metrics_rows = []
    for code in codes:
        code_df = results.filter(pl.col("code") == code)
        y_true = code_df["boolean_value"].to_numpy()
        y_pred = code_df["probability"].to_numpy()

        if len(y_true) == 0:
            logger.warning(f"  {code}: no samples")
            continue

        prevalence = float(y_true.mean())
        n_classes = len(set(y_true))
        if n_classes < 2:
            logger.warning(f"  {code}: single class (all {y_true[0]}) — cannot compute AUROC")
            metrics_rows.append({"code": code, "auroc": None, "prevalence": prevalence, "n": len(y_true)})
            continue

        auc = roc_auc_score(y_true, y_pred)
        logger.info(f"  {code}: AUROC={auc:.4f}  n={len(y_true):,}  prevalence={prevalence:.4f}")
        metrics_rows.append({"code": code, "auroc": auc, "prevalence": prevalence, "n": len(y_true)})

    metrics = pl.DataFrame(metrics_rows)

    # Write metrics
    output_fp.parent.mkdir(parents=True, exist_ok=True)
    metrics.write_parquet(output_fp)
    logger.info(f"Wrote {len(metrics)} code metrics to {output_fp}")

    return metrics
