"""Evaluate Monte Carlo trajectory rollouts against ground truth labels.

For each (subject_id, prediction_time) and a set of target codes, computes
the probability that each code appears within a time window across all
trajectory repetitions.  Then computes per-code AUROC against labels.

Memory-safe: reads one trajectory parquet at a time, accumulating only a
small hit-count DataFrame.

Usage example::

    from ethos.inference.evaluate_trajectories import evaluate_trajectories

    evaluate_trajectories(
        trajectory_dir="trajectories/merged",
        labels_fp="labels.parquet",
        codes=["MEDS_DEATH", "HOSPITAL_ADMISSION"],
        duration_days=30,
        output_fp="results.parquet",
    )
"""

from pathlib import Path

import polars as pl
from loguru import logger
from sklearn.metrics import roc_auc_score


def evaluate_trajectories(
    trajectory_dir: Path | str,
    labels_fp: Path | str,
    codes: list[str],
    duration_days: int | float,
    output_fp: Path | str,
) -> pl.DataFrame:
    """Compute per-code trajectory probabilities and AUROC.

    Args:
        trajectory_dir: Directory containing ``0.parquet`` through
            ``{n-1}.parquet`` trajectory files.
        labels_fp: Parquet file with columns ``subject_id``,
            ``prediction_time``, ``code``, ``boolean_value``.
        codes: List of code strings to evaluate.
        duration_days: Time window in days from ``prediction_time``.
        output_fp: Path to write the results parquet.

    Returns:
        DataFrame with columns ``(subject_id, prediction_time, code,
        probability, boolean_value)``.
    """
    trajectory_dir = Path(trajectory_dir)
    output_fp = Path(output_fp)

    # Load labels and filter to requested codes
    labels = pl.read_parquet(labels_fp).filter(pl.col("code").is_in(codes))
    logger.info(f"Loaded {len(labels):,} label rows for {len(codes)} codes from {labels_fp}")

    # Initialize accumulator from labels: unique (subject_id, prediction_time, code) with hit_count=0
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

        # Read single trajectory file
        traj = pl.read_parquet(traj_fp)

        # Filter: code in target set AND within time window
        hits = (
            traj.lazy()
            .filter(
                pl.col("code").is_in(code_set)
                & (pl.col("time") <= (pl.col("prediction_time") + duration))
            )
            .select("subject_id", "prediction_time", "code")
            .unique()
            .collect()
        )

        if len(hits) == 0:
            continue

        # Join with accumulator to increment hit_count
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

    # Compute probabilities
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

    # Write results
    output_fp.parent.mkdir(parents=True, exist_ok=True)
    results.write_parquet(output_fp)
    logger.info(f"Wrote {len(results):,} rows to {output_fp}")

    # Per-code AUROC
    for code in codes:
        code_df = results.filter(pl.col("code") == code)
        y_true = code_df["boolean_value"].to_numpy()
        y_pred = code_df["probability"].to_numpy()

        if len(y_true) == 0:
            logger.warning(f"  {code}: no samples")
            continue

        n_classes = len(set(y_true))
        if n_classes < 2:
            logger.warning(f"  {code}: single class (all {y_true[0]}) — cannot compute AUROC")
            continue

        auc = roc_auc_score(y_true, y_pred)
        prevalence = y_true.mean()
        logger.info(f"  {code}: AUROC={auc:.4f}  n={len(y_true):,}  prevalence={prevalence:.4f}")

    return results
