from datetime import timedelta
from pathlib import Path

import hydra
import polars as pl
from loguru import logger
from omegaconf import DictConfig
from sklearn.metrics import roc_auc_score


def load_trajectories(trajectories_dir: Path) -> dict[int, pl.DataFrame]:
    """Load all trajectory parquet files, keyed by rollout index."""
    trajectory_fps = sorted(trajectories_dir.glob("*.parquet"), key=lambda p: int(p.stem))
    logger.info(f"Found {len(trajectory_fps)} trajectory files in {trajectories_dir}")
    return {int(fp.stem): pl.read_parquet(fp) for fp in trajectory_fps}


def compute_code_probabilities(
    trajectories: dict[int, pl.DataFrame],
    codes: list[str],
    duration: timedelta,
) -> pl.DataFrame:
    """For each (subject_id, prediction_time, code), compute the fraction of rollouts
    where the code appears within `duration` of prediction_time."""
    n_rollouts = len(trajectories)
    logger.info(f"Computing probabilities across {n_rollouts} rollouts for {len(codes)} codes")

    code_hits = []
    for rollout_idx, df in trajectories.items():
        # Filter to target codes and within duration
        hits = df.filter(
            pl.col("code").is_in(codes)
            & (pl.col("time") <= pl.col("prediction_time") + duration)
        ).select("subject_id", "prediction_time", "code").unique()

        code_hits.append(hits.with_columns(pl.lit(1).alias("hit")))

    if not code_hits:
        raise ValueError("No code hits found in any trajectory rollout")

    # Build the full key space: all (subject_id, prediction_time) from rollout 0 x all codes
    sample_df = next(iter(trajectories.values()))
    keys = sample_df.select("subject_id", "prediction_time").unique()
    code_df = pl.DataFrame({"code": codes})
    full_keys = keys.join(code_df, how="cross")

    # Stack all hits, count per key
    all_hits = pl.concat(code_hits).group_by("subject_id", "prediction_time", "code").agg(
        pl.col("hit").sum().alias("n_hits")
    )

    result = full_keys.join(
        all_hits, on=["subject_id", "prediction_time", "code"], how="left"
    ).with_columns(
        (pl.col("n_hits").fill_null(0) / n_rollouts).alias("predicted_probability"),
    ).drop("n_hits")

    return result


def compute_aurocs(
    probabilities: pl.DataFrame,
    ground_truth: pl.DataFrame,
) -> pl.DataFrame:
    """Compute AUROC per code by joining predictions with ground truth."""
    merged = probabilities.join(
        ground_truth.select("subject_id", "prediction_time", "code", "boolean_value"),
        on=["subject_id", "prediction_time", "code"],
        how="inner",
    )

    n_matched = len(merged)
    n_pred = len(probabilities)
    if n_matched < n_pred:
        logger.warning(
            f"Only {n_matched}/{n_pred} predictions matched ground truth. "
            "Unmatched predictions are excluded from AUROC calculation."
        )

    results = []
    for code in merged.get_column("code").unique().sort().to_list():
        code_df = merged.filter(pl.col("code") == code)
        y_true = code_df.get_column("boolean_value").to_numpy()
        y_pred = code_df.get_column("predicted_probability").to_numpy()

        n = len(y_true)
        prevalence = y_true.mean()

        if y_true.sum() == 0 or y_true.sum() == n:
            logger.warning(f"Code '{code}': only one class present (n={n}, prevalence={prevalence:.4f}), skipping AUROC")
            auroc = None
        else:
            auroc = roc_auc_score(y_true, y_pred)

        results.append({"code": code, "n": n, "prevalence": prevalence, "auroc": auroc})

    return pl.DataFrame(results)


@hydra.main(
    config_path="../configs", config_name="evaluate_trajectories", version_base=None
)
def main(cfg: DictConfig):
    trajectories_dir = Path(cfg.trajectories_dir)
    ground_truth_fp = Path(cfg.ground_truth_fp)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    codes = list(cfg.codes)
    duration = timedelta(days=cfg.duration_days)

    logger.info(f"Evaluating {len(codes)} codes with duration={duration}")
    logger.info(f"Trajectories: {trajectories_dir}")
    logger.info(f"Ground truth: {ground_truth_fp}")

    # Load data
    trajectories = load_trajectories(trajectories_dir)
    ground_truth = pl.read_parquet(ground_truth_fp)

    # Compute predicted probabilities from Monte Carlo rollouts
    probabilities = compute_code_probabilities(trajectories, codes, duration)
    probabilities.write_parquet(output_dir / "predicted_probabilities.parquet")
    logger.info(f"Wrote predicted probabilities to {output_dir / 'predicted_probabilities.parquet'}")

    # Compute AUROC per code
    aurocs = compute_aurocs(probabilities, ground_truth)
    aurocs.write_parquet(output_dir / "aurocs.parquet")

    logger.info("AUROC results per code:")
    print(aurocs)


if __name__ == "__main__":
    main()
