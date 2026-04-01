from pathlib import Path

import hydra
from loguru import logger
from omegaconf import DictConfig

from .evaluate_trajectories import evaluate_trajectories


@hydra.main(version_base=None, config_path="../configs", config_name="evaluate_trajectories")
def main(cfg: DictConfig) -> None:
    codes = list(cfg.codes)
    logger.info(f"Evaluating {len(codes)} codes with duration={cfg.duration_days} days")

    evaluate_trajectories(
        trajectory_dir=cfg.trajectory_dir,
        input_dir=cfg.input_dir,
        codes=codes,
        duration_days=cfg.duration_days,
        output_fp=cfg.output_fp,
    )


if __name__ == "__main__":
    main()
