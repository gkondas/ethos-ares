from datetime import timedelta
from pathlib import Path

import hydra
import polars as pl
import torch as th
from loguru import logger
from omegaconf import DictConfig

from .generate_trajectories import generate_trajectories


@hydra.main(version_base=None, config_path="../configs", config_name="generate_trajectories")
def main(cfg: DictConfig) -> None:
    th.manual_seed(cfg.seed)

    index_fp = Path(cfg.index_fp)
    logger.info(f"Loading index from {index_fp}")
    index_df = pl.read_parquet(index_fp)

    generate_trajectories(
        model_fp=cfg.model_fp,
        input_dir=cfg.input_dir,
        index_df=index_df,
        output_dir=cfg.output_dir,
        n_trajectories=cfg.n_trajectories,
        time_limit=timedelta(days=cfg.time_limit_days),
        device=cfg.device,
        temperature=cfg.temperature,
        no_compile=cfg.no_compile,
        do_overwrite=cfg.do_overwrite,
        chunk_size=cfg.chunk_size,
    )


if __name__ == "__main__":
    main()
