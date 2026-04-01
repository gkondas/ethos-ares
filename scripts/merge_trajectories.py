"""Merge per-rank trajectory parquets produced by a SLURM array job.

After running generate-trajectories.sh with --array=0-N, each rank writes to:
    output_dir/rank_{i}/0.parquet, rank_{i}/1.parquet, ...

This script merges them into:
    output_dir/0.parquet, output_dir/1.parquet, ...

Usage:
    python scripts/merge_trajectories.py <output_dir> [--delete-ranks]

Arguments:
    output_dir      Directory containing rank_* subdirectories.
    --delete-ranks  Delete rank_* subdirectories after merging.
"""

import argparse
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def merge(output_dir: Path, delete_ranks: bool) -> None:
    rank_dirs = sorted(output_dir.glob("rank_*"), key=lambda p: int(p.name.split("_")[1]))
    if not rank_dirs:
        print(f"No rank_* subdirectories found in {output_dir}")
        return

    print(f"Found {len(rank_dirs)} rank directories: {[d.name for d in rank_dirs]}")

    # Collect trajectory indices from the first rank dir
    traj_files = sorted(rank_dirs[0].glob("*.parquet"), key=lambda p: int(p.stem))
    if not traj_files:
        print(f"No parquet files found in {rank_dirs[0]}")
        return

    n_trajectories = len(traj_files)
    print(f"Merging {n_trajectories} trajectory files across {len(rank_dirs)} ranks...")

    for traj_file in traj_files:
        traj_idx = traj_file.stem
        out_fp = output_dir / f"{traj_idx}.parquet"

        tables = []
        for rank_dir in rank_dirs:
            fp = rank_dir / f"{traj_idx}.parquet"
            if fp.exists():
                tables.append(pq.read_table(fp))
            else:
                print(f"  Warning: missing {fp}")

        if tables:
            merged = pa.concat_tables(tables)
            pq.write_table(merged, out_fp)
            total_rows = sum(t.num_rows for t in tables)
            print(f"  {traj_idx}.parquet: {total_rows:,} rows from {len(tables)} ranks")

    if delete_ranks:
        for rank_dir in rank_dirs:
            shutil.rmtree(rank_dir)
        print(f"Deleted {len(rank_dirs)} rank directories.")

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--delete-ranks", action="store_true")
    args = parser.parse_args()
    merge(args.output_dir, args.delete_ranks)
