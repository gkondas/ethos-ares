#!/bin/bash
#SBATCH --job-name=ethos-tokenize
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gbk2114@cumc.columbia.edu

HYDRA_FULL_ERROR=1

input_dir="/users/gbk2114/data/MIMIC_MEDS/MEDS_cohort/data"
output_dir="/users/gbk2114/data/ethos-tokenize"


cd $HOME/ethos-ares


# uv run ethos_tokenize -m worker='range(0,7)' \
#     input_dir=$input_dir/train \
#     output_dir=$output_dir \
#     out_fn=train
#
#
uv run ethos_tokenize -m worker='range(0,2)' \
    input_dir=$input_dir/tuning \
    vocab=$output_dir/train \
    output_dir=$output_dir \
    out_fn=tuning
#
# uv run ethos_tokenize -m worker='range(0,2)' \
#     input_dir=$input_dir/held_out \
#     vocab=$output_dir/train \
#     output_dir=$output_dir \
#     out_fn=test
