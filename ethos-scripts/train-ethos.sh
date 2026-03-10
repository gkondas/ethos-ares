#!/bin/bash
#SBATCH --job-name=ethos-train
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gbk2114@cumc.columbia.edu

export HYDRA_FULL_ERROR=1

data_path="/users/gbk2114/data/ethos-tokenize"


cd $HOME/ethos-ares

model_name="$(date +%Y-%m-%d_%H-%M-%S)"

uv run torchrun --no_python --standalone --nproc_per_node=2 ethos_train \
  data_fp=$data_path/train \
  wandb_run_name="$model_name" \
  out_dir="${data_path}/models/${model_name}"


