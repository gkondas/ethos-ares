#!/bin/bash
#SBATCH --job-name=ethos-train
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gbk2114@cumc.columbia.edu

export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

data_path="/users/gbk2114/data/ethos-tokenize"


cd $HOME/ethos-ares


echo "HOSTNAME=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi -L

uv run ethos_train \
  data_fp=$data_path/train \
  out_dir="${data_path}/models"


