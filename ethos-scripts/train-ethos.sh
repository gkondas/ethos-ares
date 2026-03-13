#!/bin/bash
#SBATCH --job-name=ethos-train
#SBATCH --output=logs/%x_%j.out
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
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

uv run torchrun --standalone --nproc_per_node=2 ethos_train \
  data_fp=$data_path/train \
  out_dir="${data_path}/models"


