#!/bin/bash
#SBATCH --job-name=ethos-train-multinode
#SBATCH --output=logs/%x_%j.out
#SBATCH --partition=gpu
#SBATCH --nodes=4
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gbk2114@cumc.columbia.edu

export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=^lo,docker

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29500
data_path="/users/gbk2114/data/ethos-tokenize"

cd $HOME/ethos-ares

echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "SLURM_NNODES=$SLURM_NNODES"
echo "SLURM_NODEID=$SLURM_NODEID"
echo "HOSTNAME=$(hostname)"
echo "MASTER_ADDR=$MASTER_ADDR"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi -L

srun bash -c "$HOME/.local/bin/uv run torchrun \
  --no_python \
  --nnodes=$SLURM_NNODES \
  --nproc_per_node=2 \
  --node_rank=\$SLURM_NODEID \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  ethos_train \
  data_fp=$data_path/train \
  out_dir=${data_path}/models"
