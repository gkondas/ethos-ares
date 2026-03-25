#!/bin/bash
#SBATCH --job-name=ethos-gen-traj
#SBATCH --output=logs/%x_%j.out
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gbk2114@cumc.columbia.edu

export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd $HOME/ethos-ares

echo "HOSTNAME=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi -L

uv run ethos_generate_trajectories "$@"
