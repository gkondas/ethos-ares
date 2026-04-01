#!/bin/bash
#SBATCH --job-name=ethos-eval-traj
#SBATCH --output=logs/%x_%j.out
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gbk2114@cumc.columbia.edu

export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd $HOME/ethos-ares

echo "HOSTNAME=$(hostname)"

uv run ethos_evaluate_trajectories "$@"
