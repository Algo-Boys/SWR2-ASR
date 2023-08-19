#!/bin/bash

#SBATCH --job-name=swr-teamprojekt
#SBATCH --partition=a100
#SBATCH --time=00:30:00

### Note: --gres=gpu:x should equal to ntasks-per-node
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --chdir=/mnt/lustre/mladm/mfa252/SWR2-cool-projekt-main/
#SBATCH --output=/mnt/lustre/mladm/mfa252/%x-%j.out

source venv/bin/activate

### the command to run
srun ./hpc_train.sh
