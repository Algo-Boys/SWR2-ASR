#!/bin/bash

#SBATCH --job-name=swr-teamprojekt
#SBATCH --partition=a100
#SBATCH --time=24:00:00

### Note: --gres=gpu:x should equal to ntasks-per-node
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=32gb
#SBATCH --chdir=/mnt/lustre/mladm/mfa252/ref/
#SBATCH --output=/mnt/lustre/mladm/mfa252/%x-%j.out

source venv/bin/activate

### the command to run
srun ./hpc_train.sh
