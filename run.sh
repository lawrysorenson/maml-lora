#!/bin/bash

#SBATCH --time=24:00:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=40G   # memory per CPU core
#SBATCH --qos=cs
#SBATCH --cpus-per-task=2
#SBATCH -J "maml-lora"
# #SBATCH --partition=cs2


# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE

source ~/.bashrc
conda activate hug

python3 train.py