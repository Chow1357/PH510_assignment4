#!/bin/bash

# Propagate environment variables to the compute node
#SBATCH --export=ALL

# Run in the standard partition (queue)
#SBATCH --partition=teaching

# Distribute processes in round-robin fashion
#SBATCH --distribution=block:block

# No of cores required (max. of 16, 4GB RAM per core)
#SBATCH --ntasks=16

# Runtime (hard, HH:MM:SS)
#SBATCH --time=24:00:00

# Job name
#SBATCH --job-name=integral 

# Output file
#SBATCH --output=slurm-%j.out

# choose which version to load
module load mpi 
# Modify the line below to run your program  python3 Task1-code.py
cd /users/kwb21127/repos/PH510_assignment4

perf stat -e cycles,instructions,cache-misses \
mpirun -np 16 /opt/software/anaconda/python-3.10.9/bin/python task_3_changes.py

