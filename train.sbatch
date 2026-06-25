#!/bin/bash
#SBATCH --job-name=ffs-01
#SBATCH --chdir=/home/urid/dev/Fast-FoundationStereo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --nodelist=vtg-ptk-dsk03
#SBATCH --signal=B:TERM@30             # SIGTERM to the batch shell 30s before kill

# sbatch train.sbatch
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ffs
#srun python3 scripts/finetune_isaac.py
#srun python3 scripts/benchmark_isaac_fs_rs.py
srun python3 scripts/benchmark_inbolt_trained_isaac.py

