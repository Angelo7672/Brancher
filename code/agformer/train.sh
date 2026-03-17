#!/usr/bin/bash
#SBATCH --job-name=brancher
#SBATCH --account=EU-25-52
#SBATCH --partition=qgpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --output=myJobtrain.out
#SBATCH --error=myJobtrain.err

# ============================================
# 1. Setup ambiente
# ============================================
module purge

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate AR17

# ============================================
# Info SLURM del job
# ============================================
echo "============================================"
echo "SLURM JOB INFO"
echo "Job ID        : $SLURM_JOB_ID"
echo "Job Name      : $SLURM_JOB_NAME"
echo "User          : $SLURM_JOB_USER"
echo "Node(s)       : $SLURM_JOB_NODELIST"
echo "NTasks        : $SLURM_NTASKS"
echo "CPUs per task : $SLURM_CPUS_PER_TASK"
echo "GPUs          : $SLURM_GPUS"
echo "Started on    : $(hostname)"
echo "Start time    : $(date)"
echo "============================================"

# Debug utile
nvidia-smi
conda list | grep torch

# ============================================
# 2. Avvia l'addestramento
# ============================================
python AR17/code/agformer/inference.py

# ============================================
# 3. Cleanup finale
# ============================================
echo "Job $SLURM_JOB_ID terminato alle $(date)"