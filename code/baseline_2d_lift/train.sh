#!/usr/bin/bash
#SBATCH --job-name=baseline
#SBATCH --account=EU-25-52
#SBATCH --partition=qgpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --output=myJob.out
#SBATCH --error=myJob.err

# ============================================
# 1. Setup ambiente
# ============================================
# Evita conflitti con moduli pre-caricati del cluster
module purge

# Attiva il tuo environment Conda
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate AR17

# (Opzionale) Mostra informazioni per debug
echo "Job started on $(hostname) at $(date)"
nvidia-smi
conda list | grep torch

# ============================================
# 2. Avvia l'addestramento
# ============================================
python AR17/code/baseline_2d_lift/test.py

# ============================================
# 3. Cleanup finale
# ============================================
echo "Job terminato alle $(date)"