#!/bin/bash
# Template for pasqal-dgx (Slurm): adapt the /work/isokolov/heisenberg_cheb paths (venv, repo copy, FRESNEL spec).
#SBATCH --job-name=noisy_turbo
#SBATCH --partition=internal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_%j.log
date; hostname; nvidia-smi --query-gpu=name --format=csv,noheader
module purge; module load devel/python/3.12.11
source /work/isokolov/heisenberg_cheb/.venv/bin/activate
cd /work/isokolov/heisenberg_cheb/emulators/scripts
export FRESNEL_SPECS=/work/isokolov/heisenberg_cheb/fresnel_specs.json
python cluster/gpu_turbo_check.py
date
