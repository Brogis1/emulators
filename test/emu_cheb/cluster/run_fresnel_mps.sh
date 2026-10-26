#!/bin/bash
# Template for pasqal-dgx (Slurm): adapt the /work/isokolov/heisenberg_cheb paths (venv, repo copy, FRESNEL spec).
#SBATCH --job-name=fresnel_mps
#SBATCH --partition=internal,cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --array=0-39%20
#SBATCH --output=logs/fresnel_mps_%A_%a.log
module purge; module load devel/python/3.12.11
source /work/isokolov/heisenberg_cheb/.venv/bin/activate
cd /work/isokolov/heisenberg_cheb/emulators  # repo root (emu_cheb installed: pip install -e ci/emu_cheb)
export FRESNEL_SPECS=/work/isokolov/heisenberg_cheb/fresnel_specs.json CUDA_VISIBLE_DEVICES=""
cases=(fresnel_relaxation fresnel_full)
case=${cases[$((SLURM_ARRAY_TASK_ID / 20))]}
python test/emu_cheb/validation/fresnel_mps_task.py $case $((SLURM_ARRAY_TASK_ID % 20)) /work/isokolov/heisenberg_cheb/fresnel_mps
