#!/bin/bash
# Template for pasqal-dgx (Slurm): adapt the /work/isokolov/heisenberg_cheb paths (venv, repo copy, FRESNEL spec).
#SBATCH --job-name=memcheck
#SBATCH --partition=internal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=slurm_%j.log
date; hostname; nvidia-smi --query-gpu=name --format=csv,noheader
module purge; module load devel/python/3.12.11
source /work/isokolov/heisenberg_cheb/.venv/bin/activate
cd /work/isokolov/heisenberg_cheb/emulators  # repo root (emu_cheb installed: pip install -e ci/emu_cheb)
export FRESNEL_SPECS=/work/isokolov/heisenberg_cheb/fresnel_specs.json
python -W ignore -m emu_cheb.solvers.noisy_cheb | tail -6
python -W ignore -m emu_cheb.solvers.heisenberg_cheb_fast | tail -4
python test/emu_cheb/cluster/gpu_memory_check.py
python test/emu_cheb/cluster/gpu_turbo_check.py | grep -c "vec=True"
date
