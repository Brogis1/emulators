#!/bin/bash
# Template for pasqal-dgx (Slurm): adapt the /work/isokolov/heisenberg_cheb paths (venv, repo copy, FRESNEL spec).
#SBATCH --job-name=harness
#SBATCH --partition=cpu,internal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --array=0
#SBATCH --output=logs/harness_%A_%a.log
module purge; module load devel/python/3.12.11
source /work/isokolov/heisenberg_cheb/.venv/bin/activate
cd /work/isokolov/heisenberg_cheb/emulators  # repo root (emu_cheb installed: pip install -e ci/emu_cheb)
export FRESNEL_SPECS=/work/isokolov/heisenberg_cheb/fresnel_specs.json EMU_MPS_CACHE=/work/isokolov/heisenberg_cheb/emu_cache
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4
engines=(turbo)
e=${engines[$SLURM_ARRAY_TASK_ID]}
python -W ignore test/emu_cheb/validation/noisy_vs_emu_mps.py noiseless dephasing depolarizing relaxation eff_noise leakage readout fresnel_dephasing \
    fresnel_relaxation fresnel_readout fresnel_full local_dmm_noisy xy_slm_noisy heisenberg_local_noisy --traj 200 --engine $e
python -W ignore test/emu_cheb/validation/noisy_vs_emu_mps.py shaped local_dmm slm_rydberg xy_slm heisenberg_local --traj 1 --engine $e
