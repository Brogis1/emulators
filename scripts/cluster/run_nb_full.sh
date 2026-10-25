#!/bin/bash
# Template for pasqal-dgx (Slurm): adapt the /work/isokolov/heisenberg_cheb paths (venv, repo copy, FRESNEL spec).
#SBATCH --job-name=nb_full
#SBATCH --partition=cpu,internal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=slurm_%j.log
date; hostname
module purge; module load devel/python/3.12.11
source /work/isokolov/heisenberg_cheb/.venv/bin/activate
cd /work/isokolov/heisenberg_cheb/emulators/scripts
export FRESNEL_SPECS=/work/isokolov/heisenberg_cheb/fresnel_specs.json CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8
sed 's/FULL = False  #/FULL = True  #/' noisy_cheb_benchmark.ipynb > noisy_cheb_benchmark_full.ipynb
grep -c "FULL = True" noisy_cheb_benchmark_full.ipynb
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 noisy_cheb_benchmark_full.ipynb \
    --output noisy_cheb_benchmark_full.ipynb
date
