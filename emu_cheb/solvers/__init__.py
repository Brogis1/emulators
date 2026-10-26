"""The Chebyshev engines behind emu_cheb (README.md in this folder: algorithms, conventions, validation, timings).

rydberg_cheb_fast, heisenberg_cheb_fast: noiseless quench solvers (constant H). noisy_cheb (reference),
noisy_cheb_fast, noisy_cheb_turbo (fastest): quantum trajectories with emu-mps' full noise model.
Each module's self-check: python -m emu_cheb.solvers.<module>.
"""
