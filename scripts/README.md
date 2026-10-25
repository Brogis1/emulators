# Fast exact emulation of neutral-atom dynamics, with and without noise

Exact state-vector solvers for pulser sequences, built on the Chebyshev / Triton tricks of Mathieu Garrigues'
`rydberg_cheb_fast.py`. Without noise they are exact to round-off; with noise they run exact quantum trajectories (no
time step, no truncation), so the only error is the statistical one, $1/\sqrt{N_{traj}}$. They read the same
`SequenceData` that emu-base prepares for emu-mps, so every pulser noise channel and pulse feature arrives exactly as
in emu-mps — and each has been checked against emu-mps ([Validation](#validation)).

Range: a state vector of $2^n$ amplitudes, up to n ≈ 26 atoms on a 40 GB A100 (emu-mps goes further when entanglement
stays low). Largest speed-ups: constant Hamiltonians (quenches) and many trajectories (noise statistics).

## Layout

```
scripts/
├── README.md
├── rydberg_cheb_fast.py      noiseless Rydberg quench solver (Mathieu Garrigues)
├── heisenberg_cheb_fast.py   noiseless Heisenberg / XYZ solver, Pauli-noise trajectories
├── noisy_cheb.py             noisy engine, REFERENCE (clarity first, one trajectory at a time)
├── noisy_cheb_fast.py        noisy engine, batched (validated against the reference)
├── noisy_cheb_turbo.py       noisy engine to USE (fastest; validated against fast and the reference)
├── noisy_vs_emu_mps.py       validation harness: every noise channel, pulse feature and FRESNEL vs emu-mps
├── notebooks/                worked examples with exact targets, emu-mps comparison, timings
├── cluster/                  pasqal-dgx: Slurm templates, persistent GPU test runner, GPU / FRESNEL checks
└── experiments/              ideas tried but not adopted (yet): tiled kernel, multi-GPU driver
```

| Path | What it is |
|---|---|
| `rydberg_cheb_fast.py` | Mathieu Garrigues' noiseless Rydberg solver: quench, batched realizations, Triton kernel, memory planner, Lanczos guard. Two backward-compatible additions: a start vector for the series (`_series(start=...)`) and the ellipse parameter `rho` of the Bessel cut (`_bessel`, `_coefficients`). |
| `heisenberg_cheb_fast.py` | Noiseless Heisenberg/XYZ solver (`evolve`), Pauli-noise trajectories (`evolve_noisy`: dephasing/depolarizing only), Rydberg in Heisenberg form (`rydberg_as_heisenberg`, `evolve_noisy_rydberg`). |
| `noisy_cheb.py` | Reference noisy engine: all pulser noise channels, any pulse, one trajectory at a time. The yardstick of the others. |
| `noisy_cheb_fast.py` | Same trajectories, batched: one block for all trajectories of a sample, jumps detected at segment ends, cached Bessel tables, interaction part reused across drive steps, optional complex64. |
| `noisy_cheb_turbo.py` | Same trajectories, plus: per-row Hamiltonians (shot-to-shot samples batched), Lanczos bounds, batched jumps, Newton jump location and a single-launch series kernel for small registers on GPU. |
| `noisy_vs_emu_mps.py` | Harness: `python3 noisy_vs_emu_mps.py [CASE ...] --engine reference|fast|turbo`. |
| `notebooks/noisy_cheb_benchmark.ipynb` | Like `examples/emu_mps/heisenberg.ipynb`: exact state vector and exact Lindblad as targets, emu-mps and our engines, observable dynamics, NMR-like local control with connected correlators, FRESNEL's model, timings. `FULL = False` (n = 4, ~6 min) or `True` (n = 6). |
| `notebooks/noisy_cheb_benchmark_full.ipynb` | The `FULL = True` run, executed on the cluster (CPU). |
| `notebooks/heisenberg_benchmark.ipynb` | Noiseless: emu-mps vs Chebyshev with Gershgorin / Lanczos bounds, with an explanation of both. |
| `cluster/run_*.sh` | Slurm templates (adapt the `/work/isokolov/heisenberg_cheb` paths): harness, GPU checks, memory check, FRESNEL array, full notebook. |
| `cluster/runner.py` | Persistent GPU test runner (see [Fast iteration on the cluster](#fast-iteration-on-the-cluster)). |
| `cluster/gpu_turbo_check.py` | turbo vs fast trajectory by trajectory on GPU (small n), then timings. |
| `cluster/gpu_turbo_large.py` | turbo vs fast at n = 14–20: exactness at n = 16, timings and matvec counts. |
| `cluster/gpu_memory_check.py` | Peak GPU memory of every engine at n = 20–26, runs repeated in one process. |
| `cluster/fresnel_mps_task.py`, `cluster/fresnel_compare.py` | 2000-trajectory emu-mps job array for FRESNEL's model; three-way comparison emu-mps / ours / exact Lindblad. |
| `experiments/tiled_step.py` | IO-aware tiled step kernel prototype: correct, but slower than the current kernel in Triton. |
| `experiments/multi_gpu.py` (+ `multi_gpu_check.py`) | Trajectories split over several GPUs, one process per GPU; identical results to one GPU. |
| `experiments/bench_large_interleaved.py` | fast vs turbo interleaved at large n, time split by component. |

Scripts in subfolders put `scripts/` on the import path themselves; run everything from `scripts/`
(e.g. `python3 cluster/gpu_turbo_check.py`), notebooks from `notebooks/`.

## Which engine?

* Noiseless quench (constant H), Rydberg: `rydberg_cheb_fast.evolve`; Heisenberg/XYZ: `heisenberg_cheb_fast.evolve`.
  One Chebyshev series gives every output time; batched over realizations.
* Anything with noise, pulse shapes, local addressing, DMM, SLM, XY or Heisenberg couplings: **`noisy_cheb_turbo`**.
  `noisy_cheb` and `noisy_cheb_fast` give the same trajectories (same random streams) and stay as references.
* Statistics-level runs (errors ~1e-2): `single=True` (complex64), ~1.8–1.9× faster from n ≈ 20.

## Quick start

```python
import numpy as np, pulser
import noisy_cheb_turbo as turbo
from noisy_cheb import readout

reg = pulser.Register.rectangle(1, 6, spacing=7.0, prefix="q")      # any pulser sequence: global / local channels,
seq = pulser.Sequence(reg, pulser.devices.MockDevice)                # DMM, SLM mask, XY channel
seq.declare_channel("r", "rydberg_global")
seq.add(pulser.Pulse.ConstantPulse(2000, 2 * np.pi, 3.0, 0.0), "r")

noise = pulser.NoiseModel(relaxation_rate=0.1, dephasing_rate=0.2)   # or device.default_noise_model
res = turbo.simulate(seq, noise, n_trajectories=2000, eval_fractions=np.linspace(0, 1, 21), device="cuda")

occ = res.occupations.mean(0)                                 # <n_i>(t), shape (T, n)
err = res.occupations.std(0) / np.sqrt(2000)                  # standard error
conn = res.correlations - occ[:, :, None] * occ[:, None, :]  # connected <n_i n_j> - <n_i><n_j>
bits = readout(res.probabilities[-1], 0.15, 0.06)            # measured bitstrings with SPAM readout errors
```

Heisenberg / NMR-like couplings: replace the interaction of the samples by a $3\times n\times n$ stack
$[J^{xx}, J^{yy}, J^{zz}]$ — independent matrices, any graph, no geometric profile imposed (pulser's own XY channel is
$\frac{C_3}{r^3}(1-3\cos^2\theta)\frac{XX+YY}{2}$ plus a $C_6/r^6$ shift on $|uu\rangle$; the stack replaces all of it):

```python
import dataclasses, torch
from emu_base import HamiltonianType
J3 = torch.as_tensor(np.stack([J, J, J]))   # e.g. isotropic scalar couplings J_ij (rad/us)

def to_heisenberg(sd):                       # module-level function (picklable, e.g. for experiments/multi_gpu.py)
    return dataclasses.replace(sd, interaction_matrix=lambda t: J3, hamiltonian_type=HamiltonianType.XY)

res = turbo.simulate(seq, noise, n_trajectories=2000, eval_fractions=fr, replace_sequence_data=to_heisenberg)
```

Options: `initial_state="0101"` (basis state, atom 0 first, 0 = ground / "u"), `single=True`, `seed=...`,
`bounds="gershgorin"`, `small_kernel=False`, `vectorized=False` (first validated turbo version), `memory_bytes=...`.
Results: `res.probabilities` (mean $|\psi|^2$ over all atoms; bad and leaked atoms in bit 0), `res.occupations` and
`res.leaked` per trajectory, `res.correlations` (mean $\langle n_in_j\rangle$), `res.stats` (jumps, matvecs, accuracy
indicators such as `jump_norm_dev` and `max_growth`).

## Conventions

* Units: frequencies in rad/µs, sequence times in ns (µs inside the solvers). Atom 0 = most significant bit.
* Local basis as emu-mps: index 0 = ground ("g" / "u"), 1 = excited ("r" / "d"); $n = |1\rangle\langle1|$.
* $H = \sum_i \frac{\Omega_i}{2}(\cos\phi_i X_i + \sin\phi_i Y_i) - \sum_i \delta_i n_i + \sum_k\sum_{i<j} M_k[i,j]\, O_k(i)O_k(j)$
  with $(M, O) = (J, n)$ for Rydberg; $(C_3, X/\sqrt2), (C_3, Y/\sqrt2), (C_6, |0\rangle\langle0|)$ for pulser's XY
  channel; $(J^{xx}, X), (J^{yy}, Y), (J^{zz}, Z)$ for a Heisenberg stack.
* Drives are piecewise constant on emu-mps' time grid (`dt`, midpoint samples): both codes integrate the very same
  discretized pulse.

## How it works (short)

* **Chebyshev series.** With $\tilde H = (H - c)/w$ (spectrum in $[-1,1]$),
  $e^{-iHt}\psi = e^{-ict}\sum_k (2-\delta_{k0})(-i)^k J_k(wt)\, T_k(\tilde H)\psi$, $T_{k+1} = 2\tilde H T_k - T_{k-1}$:
  one matrix-vector product per term, about $wt$ terms; only the coefficients depend on $t$, so one series serves every
  output time. $[c-w, c+w]$ comes from Gershgorin (rigorous) or Lanczos (tighter, guarded).
* **Noise = quantum trajectories** (as emu-mps): between jumps $\psi$ evolves with $H_{\rm eff} = H - iG$,
  $G = \frac12\sum L^\dagger L$, and jumps when $\|\psi\|^2$ reaches a random threshold. $H_{\rm eff}$ is not
  Hermitian, but the same series converges: its numerical range lies in an ellipse around $[-1,1]$, the series is cut
  where $|J_k|\rho^k$ is negligible, and every series is checked against $\|T_k\psi\| \le (1+\sqrt2)\rho^k\|\psi\|$
  (a violation stops the run instead of returning wrong numbers). Pauli noise keeps $\rho = 1$.
* **Jump times** to ~1e-15: interpolation of the norm between probe times (reference, fast, turbo at large n) or Newton
  iteration with the exact derivative $dN/dt = -2\langle G\rangle$ (turbo, small registers on GPU).
* **Leakage**: the leaked atom is dark (nothing drives, couples to or leaves $|x\rangle$), so the trajectory continues
  without it. **Bad atoms** are removed, as in emu-mps. **Readout errors** are applied exactly to the probabilities.
* **Speed**: fused Triton step kernels (one launch per Chebyshev term), trajectories batched as rows of one block,
  shot-to-shot samples batched with per-row Hamiltonians, and for $2^n \le 4096$ a kernel that runs a whole series in
  one launch (one thread block per trajectory).
* **Memory**: batch sizes from 40 % of the free GPU memory (after emptying PyTorch's cache); caches of $2^n$ vectors
  bounded to 1 GB; no view keeps a large block alive; the reference engine stops with a clear `MemoryError` when its
  probe block does not fit (n ≈ 25+).

## Noise channels

| pulser channel | how | checked against emu-mps |
|---|---|---|
| dephasing, depolarizing | Lindblad jumps ($L^\dagger L \propto 1$) | yes (statistical) |
| relaxation | Lindblad jumps, non-Hermitian $H_{\rm eff}$ | yes |
| eff_noise (any 2×2 operators) | Lindblad jumps | yes |
| leakage (qutrit) | jumps into a dark state | yes |
| SPAM state preparation | bad atoms per sample (from PulserData) | yes (per trajectory) |
| SPAM readout | exact classical channel on P | yes (bitstrings) |
| amplitude, laser waist, detuning, high-frequency detuning, Doppler, register | per-sample drives / couplings (from PulserData) | yes (per trajectory) |
| hyperfine dephasing | digital basis only: not supported (nor by emu-mps) | – |

## Validation

All of it rerun with each engine (`noisy_cheb`, `noisy_cheb_fast`, `noisy_cheb_turbo`):

* **Exactness.** Single trajectories against dense quantum jumps with the same random numbers: ≤4e-15
  (`python3 noisy_cheb.py`). Fast and turbo against the reference, trajectory by trajectory: ≤2e-14 on CPU and GPU,
  identical jump counts (`python3 noisy_cheb_fast.py`, `python3 noisy_cheb_turbo.py`, `cluster/gpu_turbo_check.py`,
  `cluster/gpu_turbo_large.py`). Noiseless pulse features (shaped pulses, local pulses + DMM, SLM mask, XY channel,
  Heisenberg with local pulses) against dense piecewise exponentials: ≤1.5e-15.
* **Lindblad limit.** Trajectory averages against the dense Lindblad equation: within 3 standard errors.
* **emu-mps** (`noisy_vs_emu_mps.py`, 3 atoms): every channel above, the pulse features with and without noise,
  FRESNEL's model (dephasing 0.222/µs, relaxation 0.01/µs, readout 0.15/0.06). Deterministic cases agree trajectory
  by trajectory to 3e-15–8e-11 (quenches, shot-to-shot channels) and up to 1e-6 for shaped / local / XY pulses —
  emu-mps' own TDVP error, ours being ≤1.5e-15 against dense exponentials. Random cases agree within the statistics
  (max z ≤ 3.0 over 33 values), with noise effects of 6–18 standard errors. FRESNEL with 2000 emu-mps trajectories
  (cluster array) and exact Lindblad: all three agree (max z ≤ 2.9), noise effect 39 standard errors.
* **Memory** (`cluster/gpu_memory_check.py`, A100-40GB, time-dependent drive): peak 15 GiB (reference, n = 24),
  20–21 GiB (fast / turbo, n = 24, 8 trajectories), 34 GiB (`heisenberg_cheb_fast.evolve`, n = 26); no growth across
  repeated runs.

Statistical tests take the per-trajectory spread from a large separate sample: early in a trajectory,
population-changing jumps are rare events and a small sample's own spread underestimates the error.

How to rerun: the self-checks (`python3 <engine>.py`, seconds to minutes, CPU); `python3 noisy_vs_emu_mps.py --engine
turbo` (emu-mps results of cases without shot-to-shot noise are cached in `EMU_MPS_CACHE`); the notebooks; the GPU
checks through `cluster/run_*.sh`.

## Performance (measured, one A100-40GB unless noted)

Noiseless, Heisenberg XYZ chain, quench to 4 µs, 19 output times (`heisenberg_cheb_fast.evolve`):

| n | 16 | 20 | 22 | 24 | 26 |
|---|---|---|---|---|---|
| Gershgorin | 0.37 s | 0.46 s | 0.67 s | 1.7 s | 6.9 s |
| Lanczos | 0.05 s | 0.10 s | 0.30 s | 1.4 s | 6.6 s |

(emu-mps at 2 µs, 11 times: 68–129 s at n = 20–24, error 1e-5 to 1e-11.)

Noisy, FRESNEL noise model, per trajectory:

| n | emu-mps | noisy_cheb (reference) | noisy_cheb_fast | noisy_cheb_turbo |
|---|---|---|---|---|
| 8 (2 µs) | 34 s | 0.11 s | 4.7 ms | 1.5 ms |
| 10 (2 µs) | 46 s | 0.12 s | 6.3 ms | 1.8 ms |
| 12 (2 µs) | 62 s | 0.16 s | 9.3 ms | 4.0 ms |
| 14 (4 µs) | | | 36 ms | 22 ms |
| 16 (4 µs) | 172 s | 0.46 s | 91 ms | 68 ms |
| 18 (4 µs) | | | 312 ms | 276 ms |
| 20 (4 µs) | 241 s | 1.96 s | 1.33 s | 1.25 s |

Shot-to-shot noise (amplitude, Doppler + Lindblad), 2 µs, n = 8–12: turbo 9.6–13 ms per trajectory, 12–14× faster than
noisy_cheb_fast (which cannot batch samples with different H). From n ≈ 18 both are limited by memory traffic of the
same kernels; turbo keeps 5–15 % (Lanczos bounds, cheaper jumps).

Further factors (measured): `single=True` 1.37× (n = 10), 1.77× (n = 20), 1.92× (n = 22) end-to-end, means within
~5e-7 of complex128; two GPUs (`experiments/multi_gpu.py`) 1.62× wall / 1.79× compute on an 80 s run (n = 18), plus
~4.5 s of worker start-up.

## Experiments (`experiments/`)

* **Tiled step kernel** (`tiled_step.py`): handles groups of bits per thread block so that all their flips happen in
  registers (2–3 passes over the state per step instead of ~12). Correct (2e-16 / 1e-7), but slower in Triton (best
  0.64 vs 0.52 ms per step at n = 22): in-tile permutations (reshape + flip, `tl.gather`) cost more than the DRAM
  traffic saved. A gain needs a CUDA kernel with an explicit shared-memory tile (cuStateVec-style).
* **Multi-GPU** (`multi_gpu.py`): one process per GPU, same per-trajectory random streams, identical results;
  worthwhile for runs of a minute or more (SequenceData must be picklable).

## Fast iteration on the cluster

`cluster/runner.py` keeps torch / pulser / emu-mps imported, Triton kernels compiled and test data cached on a GPU
allocation, and runs task files dropped into its inbox (modules reloaded when changed): a test takes seconds instead of
a batch job. Start it inside tmux (release the allocation when done):

```
srun -p internal -G2 -c 16 --mem 96G --time 02:00:00 python -u cluster/runner.py /work/<user>/runner
```

Tasks are Python files executed in the runner namespace (`G` persists across tasks); upload them under a temporary
name and rename into `<dir>/inbox/` (the runner must not read a half-written file); output appears in
`<dir>/outbox/<name>.out`, then `<name>.done`. Do not start multiprocessing from a task: run such scripts as a
subprocess (`experiments/multi_gpu_check.py`).

## Known limitations and caveats

* State vector: n ≲ 26 on one 40 GB GPU.
* Time-dependent drives restart the series at every drive step (piecewise constant, as emu-mps): exact, but the
  one-series advantage of quenches is lost (~10–15 terms per 10 ns step).
* Lanczos bounds only for sequences with at most 8 constant pieces (otherwise the Lanczos work exceeds the savings).
* The single-launch kernel needs a GPU and $2^n \le 4096$ (n ≤ 12).
* Leakage: operators must stay in {g, r} or map {g, r} into x, never leave x.
* Suspected emu-base issue (affects emu-mps and us alike, since we use its operators): `eff_noise` 3×3 operators are
  converted from pulser's basis by flipping only their {g, r} block, so a leakage operator $|x\rangle\langle r|$
  arrives as $|x\rangle\langle g|$ (`emu_base/jump_lindblad_operators.py`).
* FRESNEL's device spec is fetched from Pasqal Cloud and kept outside the repository (`FRESNEL_SPECS`, default
  `~/.cache/pasqal/fresnel_specs.json`).
