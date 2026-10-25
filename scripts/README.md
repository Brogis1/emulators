# Fast exact emulation of neutral-atom dynamics, with and without noise

Exact state-vector solvers for pulser sequences, built on the Chebyshev / Triton tricks of Mathieu Garrigues'
`rydberg_cheb_fast.py`. Without noise they are exact to round-off; with noise they run exact quantum trajectories
(no time step, no truncation), so the only error is the statistical one, $1/\sqrt{N_{traj}}$. They read the same
`SequenceData` that emu-base prepares for emu-mps, so every pulser noise channel and pulse feature arrives exactly as
in emu-mps, and every one of them has been checked against emu-mps (see [Validation](#validation)).

Limits: a state vector of $2^n$ amplitudes, i.e. up to n ≈ 26–27 atoms on a 40 GB A100 (emu-mps goes further when
entanglement stays low). The largest speed-ups are for constant Hamiltonians (quenches) and many trajectories.

## Files

| File | What it is |
|---|---|
| `rydberg_cheb_fast.py` | Mathieu Garrigues' noiseless Rydberg solver (quench, batched realizations, Triton kernel, memory planner, Lanczos guard). Two backward-compatible additions: a start vector for the series (`_series(start=...)`) and the ellipse parameter `rho` of the Bessel cut (`_bessel`, `_coefficients`). |
| `heisenberg_cheb_fast.py` | Noiseless Heisenberg/XYZ solver (`evolve`), Pauli-noise trajectories (`evolve_noisy`, dephasing/depolarizing only), Rydberg through the Heisenberg form (`rydberg_as_heisenberg`, `evolve_noisy_rydberg`). |
| `noisy_cheb.py` | **Reference** noisy engine: every pulser noise channel, any pulse, one trajectory at a time, written for clarity. The yardstick for the faster engines. |
| `noisy_cheb_fast.py` | Same trajectories, batched (one block for all trajectories of a sample), jumps detected at segment ends, cached Bessel tables, interaction part reused across drive steps, optional complex64. |
| `noisy_cheb_turbo.py` | **The engine to use.** Same trajectories again, plus per-row Hamiltonians (shot-to-shot samples batched), Lanczos bounds, batched jumps with Newton jump location, and a single-launch series kernel for small registers on GPU. |
| `noisy_vs_emu_mps.py` | The validation harness: every noise channel, pulse feature and FRESNEL's model against noisy emu-mps (`--engine reference|fast|turbo`). |
| `noisy_cheb_benchmark.ipynb` | Notebook (like `examples/emu_mps/heisenberg.ipynb`): exact state vector and exact Lindblad as targets, emu-mps and our engines, observable dynamics, correlators, timings. `FULL = False` (default, ~6 min) or `True` (n = 6, ~15 min+). |
| `heisenberg_benchmark.ipynb` | Noiseless benchmark: emu-mps vs Chebyshev with Gershgorin / Lanczos bounds, with an explanation of both. |
| `gpu_turbo_check.py`, `gpu_turbo_large.py` | GPU checks: turbo vs fast trajectory by trajectory, then timings (small registers / n = 14–20). |
| `fresnel_mps_task.py`, `fresnel_compare.py` | 2000-trajectory emu-mps job array for FRESNEL's model and the three-way comparison (emu-mps, ours, exact Lindblad). |
| `cluster/*.sh` | Slurm templates for pasqal-dgx (adapt the paths). |

## Which engine?

* Noiseless quench (constant H), Rydberg: `rydberg_cheb_fast.evolve`; Heisenberg/XYZ: `heisenberg_cheb_fast.evolve`.
  One Chebyshev series gives every output time; batched over realizations.
* Anything with noise, any pulse shape, local addressing, DMM, SLM, XY or Heisenberg couplings: `noisy_cheb_turbo`.
  `noisy_cheb` and `noisy_cheb_fast` give the same trajectories (same random streams) and stay as references.

## Quick start

```python
import numpy as np, pulser
import noisy_cheb_turbo as turbo
from noisy_cheb import readout

# any pulser sequence (global/local channels, DMM, SLM mask, XY channel)
reg = pulser.Register.rectangle(1, 6, spacing=7.0, prefix="q")
seq = pulser.Sequence(reg, pulser.devices.MockDevice)
seq.declare_channel("r", "rydberg_global")
seq.add(pulser.Pulse.ConstantPulse(2000, 2 * np.pi, 3.0, 0.0), "r")

noise = pulser.NoiseModel(relaxation_rate=0.1, dephasing_rate=0.2)   # or device.default_noise_model
res = turbo.simulate(seq, noise, n_trajectories=2000, eval_fractions=np.linspace(0, 1, 21), device="cuda")

occ = res.occupations.mean(0)                        # <n_i>(t), shape (T, n)
err = res.occupations.std(0) / np.sqrt(2000)         # standard error
conn = res.correlations - occ[:, :, None] * occ[:, None, :]   # connected <n_i n_j> - <n_i><n_j>
bits = readout(res.probabilities[-1], 0.15, 0.06)    # measured bitstring distribution with SPAM readout errors
```

Heisenberg / NMR-like couplings: build a `SequenceData` from any pulser sequence (local pulses, DMM chemical shifts)
and replace its interaction by a $3\times n\times n$ stack $[J^{xx}, J^{yy}, J^{zz}]$ — independent matrices, any
graph, no geometric profile imposed:

```python
import dataclasses, torch
from emu_base import HamiltonianType
J3 = torch.as_tensor(np.stack([J, J, J]))   # isotropic scalar couplings J_ij (rad/us)
to_heisenberg = lambda sd: dataclasses.replace(sd, interaction_matrix=lambda t: J3, hamiltonian_type=HamiltonianType.XY)
res = turbo.simulate(seq, noise, n_trajectories=2000, eval_fractions=fr, replace_sequence_data=to_heisenberg)
```

Other options: `initial_state="0101"` (basis state, atom 0 first, 0 = ground / "u"), `single=True` (complex64 for
statistics-level runs), `seed=...`, `bounds="gershgorin"`, `vectorized=False` (first validated turbo version).
Results: `res.probabilities` (mean $|\psi|^2$, all atoms; bad and leaked atoms in bit 0), `res.occupations` and
`res.leaked` per trajectory, `res.correlations` (mean), `res.stats` (jumps, matvecs, accuracy indicators).

## Conventions

* Units: frequencies in rad/µs, times in ns for sequences (µs inside the solvers). Atom 0 = most significant bit.
* Local basis as emu-mps: index 0 = ground ("g" / "u"), 1 = excited ("r" / "d"); $n = |1\rangle\langle 1|$.
* $H = \sum_i \frac{\Omega_i}{2}(\cos\phi_i X_i + \sin\phi_i Y_i) - \sum_i \delta_i n_i + \sum_k\sum_{i<j} M_k[i,j]\, O_k(i)O_k(j)$
  with $(M, O)$ = $(J, n)$ for Rydberg; $(C_3, X/\sqrt2), (C_3, Y/\sqrt2), (C_6, |0\rangle\langle0|)$ for pulser's XY
  channel (i.e. $\frac{C_3}{r^3}(1-3\cos^2\theta)\,\frac{XX+YY}{2}$ plus a van der Waals shift on $|uu\rangle$);
  $(J^{xx}, X), (J^{yy}, Y), (J^{zz}, Z)$ for a Heisenberg stack (which replaces the whole interaction).
* Drives are piecewise constant on emu-mps' time grid (`dt`, midpoint samples), so both codes integrate the very
  same discretized pulse.

## How it works (short)

* **Chebyshev series.** With $\tilde H = (H - c)/w$ (spectrum in $[-1,1]$),
  $e^{-iHt}\psi = e^{-ict}\sum_k (2-\delta_{k0})(-i)^k J_k(wt)\, T_k(\tilde H)\psi$, $T_{k+1} = 2\tilde H T_k - T_{k-1}$:
  one matrix-vector product per term, about $w t$ terms; only the coefficients depend on $t$, so one series serves
  every output time. The interval $[c-w, c+w]$ comes from Gershgorin (rigorous) or Lanczos (tighter, guarded).
* **Noise = quantum trajectories** (as emu-mps): between jumps $\psi$ evolves with $H_{\rm eff} = H - iG$,
  $G = \frac12\sum L^\dagger L$, and jumps when $\|\psi\|^2$ reaches a random threshold. $H_{\rm eff}$ is not Hermitian,
  but the same series still converges: its numerical range lies in an ellipse around $[-1,1]$, the series is cut
  where $|J_k|\rho^k$ is negligible, and every series is checked against the bound $\|T_k\psi\| \le (1+\sqrt2)\rho^k$
  (a violation stops the run instead of returning wrong numbers). Pauli noise keeps $\rho = 1$.
* **Jump times** to ~1e-15: interpolation of the norm between probe times (reference, fast) or Newton iteration with
  the exact derivative $dN/dt = -2\langle G\rangle$ (turbo on GPU).
* **Leakage**: the leaked atom is dark (nothing drives, couples to or leaves $|x\rangle$), so the trajectory continues
  without it. **Bad atoms** are removed, as in emu-mps. **Readout errors** are applied exactly to the probabilities.
* **Speed**: fused Triton step kernels (one memory pass per term), trajectories batched as rows of one block,
  shot-to-shot samples batched with per-row Hamiltonians, and for $2^n \le 4096$ a kernel that runs a whole series
  in one launch (one thread block per trajectory).

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

Everything below was rerun with each engine (`noisy_cheb`, `noisy_cheb_fast`, `noisy_cheb_turbo`):

* **Exactness.** Single trajectories against dense quantum jumps with the same random numbers: ≤4e-15
  (`python3 noisy_cheb.py`). Fast and turbo against the reference, trajectory by trajectory: ≤2e-14 on CPU and GPU,
  identical jump counts (`python3 noisy_cheb_fast.py`, `python3 noisy_cheb_turbo.py`, `gpu_turbo_check.py`).
  Noiseless pulse features (shaped pulses, local pulses + DMM, SLM mask, XY channel, Heisenberg with local pulses)
  against dense piecewise exponentials: ≤1.5e-15.
* **Lindblad limit.** Trajectory averages against the dense Lindblad equation: within 3 standard errors.
* **emu-mps** (`noisy_vs_emu_mps.py`, 3 atoms): every channel in the table above, the pulse features with and
  without noise, and FRESNEL's model (dephasing 0.222/µs, relaxation 0.01/µs, readout 0.15/0.06):
  deterministic cases agree trajectory by trajectory to 3e-15–8e-11 for quenches and shot-to-shot channels and
  up to 1e-6 for shaped / local / XY pulses — emu-mps' own TDVP error, ours being ≤1.5e-15 against dense
  exponentials; random cases agree within the statistics, with noise effects of 6–17 standard errors. FRESNEL with 2000 emu-mps trajectories
  (cluster array) and exact Lindblad: all three agree (max z ≤ 2.9), noise effect 39 standard errors.

How to rerun: the self-checks above (seconds to minutes, CPU); `python3 noisy_vs_emu_mps.py --engine turbo`
(emu-mps results of cases without shot-to-shot noise are cached in `EMU_MPS_CACHE`); the notebook; the GPU
checks through `cluster/*.sh`.

## Performance (measured, one A100-40GB unless noted)

Noiseless, Heisenberg XYZ chain, quench to 4 µs, 19 output times (`heisenberg_cheb_fast.evolve`):

| n | 16 | 20 | 22 | 24 | 26 |
|---|---|---|---|---|---|
| Gershgorin | 0.37 s | 0.46 s | 0.67 s | 1.7 s | 6.9 s |
| Lanczos | 0.05 s | 0.10 s | 0.30 s | 1.4 s | 6.6 s |

(emu-mps at 2 µs, 11 times: 68–129 s at n = 20–24, error 1e-5 to 1e-11.)

Noisy, FRESNEL noise model, per trajectory:

| | n = 8 | 10 | 12 | 16 | 20 |
|---|---|---|---|---|---|
| emu-mps (4 µs; 2 µs for n ≤ 12) | 34 s | 46 s | 62 s | 172 s | 241 s |
| noisy_cheb, reference (4 µs) | | | 0.29 s | 0.46 s | 1.96 s |
| noisy_cheb_turbo (2 µs) | 1.9 ms | 2.5 ms | 4.8 ms | | |

Shot-to-shot noise (amplitude, Doppler + Lindblad), 2 µs: turbo 8–14 ms per trajectory at n = 8–12, 12–18× faster
than noisy_cheb_fast (which cannot batch samples with different H).

## Known limitations and caveats

* State vector: n ≲ 26–27 on one 40 GB GPU.
* Time-dependent drives restart the series at every drive step (piecewise constant, as emu-mps): exact, but the
  one-series advantage of quenches is lost (~10–15 terms per 10 ns step).
* Lanczos bounds are used only for sequences with at most 8 constant pieces (otherwise the Lanczos work exceeds the
  savings); they pay off mainly from n ≈ 16 on.
* The single-launch kernel needs a GPU and $2^n \le 4096$ (n ≤ 12).
* Leakage: operators must stay in {g, r} or map {g, r} into x, never leave x.
* Suspected emu-base issue (affects emu-mps and us alike, since we use its operators): `eff_noise` 3×3 operators
  are converted from pulser's basis by flipping only their {g, r} block, so a leakage operator $|x\rangle\langle r|$
  arrives as $|x\rangle\langle g|$ (`emu_base/jump_lindblad_operators.py`).
* Statistical tests: early in a trajectory, population-changing jumps are rare events, and a small sample's own
  spread underestimates the error; the harness and the notebook take the spread from a large separate sample.
