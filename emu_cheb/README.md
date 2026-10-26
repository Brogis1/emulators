# emu_cheb — an emu-sv-style backend for the Chebyshev engines

`emu_cheb` runs pulser sequences on exact Chebyshev engines behind the same interface as emu-sv and emu-mps:
`ChebBackend(sequence, config=ChebConfig(...)).run()` returns a pulser `Results`. The engines live in
[`emu_cheb.solvers`](solvers/README.md) (validated against emu-mps for every noise channel and FRESNEL's noise
model; algorithms, validation and timings there); `ChebBackend` only translates. Status: experimental.

Authors: Igor Sokolov & Mathieu Garrigues (contributor that created the basic approach that was generalized here to other Hamiltonians and implemented for noisy simulations)

## Speed

Exact, and much faster than emu-mps (measured; details and conditions in [solvers/README.md](solvers/README.md),
"Performance"). Noisy, FRESNEL's noise model, time per trajectory, one A100 (emu-mps: one trajectory on the GPU):

| n | emu-mps | emu_cheb (turbo) | speed-up |
|---|---|---|---|
| 8 (2 µs) | 34 s | 1.5 ms | ~23,000× |
| 12 (2 µs) | 62 s | 4.0 ms | ~15,000× |
| 16 (4 µs) | 172 s | 68 ms | ~2,500× |
| 20 (4 µs) | 241 s | 1.25 s | ~190× |

- **Noiseless quench** (A100): 0.10–1.4 s at n = 20–24 for 4 µs and 19 output times, where emu-mps takes 68–129 s
  for half the duration (2 µs, 11 times) with errors 1e-5–1e-11: ~90–700× faster, and exact.
- **CPU, small registers:** 0.4–0.5 ms per noisy trajectory at n = 3 against 0.7–2.7 s for emu-mps (validation
  harness). Against emu-sv (CPU, `examples/emu_cheb/`): 16-atom quench 0.43 s against 4.3 s; for a few atoms with
  Lindblad noise emu-sv's density matrix is cheaper than trajectories (6 atoms: 4 s against 8 s for 4000).
- **Where it comes from:** one Chebyshev series for all output times, fused Triton kernels, trajectories and
  shot-to-shot samples batched. turbo vs fast: 2.3–3.5× at n = 8–12 (Lindblad noise), 12–14× (shot-to-shot noise),
  fading to 1.07× at n = 20, where memory traffic dominates; `single=True` adds 1.8–1.9× at n = 20–22.

## Install

From a clone of this repository, like the other emulators (`dev_install.sh` installs every `ci/*` package):

```bash
pip install -e ci/emu_cheb          # or: uv pip install -e ci/emu_cheb
python -c "import emu_cheb; print(emu_cheb.__version__)"
```

It depends on emu-base and emu-sv (pulser, torch); Triton (shipped with CUDA builds of torch) enables the GPU kernels.
The validation harness also needs emu-mps (`pip install -e "ci/emu_cheb[validation]"` or `pip install -e ci/emu_mps`).

Every trajectory is propagated exactly (Chebyshev series: no time step, no Krylov or bond-dimension truncation);
noise follows emu-mps' model (PulserData samples, Monte Carlo wave functions). Range: a state vector of 2^n
amplitudes, up to n ≈ 26 on a 40 GB A100.

## Quick start

```python
import numpy as np, pulser
from emu_cheb import ChebBackend, ChebConfig, Occupation, CorrelationMatrix, BitStrings, connected

reg = pulser.Register.rectangle(1, 6, spacing=7.0, prefix="q")      # any pulser sequence: global / local channels,
seq = pulser.Sequence(reg, pulser.devices.MockDevice)                # DMM, SLM mask, XY channel
seq.declare_channel("r", "rydberg_global")
seq.add(pulser.Pulse.ConstantPulse(2000, 2 * np.pi, 3.0, 0.0), "r")

noise = pulser.NoiseModel(relaxation_rate=0.1, dephasing_rate=0.2, p_false_pos=0.15, p_false_neg=0.06)
times = np.linspace(0, 1, 21).tolist()
config = ChebConfig(observables=[Occupation(evaluation_times=times), CorrelationMatrix(evaluation_times=times),
                                 BitStrings(num_shots=1000)],
                    noise_model=noise, n_trajectories=2000)  # set n_trajectories for noise (pulser's default is 1)
backend = ChebBackend(seq, config=config)
results = backend.run()

results.occupation[-1]                                 # <n_i>(T), as emu-sv
connected(results.occupation[-1], results.correlation_matrix[-1])  # <n_i n_j> - <n_i><n_j>
results.final_bitstrings                               # 2000 x 1000 shots, readout errors included
backend.raw.occupations.std(0) / np.sqrt(2000)         # error bars: the engine's per-trajectory results
backend.raw.stats["optimization"]                      # the engine that ran ("turbo" here)
```

Swapping emu-sv or emu-mps for emu_cheb in an existing notebook: replace `SVBackend` / `MPSBackend` by `ChebBackend`
and `SVConfig` / `MPSConfig` by `ChebConfig`; observables, noise models and `Results` are pulser's. Initial states and
operators are emu-sv's (`StateVector`, `DenseOperator`, re-exported here).

## Engines: `optimization=`

| `optimization=` | runs (in `emu_cheb.solvers`) | handles |
|---|---|---|
| `"auto"` (default) | `"quench"` if the run allows it, else `"turbo"` | everything |
| `"quench"` | `rydberg_cheb_fast.evolve` (Rydberg), `heisenberg_cheb_fast.evolve` (Heisenberg stack, pulser XY) | noiseless, constant drive, phase 0, from the ground state; real float64 series, the fastest; off-diagonal observables through its `on_state` hook (no array of states) |
| `"turbo"` | `noisy_cheb_turbo.run_sequence_data` | any pulse, phase, local addressing / DMM / SLM, every noise channel, Rydberg / XY / Heisenberg |
| `"fast"` | `noisy_cheb_fast.run_sequence_data` | same as turbo, same trajectories (validated yardstick) |
| `"reference"` | `noisy_cheb.run_sequence_data` | same physics, one trajectory at a time (validated yardstick; its random stream is laid out differently, so it agrees with fast / turbo statistically, not trajectory by trajectory) |

`"auto"` picks `"quench"` when all of these hold, otherwise `"turbo"`; `optimization="quench"` raises a `ValueError`
naming the first that fails:
- no Lindblad noise and no shot-to-shot noise (SPAM readout errors alone are fine: they only act on `BitStrings`);
- the drive and the interaction are constant over the whole sequence (one piece: no pulse shape, no SLM mask);
- laser phase 0 (the quench solvers need a real H);
- initial state all ground;

All three interaction modes (Rydberg, XY, Heisenberg) run on every engine. For `"quench"`, pulser's XY stack
[C3, C6] is mapped exactly onto the Heisenberg solver: J^xx = J^yy = C3/2, J^zz = C6/4, δ_i → δ_i + Σ_j C6_ij/2
(|u⟩⟨u| = 1 − n turns C6 |uu⟩⟨uu| into C6/4 Z_iZ_j plus detuning shifts and a constant, i.e. a global phase).

**Can the Heisenberg solver run Rydberg?** Yes: with n = (1 − Z)/2, Σ J_ij n_i n_j = Σ J_ij/4 Z_i Z_j + Σ_i (Σ_j
J_ij/2) n_i + const (`heisenberg_cheb_fast.rydberg_as_heisenberg`), equal up to a global phase, so probabilities are
identical (checked by `heisenberg_cheb_fast`'s self-check). The backend still runs Rydberg with `rydberg_cheb_fast`:
it has no pair flips, so each step costs less. `heisenberg_cheb_fast.evolve_noisy` (Pauli noise only) is superseded
by turbo and not used by the backend.

### Calling the engines directly

The backend is a thin layer; the engines can be called as they are:

```python
import numpy as np
from emu_cheb.solvers import rydberg_cheb_fast as ryd, heisenberg_cheb_fast as heis, noisy_cheb_turbo as turbo

# Noiseless quench, constant H, from |0...0>, R realizations at once (times in us, rates in rad/us)
J = ryd.couplings(positions)                                          # (n, n) C6 / r^6
P = ryd.evolve(J[None], omega[None], delta[None], times_us, device="cuda")   # (T, R, 2^n) probabilities
occ, corr = ryd.moments(P[:, 0], n)                                   # <n_i>, <n_i n_j>

# Noiseless Heisenberg quench: any (3, n, n) stack [Jxx, Jyy, Jzz]
P = heis.evolve(J3[None], omega[None], delta[None], times_us, device="cuda")
# the Rydberg quench above through the Heisenberg solver (same probabilities, slower)
P_same = heis.evolve(*[x[None] for x in heis.rydberg_as_heisenberg(J, omega, delta)], times_us, device="cuda")

# Anything else: pulser sequence + noise model (same call for noisy_cheb_fast and noisy_cheb)
res = turbo.simulate(seq, noise, n_trajectories=2000, eval_fractions=np.linspace(0, 1, 21), device="cuda")
res.occupations.mean(0), res.correlations, res.probabilities   # = backend.raw of a ChebBackend run
```

All engines take an opt-in `on_state` hook that receives psi at each output time while they hold it (for observables
that need amplitudes): `rydberg_cheb_fast.evolve(..., on_state=f)` / `heisenberg_cheb_fast.evolve` call `f(j, psi)`
with psi (R, 2^n); the noisy engines' `run_sequence_data(..., on_state=f)` call `f(j, traj_ids, psi, active)`.
Default `None`: the code path is unchanged (the engines' self-checks print the same numbers).

## Hamiltonians

Conventions as emu-sv / emu-mps: frequencies in rad/µs, sequence times in ns, atom 0 = most significant bit of the
basis index, local index 0 = ground ("g" / "u"), 1 = excited ("r" / "d"), n = |1⟩⟨1|. The drive (Ω_i, δ_i, φ_i)
comes from pulser's samples, piecewise constant on the `dt` grid (midpoint samples, as emu-sv / emu-mps); global
and local channels, DMM and SLM masks all arrive this way.

$$H(t) = \sum_i \frac{\Omega_i(t)}{2}\left(\cos\phi_i X_i + \sin\phi_i Y_i\right) - \sum_i \delta_i(t)\, n_i + H_{\rm int}$$

| mode | how it is chosen | $H_{\rm int}$ |
|---|---|---|
| Rydberg | pulser `ising` channels (`rydberg_global`, ...) | $\sum_{i<j} U_{ij}\, n_i n_j$, $U_{ij} = C_6/r_{ij}^6$ |
| XY | pulser `XY` channel (`mw_global`) | $\sum_{i<j} \left[\frac{C_3^{ij}}{2}(X_iX_j + Y_iY_j) + C_6^{ij}\,\lvert 0\rangle\langle 0\rvert_i \lvert 0\rangle\langle 0\rvert_j\right]$, $C_3^{ij} = \frac{C_3}{r_{ij}^3}(1 - 3\cos^2\theta_{ij})$ (pulser's 2-layer stack) |
| Heisenberg | `ChebConfig(heisenberg=...)` | $\sum_{i<j} \left(J^{xx}_{ij} X_iX_j + J^{yy}_{ij} Y_iY_j + J^{zz}_{ij} Z_iZ_j\right)$ |

`heisenberg=` replaces the sequence's interaction entirely:
- a `(3, n, n)` stack `[Jxx, Jyy, Jzz]` (rad/µs, symmetric, zero diagonal): any graph, no geometric profile imposed,
  e.g. NMR scalar couplings; constant in time;
- `(Jx, Jy, Jz)`: J^a = J_a U(t), U the sequence's own matrix (C3/r³ for an XY-mode sequence, as
  `examples/emu_mps/heisenberg.ipynb`; C6/r⁶ for a Rydberg one); keeps U's time dependence (SLM mask).

In both cases the drive keeps the form above (X, Y and n of each atom).

**XY model with any couplings.** The interaction is just a matrix, so the XY model
$\sum_{i<j} \frac{J_{ij}}{2}(X_iX_j + Y_iY_j)$ on any graph takes either
- pulser's XY channel (`mw_global`) with `ChebConfig(interaction_matrix=np.stack([J, C6]))`: pulser / emu-base want
  the 2-layer XY stack (a plain (n, n) matrix is rejected); `C6 = 0` gives the pure XY model, and without
  `interaction_matrix` pulser uses the register's dipolar $C_3$ couplings and $C_6/r^6$ shift; or
- `ChebConfig(heisenberg=np.stack([J / 2, J / 2, Jzz]))` on any sequence (`Jzz = 0`: XY; otherwise XXZ).

Both are checked against dense evolution for every engine (`test_backend.py`, `examples/emu_cheb/xy.ipynb`).

## Noise

The noise model is pulser's `NoiseModel` (or the device's with `prefer_device_noise_model=True`), sampled by
emu-base's `PulserData` exactly as for emu-mps; `n_trajectories` samples, one trajectory each.

| pulser channel | how |
|---|---|
| dephasing, depolarizing, relaxation, eff_noise (2×2), leakage | Lindblad jumps, Monte Carlo wave functions with emu-mps' jump rule; non-Hermitian H_eff propagated exactly |
| SPAM state preparation | bad atoms per sample (from PulserData) |
| SPAM readout (`p_false_pos`, `p_false_neg`) | on `BitStrings` only, by emu-sv's sampler (`config.noise_model`'s rates, as emu-sv) |
| amplitude, laser waist, detuning, high-frequency detuning, Doppler, register | per-sample drives / couplings (from PulserData) |
| hyperfine dephasing | not supported (digital basis; nor by emu-mps) |

## Observables

| observable | cost |
|---|---|
| `Occupation`, `CorrelationMatrix`, `BitStrings` | none: from the probabilities the engines compute anyway |
| `Energy`, `EnergyVariance`, `EnergySecondMoment`, `Expectation(O)`, `Fidelity(φ)`, `StateResult` | only when requested: evaluated on each trajectory's state at each output time, through the engines' `on_state` hook |

- Each off-diagonal observable is evaluated only at its own `evaluation_times` (once per trajectory and time; the
  count is `backend.raw.stats["off_diagonal_evaluations"]`): e.g. `Occupation` at 51 times and `Energy` at the end
  computes the energy once per trajectory, not 51 times.
- Values are trajectory averages, stored at exactly the times emu-sv would store them (emu-base's
  `is_evaluation_time` on PulserData's grid). Per-trajectory diagonal data: `backend.raw`.
- `BitStrings`: `n_trajectories × num_shots` shots in total, as emu-sv's union of per-trajectory samples.
- `Energy` is ⟨H⟩ with the Hermitian H of the drive step that ends at the output time (as emu-sv), not H − iG.
- `StateResult` needs a single trajectory (noiseless runs); for noisy runs use `Expectation` or `Fidelity`.
- `connected(occupation, correlation)` gives ⟨n_i n_j⟩ − ⟨n_i⟩⟨n_j⟩ (the ZZ correlator up to affine factors).
- Other observables raise `NotImplementedError` when the config is built.

## Options

`ChebConfig(dt=10, optimization="auto", gpu=None, seed=0, heisenberg=None, single=False, bounds="lanczos",
interaction_cutoff=0.0, log_level=logging.INFO, log_file=None, **pulser_options)`:
`gpu=None` uses CUDA when available; `single=True` runs fast / turbo in complex64 (~1.8× faster from n ≈ 20, round-off
~1e-6); `bounds="gershgorin"` replaces the guarded Lanczos spectral bounds by rigorous ones; `seed` drives the
trajectories' random numbers (noise samples come from pulser's RNG). GPU memory: the engines use at most 40 % of the
free memory and size their batches to it. Algorithms, validation and timings: [`solvers/README.md`](solvers/README.md).

## Examples

`examples/emu_cheb/`: `quench_emu_cheb.ipynb` and `noisy_quench_emu_cheb.ipynb` (emu-sv's tutorials with the backend
swapped, compared with emu-sv: 16-atom quench 4e-11 and 10× faster; noisy quench within statistical error of emu-sv's
Lindblad solver), `heisenberg.ipynb` (`examples/emu_mps/heisenberg.ipynb` with `heisenberg=`: round-off against the
exact state vector, energies included; noisy runs against the exact Lindblad equation), `xy.ipynb` (the XY model on a
custom coupling graph, N = 4: pulser's XY channel with a custom `interaction_matrix`, the Heisenberg-stack route,
pulser's default couplings, single-excitation hopping, every engine incl. quench, noise against exact Lindblad).

## Tests

```bash
pytest test/emu_cheb -q                                   # ~15 s, CPU, N <= 4
EMU_CHEB_SELF_CHECKS=1 pytest test/emu_cheb -q            # + the engines' own self-checks (minutes)
```

`test/emu_cheb/test_backend.py`: noiseless Rydberg against emu-sv (occupations, correlations, energy, variance, ⟨X⟩, fidelity,
final state; turbo and quench paths), Heisenberg stacks against dense evolution, noisy runs against emu-sv's exact
Lindblad solver (occupations, bitstrings with readout errors, ⟨X⟩), the engines against each other, the auto rule,
errors. `test_engine_hook.py`: the engines' `on_state` hooks (noisy engines and quench solvers). `test_self_checks.py`:
the engines' self-checks (opt-in). Validation against emu-mps and GPU checks: `test/emu_cheb/validation/`,
`test/emu_cheb/cluster/` ([solvers/README.md](solvers/README.md)).

## Limitations

- Initial states: basis states only (a `StateVector` with one non-zero amplitude).
- No density-matrix solver: noise always goes through trajectories (`n_trajectories`; pulser's default is 1). For a few atoms
  on CPU, emu-sv's density-matrix solver is faster (6 atoms: 4 s against 8 s for 4000 trajectories); trajectories win
  at larger n, where the density matrix (4^n) does not fit, and on GPU.
- `StateResult` only with one trajectory.
- Readout errors use `config.noise_model` (as emu-sv), also with `prefer_device_noise_model=True`.
