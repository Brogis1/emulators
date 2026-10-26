# emu_cheb: an emu-sv-style backend for the Chebyshev engines

Date: 2026-10-26. Status: approved 2026-10-26 (revised the same day: optimization="auto" with the quench solvers, full
Heisenberg stacks, `backend.raw`).

> **Layout superseded (2026-10-26, later the same day):** emu_cheb became an installable package like emu_sv. The
> engines moved from `scripts/` into `emu_cheb/solvers/` (relative imports; `simulate()` builds its PulserData config
> with `ChebConfig` instead of emu-mps' `MPSConfig`), the backend from `experimental/emu_cheb/` to `emu_cheb/`
> (`engines.py` removed), packaging in `ci/emu_cheb/`, tests / validation / cluster tools in `test/emu_cheb/`,
> notebooks and experiments in `examples/emu_cheb/`. Paths below describe the design as first written.

## Goal

Call the validated Chebyshev engines in `scripts/` through the familiar pulser / emu-sv interface:

```python
from emu_cheb import ChebBackend, ChebConfig, Occupation, CorrelationMatrix, BitStrings, Energy

config = ChebConfig(observables=[Occupation(evaluation_times=times), BitStrings(num_shots=1000)],
                    noise_model=noise, n_trajectories=500, dt=10)
backend = ChebBackend(seq, config=config)
results = backend.run()                    # a pulser Results
results.occupation, results.final_bitstrings
backend.raw                                # the engine's NoisyResults: per-trajectory occupations, stats
```

A user who knows emu-sv or emu-mps changes only the import and the class names. The simple script interface
(`noisy_cheb_turbo.simulate`, see scripts/README.md) stays the main tool for physics work; the backend is a thin
adapter on top of it and never hides what the scripts give (`backend.raw`).

## Principles

1. **Wrap, don't copy.** The engines stay in `scripts/` as the single source of truth. The new package imports them.
   The only change to `scripts/` is one opt-in hook (see "Engine hook"), whose default leaves the checked code path
   unchanged.
2. **Keep the batching.** All PulserData samples go to ONE engine call (one Chebyshev series serves every output time,
   trajectories are batched). No per-sample `_run_from_sequence_data` + `Results.aggregate` loop as in emu-sv.
3. **Pay only for what is asked.** Diagonal observables come from what the engines already return (no extra
   compute). Off-diagonal observables (needed e.g. for NMR measurements in other bases) are computed only when one of
   them is in `config.observables`.
4. **Best engine by default, any engine on request.** `optimization="auto"` takes the cheapest validated engine able to run the sequence; `optimization="reference" | "fast" | "turbo" |
   "quench"` always runs exactly that code (to compare versions).

## Engines and how "auto" chooses

| `optimization=` | script entry point | solves |
|---|---|---|
| `"quench"` | `rydberg_cheb_fast.evolve` (Rydberg) / `heisenberg_cheb_fast.evolve` (Heisenberg stack) | noiseless, constant H, real float64 series, from \|0...0>; psi for off-diagonal observables through its `on_state` hook |
| `"turbo"` | `noisy_cheb_turbo.run_sequence_data` | everything: pulses, phases, local / DMM / SLM, all noise channels, Rydberg / XY / Heisenberg |
| `"fast"` | `noisy_cheb_fast.run_sequence_data` | same trajectories as turbo (validated yardstick) |
| `"reference"` | `noisy_cheb.run_sequence_data` | same trajectories, one at a time (validated yardstick) |
| `"auto"` (default) | `"quench"` if eligible, else `"turbo"` | |

Quench-eligible (all must hold; `optimization="quench"` raises `ValueError` naming the failed condition):
- no Lindblad operators and no stochastic noise (a single distinct SequenceData; SPAM readout errors alone are fine,
  they act on BitStrings only);
- one constant piece (drive and interaction matrix constant over the whole duration: `noisy_cheb._Model.pieces` has
  length 1);
- phi = 0 everywhere (the quench solvers need a real H);
- initial state all ground;
- Rydberg mode, or a Heisenberg stack (`heisenberg=` set). Pulser's XY mode without `heisenberg=` goes to turbo (its
  C6 |uu><uu| layer has no quench mapping).

Rydberg is run by `rydberg_cheb_fast`, not by `heisenberg_cheb_fast` through `rydberg_as_heisenberg`: the two are
equal up to a constant (a global phase; checked by heisenberg_cheb_fast's self-check), but the Rydberg solver has no
pair flips and is cheaper. `heisenberg_cheb_fast.evolve_noisy` (Pauli noise only) is superseded by turbo and not used.

## Layout

```
experimental/emu_cheb/
├── __init__.py        ChebBackend, ChebConfig, connected; re-exports of the supported pulser observables, Results
├── config.py          ChebConfig(EmulationConfig)
├── engines.py         puts scripts/ on sys.path and imports the engines unchanged
├── backend.py         ChebBackend(EmulatorBackend): PulserData -> engine -> Results; engine dispatch
├── observables.py     diagonal mapping + the on_state accumulator for off-diagonal observables
├── README.md          usage, every engine and how to call it, Hamiltonians, noise, observables, limitations
└── test_backend.py    CPU tests (N <= 4 atoms, seconds)
examples/emu_cheb/heisenberg.ipynb   examples/emu_mps/heisenberg.ipynb redone with ChebBackend, with timings
examples/emu_cheb/{quench,noisy_quench}_emu_cheb.ipynb   emu-sv's tutorials with ChebBackend, vs emu-sv
```

`engines.py` puts `scripts/` on `sys.path` (repo-relative, as the subfolders of `scripts/` already do) and imports
the engines; `backend.py` and `observables.py` import them from there. `experimental/` is not packaged: the notebook and the tests add it to `sys.path` (pytest does it from
the test file's location).

## ChebConfig(EmulationConfig)

Base options reused unchanged (they reach `PulserData` as for emu-sv): `observables`, `default_evaluation_times`,
`noise_model`, `n_trajectories`, `initial_state`, `interaction_matrix`, `with_modulation`,
`prefer_device_noise_model`, `default_num_shots`.

Own options:

| option | default | meaning |
|---|---|---|
| `dt` | 10 | drive sampling step (ns), as `SVConfig.dt` |
| `optimization` | `"auto"` | `"auto"`, `"quench"`, `"turbo"`, `"fast"`, `"reference"` (table above) |
| `gpu` | `None` | `None`: CUDA if available; `True` / `False` force it |
| `seed` | 0 | seed of the trajectories' random numbers (noise samples come from pulser's RNG, as emu-sv) |
| `heisenberg` | `None` | Heisenberg couplings: a `(3, n, n)` stack [Jxx, Jyy, Jzz] (rad/us, any graph, as in scripts/README.md), or `(Jx, Jy, Jz)` scaling the sequence's own interaction matrix U(t) as in `examples/emu_mps/heisenberg.ipynb` |
| `single` | `False` | complex64 (fast / turbo only; `ValueError` with `optimization="reference"` or `"quench"`) |
| `bounds` | `"lanczos"` | spectral bounds of turbo and quench: `"lanczos"` (guarded) or `"gershgorin"` |
| `interaction_cutoff` | 0.0 | as `SVConfig` (PulserData reads it) |
| `log_level`, `log_file` | `INFO`, `None` | as emu-sv; engine `stats` are logged at INFO |

`_state_type` / `_operator_type` are emu-sv's `StateVector` / `DenseOperator`, so initial states, `Fidelity` targets
and `Expectation` operators are built the emu-sv way.

Validation in `__init__` (fail before a long run): unknown `optimization` / `bounds`; `heisenberg` not of shape (3,) or
(3, n, n); an observable outside the supported list -> `NotImplementedError`; `StateResult` with more than one
trajectory -> `ValueError` suggesting `Expectation`. `noise_model.runs` / `samples_per_run` are ignored with a
warning, as emu-sv.

## ChebBackend.run()

1. `sequences = PulserData(sequence, config, dt).get_sequences()` (same noise sampling as emu-sv / emu-mps).
2. `heisenberg` set: each SequenceData gets the Heisenberg interaction and `hamiltonian_type = XY` (the engines read
   a 3-layer stack as Heisenberg only outside Rydberg mode; same transform as `_heisenberg` in
   `scripts/noisy_vs_emu_mps.py`). A (3, n, n) stack is used as is (constant in time); (Jx, Jy, Jz) gives
   J_a * U(t)[0] (keeps time dependence such as an SLM mask).
3. Initial state: `None` -> all ground; a `StateVector` with exactly one non-zero amplitude -> the bitstring the engines
   take; anything else -> `NotImplementedError` (the engines start from a basis state).
4. Output times: PulserData already puts every observable's evaluation times on its time grid. The output times are
   the grid points t / T where at least one observable is due, decided by emu-base's `is_evaluation_time` (the
   function emu-sv uses), so every observable is stored at exactly the times emu-sv would store it.
5. Engine call (dispatch above). Noisy engines: one `run_sequence_data(sequences, fractions, seed=..., device=...,
   initial_state=..., on_state=...)`. Quench: one `evolve(J, omega, delta, times_us, device, bounds=...,
   on_state=...)` on the single SequenceData, wrapped into the same `NoisyResults` (one trajectory). `on_state` is
   `None` unless an off-diagonal observable is requested.
6. `backend.raw = NoisyResults` of the engine; one `Results(atom_order=qubit_ids, total_duration=...)`, filled with
   `Results._store(observable, time, value)` for each observable at its own evaluation times.
   `ChebBackend._run_from_sequence_data(sequences, config)` (as emu-mps / emu-sv, but taking all samples) returns the
   Results, for custom SequenceData as in the notebook.

## Observables

**Diagonal (from `NoisyResults`, no extra compute):**

| observable | value |
|---|---|
| `Occupation` | `occupations.mean(0)[j]` (n,) |
| `CorrelationMatrix` | `correlations[j]` (n, n), <n_i n_j> |
| `BitStrings` | `n_trajectories * num_shots` shots (same total count as emu-sv's bag union of per-trajectory samples) drawn by emu-sv's own `StateVector.sample` on the state sqrt(P(t)) (same \|amplitude\|^2), so sampling and emu-base readout errors (`config.noise_model.p_false_pos / p_false_neg`, as emu-sv) are emu-sv's |

`connected(occupation, correlation)` returns <n_i n_j> - <n_i><n_j> (ZZ correlator up to affine factors).

**Off-diagonal (only when requested, through `on_state`):**

| observable | per trajectory, then averaged |
|---|---|
| `Expectation(O)` | `O.expect(state)` (emu-sv) |
| `Fidelity(phi)` | pulser's `Fidelity.apply` (\|<phi\|psi>\|^2) |
| `Energy` | <psi\|H\|psi>, H the Hermitian Hamiltonian of the drive piece ending at the output time (the first piece at t = 0), as emu-sv's Hamiltonian of the step just completed |
| `EnergySecondMoment`, `EnergyVariance` | \|\|H psi\|\|^2 and \|\|H psi\|\|^2 - <H>^2 from the same H psi |
| `StateResult` | emu-sv `StateVector` of the (single) trajectory |

`psi` is normalised and embedded into the full 2^n space (bad and leaked atoms in their 0 bit, as the engines'
probabilities). Energy uses the Hermitian H only, not H_eff = H - iG (the physical energy, real valued); H is built
by the backend with `noisy_cheb._build` on the trajectory's piece with G = 0 (H psi = w H~ psi + c psi).

Anything else raises `NotImplementedError` at config time.

## Engine hook (the only change to scripts/)

`run_sequence_data(..., on_state=None)` in `noisy_cheb`, `noisy_cheb_fast`, `noisy_cheb_turbo`. At output index j,
after the existing diagonal bookkeeping, the record step calls

```
on_state(j, traj_ids, psi, active)
```

- `traj_ids`: global indices of the rows (into `sequences`), so batched / chunked rows map to their trajectory.
- `psi`: (rows, 2^n_active) states, unnormalised, on the engine's device; read-only.
- `active`: bool mask of the atoms in the simulation.

The hook may be called more than once for the same (j, trajectory): turbo reruns a batch with Gershgorin bounds when
the Lanczos guard trips. The backend therefore assigns per-trajectory values (indexed by trajectory and output index),
never accumulates.

Placement: `fast._Batch._record` (turbo inherits it; `on_state` / `ids` set on the batch by `run_sequence_data`) and
the reference `trajectory().record`. With `on_state=None` the code path is unchanged; proven by the engines'
self-checks (CPU), then the emu-mps harness (`noisy_vs_emu_mps.py --engine reference|fast|turbo`) and
`cluster/gpu_turbo_check.py` on the cluster. The quench solvers get the same opt-in hook, `evolve(..., on_state=f)`
calling `f(j, psi)` with psi (R, 2^n) inside their conversion loop (the series accumulator holds psi for every output
time anyway): output stays "probabilities", no T x 2^n array of states, no extra matvecs; default None unchanged
(their self-checks print the same numbers).

Rejected alternative: patching engine classes from `emu_cheb` at runtime (no edit to scripts/, but fragile).

## Documentation

`experimental/emu_cheb/README.md` and the module docstrings state, without the reader opening scripts/:

- **Quick start** as scripts/README.md's: register, pulse, noise model, run, read results, error bars from
  `backend.raw`.
- **Every engine and how to call it**, through the backend (`optimization=`) AND directly from scripts/ (one snippet each:
  `rydberg_cheb_fast.evolve`, `heisenberg_cheb_fast.evolve`, `noisy_cheb_turbo.simulate` with the same signature for
  `noisy_cheb_fast` / `noisy_cheb`), with what each supports, the auto rule, and the Rydberg-as-Heisenberg equivalence
  (exact up to a global phase; why `rydberg_cheb_fast` is used for Rydberg).
- **The Hamiltonian of each mode**, in emu-sv's conventions (rad/us, ns, atom 0 = most significant bit, 0 = ground):
  - Rydberg (pulser `ising`): H = sum_i [Omega_i/2 (cos phi_i X_i + sin phi_i Y_i) - delta_i n_i] + sum_{i<j} U_ij n_i n_j,
    U_ij = C6 / r_ij^6.
  - XY (pulser `XY`): drive as above plus sum_{i<j} [C3_ij / 2 (X_i X_j + Y_i Y_j) + C6_ij |0><0|_i |0><0|_j]
    (emu-base's 2-layer interaction stack; index 0 = "u"), as in scripts/README.md "Conventions".
  - Heisenberg (`heisenberg=`): drive as above plus sum_{i<j} (Jxx_ij X_i X_j + Jyy_ij Y_i Y_j + Jzz_ij Z_i Z_j); the
    stack replaces pulser's interaction entirely (any graph, e.g. NMR scalar couplings); with (Jx, Jy, Jz) the J_a
    are multiples of the sequence's matrix (C3/r^3 for an XY-mode sequence, C6/r^6 for Rydberg).
- **The noise model:** which pulser NoiseModel channels are supported, how (shot-to-shot samples from PulserData,
  Lindblad channels by Monte Carlo wave functions with emu-mps' jump rule, SPAM readout as a classical channel).
- **Observables:** the diagonal / off-diagonal split, what each costs, BitStrings counting, Energy = Hermitian H.
- **Engine options:** `single`, `bounds`, GPU memory behaviour; pointer to scripts/README.md for the algorithms and
  validation.
- **Limitations** (pulser's default `n_trajectories` is 1: set it for noisy runs, as with emu-mps; basis-state initial states, StateResult only with one trajectory, no density-matrix solver).

Every public class and function gets a docstring with arguments and units; non-obvious lines get a short comment, as
in scripts/.

## Error handling

Config-time errors as listed above. At run time: mismatch between the initial state's qubit count and the register
-> `ValueError` (as emu-sv); `initial_state` together with state-preparation error -> `NotImplementedError` (as emu-sv);
`optimization="quench"` on an ineligible run -> `ValueError` naming the condition. Engine errors (spectral guard, memory)
propagate unchanged.

## Tests (test_backend.py, CPU, N <= 4 atoms, seconds; pytest)

The engines are validated (scripts/README.md); these tests check the wiring of emu_cheb, so they stay on N <= 4
atoms (2^4 = 16 amplitudes, dense references are instant) and run in seconds. Larger N only in the GPU smoke run.

1. Noiseless Rydberg with a shaped pulse and a Néel initial state: `Occupation`, `CorrelationMatrix` match
   `SVBackend` to 1e-8.
2. `heisenberg=` (3, n, n) stack and (Jx, Jy, Jz): occupations match a dense evolution of the Heisenberg H built from
   Pauli matrices in the test (independent of the engines) to 1e-8.
3. Noisy (dephasing + depolarizing, plus SPAM readout): occupations match emu-sv's density-matrix Lindblad solver
   within 5 x 0.5 / sqrt(n_traj); bitstring frequencies match readout(diagonal of emu-sv's final density matrix).
4. `optimization="fast" | "turbo"` with the same seed agree to round-off (one random stream per trajectory);
   `"reference"` (one stream for all trajectories) agrees statistically; `auto` picks quench on a noiseless
   constant pulse (Rydberg and Heisenberg) and matches turbo to 1e-8; `optimization="quench"` on a shaped pulse raises.
5. Off-diagonal: `Energy`, `EnergyVariance`, `Expectation(sigma_x on qubit 0)`, `Fidelity`, `StateResult` match
   `SVBackend` to 1e-8 (noiseless, turbo and quench paths); `Expectation` under noise matches the Lindblad reference
   within the statistical bound.
6. Hook: for each noisy engine, the trajectory mean of |psi|^2 seen by `on_state` equals `probabilities`, every
   (j, trajectory) is seen, and a no-op `on_state` leaves the results identical to `on_state=None`.
7. Errors: unsupported observable, `StateResult` with several trajectories, non-basis initial state, bad
   `heisenberg` shape.

Then one GPU smoke run on pasqal-dgx via the runner: turbo and quench, n = 10, FRESNEL noise, all observable kinds;
timing with and without off-diagonal observables to confirm the diagonal path costs nothing extra.

## Deliverables, in order

1. Engine hook in `scripts/` + engine self-checks (CPU) showing nothing changed + hook tests (6).
2. `experimental/emu_cheb` diagonal path (config, backend with all engines incl. quench and auto, `raw`) + tests 1-4, 7.
3. Off-diagonal observables + test 5.
4. `experimental/emu_cheb/README.md` (see Documentation), `examples/emu_cheb/heisenberg.ipynb`; `scripts/README.md`
   points to both.
5. Cluster: harness + `gpu_turbo_check.py` with the hook in place, GPU smoke run.

## Out of scope

Moving the engines into the package and pip packaging; density-matrix (Lindblad) solver; arbitrary initial states;
per-step user callbacks; multi-GPU (`scripts/experiments/multi_gpu.py`); `heisenberg_cheb_fast.evolve_noisy`.
