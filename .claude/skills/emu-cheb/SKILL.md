---
name: emu-cheb
description: Use when simulating Heisenberg (XXZ, XYZ) or XY spin models, custom coupling graphs or pulser mw_global (XY channel) sequences with emu_mps or emu_cheb, when choosing between emu_cheb, emu_mps and emu_sv for a pulser sequence, or when swapping MPSBackend / SVBackend for ChebBackend in the Pasqal emulators repo.
---

# emu-cheb: Heisenberg and XY simulations with emu_cheb and emu_mps

emu_cheb (`ChebBackend` / `ChebConfig`) is an exact Chebyshev emulator. It has no time step and no truncation, takes
pulser sequences and noise models like emu-sv, and runs all trajectories in one call. emu_mps is TDVP on an MPS: it is
approximate but scales past 2^n. In this repo both run any coupling graph as an XY or a Heisenberg model. The recipes
below are tested; don't re-derive the conventions from the source. Full API: `emu_cheb/README.md`.

## Setup (check first, install only what is missing)

```bash
python -c "import emu_cheb, emu_mps, emu_sv; print(emu_cheb.__file__, emu_mps.__file__)"   # paths under the repo = editable
```

If an import fails, install from a clone of the public fork **https://github.com/Brogis1/emulators**, not from
pasqal-io/emulators or PyPI. Install all four packages editable from that clone. The `ci/<pkg>` directories are
symlinks into the source tree. Use `uv pip install` if the venv has no pip (uv-managed); otherwise `pip install`:

```bash
git clone https://github.com/Brogis1/emulators.git && cd emulators   # skip if already inside it
uv pip install -e ci/emu_base -e ci/emu_sv -e ci/emu_mps -e ci/emu_cheb   # dev_install.sh does the same with plain pip
```

Don't use `pip install "git+...#subdirectory=ci/emu_cheb"`. Its pinned dependencies (`emu-base==2.10.0`,
`emu-sv`, `emu-mps`) then come from PyPI, which has neither emu_mps' Heisenberg mode nor the `("u","d")` basis.

- emu_cheb needs emu-base and emu-sv (with pulser and torch). emu_mps is only needed for the emu_mps recipes and for
  `test/emu_cheb/validation/`.
- GPU kernels need Triton, which ships with CUDA builds of torch: install torch for your CUDA version first. On CPU,
  emu_cheb runs without Triton.
- Install editable, never from PyPI. The XY/Heisenberg support and the `("u","d")` MPS basis exist only in this repo.
- Smoke test: `pytest test/emu_cheb -q` (~15 s, CPU).
- Notebooks: use the `emulators-venv` kernel. The default `python3` kernel may point to another venv.

## Pick the emulator, and say where it runs

| n atoms | use | why |
|---|---|---|
| ≤ ~16 (CPU) / ≤ ~26 (A100 40 GB) | **emu_cheb** | exact to round-off; noisy trajectories batched (ms per trajectory) |
| > 26, or long-range-entangled at large n | **emu_mps** | MPS; accuracy depends on χ and dt, and errors don't warn |
| ≤ ~6 with Lindblad noise, CPU | emu-sv density matrix | cheaper than trajectories at this size |

Before running, tell the user where it runs (CPU, local GPU, cluster) and roughly what it costs.

**Treat emu_cheb as the yardstick for emu_mps.** If the emu_mps target is too big for emu_cheb, first run emu_mps at an
n that emu_cheb can still do (≤ 20), on the same geometry and protocol. That sets χ and dt. Don't scan χ at the full
size: χ is an accuracy knob, and truncation shows up as a smooth, plausible, wrong curve.

## Conventions (all emulators)

- Units: rad/µs and ns. `evaluation_times` are fractions of the duration.
- Atom 0 is the most significant bit of the basis index. Index 1 is the excited state: `"r"`, or `"d"` in pulser's XY
  basis `("u", "d")`. `n = |1⟩⟨1| = (1 − Z)/2`, and `Occupation` measures it. The Néel state "0101" is `"udud"`.
- Drive: `Σ Ω/2 (cos φ X + sin φ Y) − Σ δ n` with Pauli X and Y, for every interaction mode.
- Couplings: give a symmetric matrix with a zero diagonal. Only i < j is used, and each pair is counted once.
- XY channel (`mw_global`) needs `MockDevice`; `AnalogDevice` has no XY channel.

## Recipes

Each H_int below is summed over i < j.

| model | emu_cheb | emu_mps |
|---|---|---|
| XY: `J/2 (XX + YY)` | `mw_global` sequence + `ChebConfig(interaction_matrix=np.stack([J, 0*J]))` | same sequence + `MPSConfig(interaction_matrix=np.stack([J, 0*J]))`, plain `MPSBackend(seq, config=cfg).run()` (`config=` is keyword-only) |
| XY + `C6 \|uu⟩⟨uu\|` shift | layer 2 = C6 | same |
| pulser's dipolar XY | no `interaction_matrix` | same |
| Heisenberg `Jxx XX + Jyy YY + Jzz ZZ` | `ChebConfig(heisenberg=np.stack([Jxx, Jyy, Jzz]))` on any sequence | private route, see below |
| XY / XXZ as Heisenberg | `heisenberg=np.stack([J/2, J/2, Jzz])` | same stack, private route |
| scaled dipolar couplings | `heisenberg=(ax, ay, az)`: J^a = a · the sequence's own matrix | build the stack yourself |

- **The factor of 2:** layer 1 of the XY stack, `J`, gives `J/2 (XX+YY)`. In Heisenberg form the same model is
  `Jxx = Jyy = J/2`.
- **No plain matrix for XY:** a plain `(n, n)` `interaction_matrix` on an XY sequence is rejected. Pass the 2-layer
  stack.
- **`heisenberg=` is a replacement:** it swaps out the sequence's whole interaction. The drive and the noise are kept.

**emu_mps Heisenberg (private API, not reachable from `MPSBackend.run`):**

```python
import dataclasses, random
from emu_base import HamiltonianType, PulserData
from emu_mps import MPSBackend, MPSConfig
stack = torch.tensor(np.stack([Jxx, Jyy, Jzz]), dtype=torch.float64)          # (3, n, n)
cfg = MPSConfig(dt=dt, observables=obs, initial_state=psi0, noise_model=noise,
                optimize_qubit_ordering=False)       # keep atom indices comparable
random.seed(0)                                       # emu_mps jumps draw from python's `random`
runs = [MPSBackend._run_from_sequence_data(
            dataclasses.replace(sd, interaction_matrix=lambda t: stack, hamiltonian_type=HamiltonianType.XY), cfg)
        for sd in PulserData(sequence=seq, config=cfg, dt=dt).get_sequences()]   # one call = one trajectory
```

**Initial states (basis states):**
- emu_cheb: `StateVector` with a single non-zero amplitude, e.g. `psi = zeros(2**n); psi[int("0101", 2)] = 1`. It only
  accepts basis states. `"auto"` runs the fastest engine (`quench`) only when the run starts from all-ground, has no
  noise, a constant drive and phase 0. Otherwise it runs `turbo`, which is still exact.
- emu_mps: `MPS.from_state_amplitudes(eigenstates=("u", "d"), amplitudes={"udud": 1.0})`. The `("u","d")` basis exists
  only in the Brogis1 fork (`emu_mps/mps.py`, `mpo.py`). Upstream emu-mps needs
  `MPS([tensor([[[1-b],[b]]]) for b in bits], eigenstates=sd.eigenstates)` instead.

**Imports and results** (same pattern for both): `from emu_cheb import ChebBackend, ChebConfig, Occupation,
CorrelationMatrix, Energy, StateVector, connected` / `from emu_mps import MPS, MPSBackend, MPSConfig, Occupation, ...`.
`times = [k / 50 for k in range(51)]` (fractions, on the dt grid); `torch.stack(results.occupation)` → (T, n);
`results.correlation_matrix`, `results.energy` likewise; `results.state[-1]` for `StateResult`.

## Noise

- Pass a pulser `NoiseModel` as `noise_model=`. **emu_cheb needs `n_trajectories=`** (pulser's default is 1).
  Trajectories are seeded by `ChebConfig(seed=...)` only (default 0); `torch.manual_seed` changes nothing.
  Error bars: `se = sqrt(raw.occupations.std(0)**2 + 1/n_traj) / sqrt(n_traj)` with `raw = backend.raw`. The `1/n_traj`
  floor matters at early times: if no trajectory has jumped yet, the sample std is ~0 and z blows up into the
  thousands although the run is correct.
- emu_mps: one run is one trajectory. Average the runs yourself and compute the error bars yourself.
- Check against the **exact Lindblad equation** (dense ρ, n ≤ ~6) with ≥ 500 trajectories: pass if max |z| < ~4.5.
  All times share the same trajectories, so mean z² is not a χ²/dof; use it only as an upper bound (< 2). Don't
  compare two Monte Carlo runs with 40 trajectories each: a gap of a few percent can't be read.
- Device noise (`prefer_device_noise_model=True`, or FRESNEL fetched from Pasqal Cloud) is calibrated on the
  ground-Rydberg transition. For an XY or Heisenberg model it is a stand-in, and the report must say so.

## Validate before quoting a number

For n ≤ ~10, compare with a dense evolution: use `dense_reference.py` next to this file (state vector and Lindblad;
copy it next to your script; `python dense_reference.py` self-checks it against emu_cheb). It is driven by the same
`PulserData` sample as the emulators, so it is exact for the dt-discretised drive. Two traps:
- Get `sd` from `PulserData(sequence=seq, config=ChebConfig(dt=dt, noise_model=noise), dt=dt)`. Without that
  `noise_model`, `sd.lindblad_ops` is empty.
- Build the reference interaction from **your** J or stack (`xy_int(J)`, `heisenberg_int(stack)`).
  `sd.interaction_matrix` is pulser's matrix (dipolar by default), not the custom one you passed to the backend.

Results are on the dt grid: `rows = [round(t * T / dt) for t in times]`. Worked examples:
`examples/emu_cheb/{xy,heisenberg}.ipynb` and `examples/emu_mps/{xy,heisenberg}.ipynb`. Save the
comparison as a script, and quote only numbers that a saved script produced.

| result | meaning |
|---|---|
| emu_cheb ~1e-13 | correct |
| emu_mps ~1e-6, shrinks ~4× when dt halves | TDVP splitting error from long-range bonds; normal |
| emu_mps ~1e-2, flat in dt | TDVP projection error. Seen without drive (Ω = 0), from a product state whose excitation sits at one end of a non-adjacent bond. A drive or an entangled start avoids it, or use emu_cheb |
| emu_mps warns "Lanczos tolerance below 1e-12" | `precision ≤ 1e-10`; harmless |
| emu_cheb `StateResult` with noise raises | use `Expectation` / `Fidelity` |

If the same obstacle has failed three times, stop and report what you tried. Then offer a smaller n or the other
emulator.
