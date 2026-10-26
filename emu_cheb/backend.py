"""ChebBackend: a pulser EmulatorBackend running the Chebyshev engines of emu_cheb.solvers (README.md).

PulserData samples the sequence and its noise exactly as for emu-sv / emu-mps; all samples then go to ONE engine call
(the engines batch the trajectories and serve every output time with one Chebyshev series), and the engine's results
fill one pulser Results. backend.raw keeps the engine's own NoisyResults (per-trajectory occupations, statistics).
"""
from __future__ import annotations

import dataclasses
import logging

import numpy as np
import torch
from pulser.backend import BitStrings, EmulatorBackend, Results

from emu_base import HamiltonianType, PulserData, is_evaluation_time

from .config import OFF_DIAGONAL, ChebConfig
from .solvers import heisenberg_cheb_fast, noisy_cheb, noisy_cheb_fast, noisy_cheb_turbo, rydberg_cheb_fast
from .observables import OffDiagonal, diagonal_value

NOISY = {"reference": noisy_cheb, "fast": noisy_cheb_fast, "turbo": noisy_cheb_turbo}


class ChebBackend(EmulatorBackend):
    """Emulates a pulser Sequence with the Chebyshev engines of emu_cheb.solvers: exact propagation (no time step, no
    truncation) of every trajectory, emu-mps' noise model.

    Args:
        sequence: the pulser Sequence. config: a ChebConfig. mimic_qpu: as emu-sv's SVBackend.

    After run(), `raw` holds the engine's noisy_cheb.NoisyResults: probabilities (T, 2^n), per-trajectory
    occupations (n_traj, T, n) for error bars, correlations (T, n, n), leaked flags, stats (incl. "optimization",
    the engine that ran).
    """

    default_config = ChebConfig(observables=[BitStrings(evaluation_times=[1.0])])
    raw = None

    def run(self) -> Results:
        """Emulates the sequence; returns one pulser Results (trajectory averages)."""
        config = self._config
        if not isinstance(config, ChebConfig):
            raise TypeError(f"ChebBackend needs a ChebConfig, got {type(config).__name__}")
        data = PulserData(sequence=self._sequence, config=config, dt=config.dt)
        results, self.raw = run_sequence_data(data.get_sequences(), config)
        return results

    @staticmethod
    def _run_from_sequence_data(sequences, config: ChebConfig) -> Results:
        """As emu-sv's / emu-mps', but taking ALL the SequenceData samples at once (one trajectory each)."""
        return run_sequence_data(sequences, config)[0]


def run_sequence_data(sequences, config: ChebConfig) -> tuple[Results, noisy_cheb.NoisyResults]:
    """(Results, the engine's NoisyResults) of the SequenceData samples `sequences` (one trajectory each)."""
    sds = list(sequences)
    if config.heisenberg is not None:
        sds = heisenberg_samples(sds, config.heisenberg)
    sd0 = sds[0]
    grid = np.asarray(sd0.target_times, float)
    fraction = grid / grid[-1]
    # every observable at exactly the grid times emu-sv would store it at (PulserData put them on the grid)
    due = {
        id(obs): [s for s in range(len(grid)) if is_evaluation_time(config, obs, fraction[s])]
        for obs in config.observables
    }
    steps = sorted({s for ss in due.values() for s in ss}) or [len(grid) - 1]
    output = {s: j for j, s in enumerate(steps)}  # grid step -> engine output index
    use_gpu = torch.cuda.is_available() if config.gpu is None else config.gpu
    device = torch.device("cuda" if use_gpu else "cpu")
    off = [obs for obs in config.observables if isinstance(obs, OFF_DIAGONAL)]
    due_outputs = {key: [output[s] for s in ss] for key, ss in due.items()}
    hook = OffDiagonal(off, sds, grid[steps], config, due_outputs) if off else None  # psi only when asked for
    initial = initial_bits(config.initial_state, sd0)

    optimization = config.optimization
    if optimization in ("auto", "quench"):
        blocker = quench_blocker(sds, initial)
        if optimization == "quench" and blocker:
            raise ValueError(f'optimization="quench" cannot run this sequence: {blocker}')
        optimization = "turbo" if blocker else "quench"
    if optimization == "quench":
        raw = run_quench(sds, grid[steps], device, config.bounds, hook)
    else:
        kwargs = dict(seed=config.seed, device=device, initial_state=initial, on_state=hook)
        if optimization != "reference":
            kwargs["single"] = config.single
        if optimization == "turbo":
            kwargs["bounds"] = config.bounds
        raw = NOISY[optimization].run_sequence_data(sds, fraction[steps], **kwargs)
    raw.stats["optimization"] = optimization
    raw.stats["off_diagonal_evaluations"] = hook.evaluations if hook else 0
    logging.getLogger("emulators").info(f"emu_cheb ({optimization}): {raw.stats}")

    results = Results(atom_order=sd0.qubit_ids, total_duration=int(grid[-1]))
    for obs in config.observables:
        for s in due[id(obs)]:
            j = output[s]
            value = hook.value(obs, j) if isinstance(obs, OFF_DIAGONAL) else diagonal_value(obs, raw, j, config)
            results._store(observable=obs, time=float(fraction[s]), value=value)
    return results, raw


def heisenberg_samples(sds, couplings) -> list:
    """The samples with the Heisenberg interaction (README.md, "Hamiltonians"): a (3, n, n) stack as is (constant in
    time), or (Jx, Jy, Jz) times the sequence's own matrix U(t) (keeps e.g. an SLM mask). hamiltonian_type XY makes
    the engines read the 3-layer stack as [Jxx, Jyy, Jzz] (as _heisenberg in test/emu_cheb/validation/noisy_vs_emu_mps.py)."""
    J = torch.as_tensor(couplings, dtype=torch.float64)
    n = len(sds[0].qubit_ids)
    if J.ndim == 3 and J.shape[1] != n:
        raise ValueError(f"the heisenberg stack is {tuple(J.shape)} for {n} atoms")

    def replace(sd):
        if J.ndim == 3:
            matrix = lambda t: J  # noqa: E731
        else:
            base = sd.interaction_matrix
            matrix = lambda t: J[:, None, None] * base(t)[0].real.double()  # noqa: E731
        return dataclasses.replace(sd, interaction_matrix=matrix, hamiltonian_type=HamiltonianType.XY)

    return [replace(sd) for sd in sds]  # omega / delta / phi tensors kept: the engines still batch identical samples


def initial_bits(state, sd) -> str | None:
    """Bitstring (atom 0 first, "1" = excited) of a basis-state initial StateVector; None for the default (ground)."""
    if state is None:
        return None
    n = len(sd.qubit_ids)
    if state.data.dim() != 1:
        raise NotImplementedError("emu_cheb starts from a basis state vector, not a density matrix")
    if state.n_qudits != n:
        raise ValueError(f"Mismatch in number of atoms: initial state has {state.n_qudits} and the sequence has {n}")
    if sd.state_prep_error > 0:
        raise NotImplementedError("Initial state and state preparation error can not be together.")
    nonzero = torch.nonzero(state.data.abs() > 1e-12).flatten()
    if len(nonzero) != 1:
        raise NotImplementedError("emu_cheb starts from a basis state: the initial state needs one non-zero amplitude")
    return format(int(nonzero[0]), f"0{n}b")


def quench_blocker(sds, initial) -> str | None:
    """Why the quench solvers cannot run these samples, or None if they can (README.md, "Engines")."""
    sd = sds[0]
    if sd.lindblad_ops or sd.state_prep_error > 0 or len({id(s.omega) for s in sds}) > 1:
        return "noise (Lindblad or shot-to-shot)"
    if initial is not None and "1" in initial:
        return "the initial state is not all ground"
    if bool((torch.as_tensor(sd.phi).real != 0).any()):
        return "a non-zero laser phase (the quench solvers need a real H)"
    if sd.hamiltonian_type != HamiltonianType.Rydberg and sd.interaction_matrix(0.0).shape[0] != 3:
        return "pulser's XY mode (the quench solvers take Rydberg or a heisenberg= stack)"
    if len(noisy_cheb._Model(sd, torch.device("cpu")).pieces) != 1:
        return "a time-dependent drive or interaction (pulse shape, SLM mask)"
    return None


def run_quench(sds, times_ns, device, bounds, hook) -> noisy_cheb.NoisyResults:
    """The noiseless constant-H run by rydberg_cheb_fast (Rydberg) or heisenberg_cheb_fast (Heisenberg stack), as a
    NoisyResults whose trajectories are all this one (the samples are identical). Probabilities at every output time;
    psi only through the solver's on_state hook, i.e. while the series holds it (no T x 2^n array of states)."""
    model = noisy_cheb._Model(sds[0], device)
    _, _, omega, delta, _, mat = model.pieces[0]
    n = model.n_total
    solver, J = (rydberg_cheb_fast, mat[0]) if len(mat) == 1 else (heisenberg_cheb_fast, mat)
    stats: dict = {}
    active = np.ones(n, bool)
    out = solver.evolve(
        J[None],
        omega[None],
        delta[None],
        np.asarray(times_ns) / 1000,  # us
        device=device,
        stats=stats,
        bounds=bounds,
        on_state=(lambda j, psi: hook(j, [0], psi, active)) if hook else None,  # psi (1, 2^n): one realization
    )[:, 0]  # (T, 2^n) probabilities
    occupations, correlations = rydberg_cheb_fast.moments(out, n)
    R = len(sds)
    return noisy_cheb.NoisyResults(
        times_ns=np.asarray(times_ns),
        probabilities=out.cpu(),
        occupations=occupations.cpu()[None].expand(R, -1, -1),
        correlations=correlations.cpu(),
        leaked=torch.zeros(R, len(times_ns), n, dtype=torch.bool),
        stats=stats,
    )
