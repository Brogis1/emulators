"""Observables of emu_cheb (README.md, "Observables").

Diagonal ones (Occupation, CorrelationMatrix, BitStrings) come from the probabilities the engines already compute: no
extra work. Off-diagonal ones (Energy, EnergyVariance, EnergySecondMoment, Expectation, Fidelity, StateResult) need
psi itself: an OffDiagonal instance is handed to the engine as its on_state hook and evaluates them while the engine
holds psi, only when such an observable is requested.
"""
from __future__ import annotations

import numpy as np
import torch
from pulser.backend import (
    CorrelationMatrix,
    Energy,
    EnergySecondMoment,
    EnergyVariance,
    Occupation,
    StateResult,
)

from emu_sv import StateVector

from .solvers import noisy_cheb

_ENERGY = (Energy, EnergyVariance, EnergySecondMoment)


def connected(occupation: torch.Tensor, correlation: torch.Tensor) -> torch.Tensor:
    """Connected correlator <n_i n_j> - <n_i><n_j> from Occupation (n,) and CorrelationMatrix (n, n) values."""
    return correlation - torch.outer(occupation, occupation)


def diagonal_value(obs, raw, j: int, config):
    """Value of a diagonal observable at engine output j, from the engine's NoisyResults `raw` (no extra compute)."""
    if isinstance(obs, Occupation):
        return raw.occupations[:, j].mean(0)  # (n,) trajectory average of <n_i>
    if isinstance(obs, CorrelationMatrix):
        return raw.correlations[j].clone()  # (n, n) trajectory average of <n_i n_j>
    # BitStrings: emu-sv's own sampler and emu-base readout errors, on a state with the same |amplitude|^2 = P(t);
    # n_trajectories x num_shots shots, the total that emu-sv's bag union of per-trajectory samples gives
    shots = (obs._num_shots or config.default_num_shots) * raw.occupations.shape[0]
    state = StateVector(raw.probabilities[j].sqrt().to(torch.complex128), gpu=False)
    return state.sample(
        num_shots=shots,
        one_state=obs.one_state,
        p_false_pos=config.noise_model.p_false_pos,
        p_false_neg=config.noise_model.p_false_neg,
    )


class OffDiagonal:
    """on_state hook of the engines (solvers.noisy_cheb.run_sequence_data): evaluates the off-diagonal observables on
    every trajectory's normalised state at every output time.

    The engine calls the hook at every output time (the union of all observables' times); each observable is
    evaluated only at its own times (`due`), and nothing is done at outputs where none of them is due.
    Values are ASSIGNED per (trajectory, output): a batch that noisy_cheb_turbo reruns (Lanczos guard) overwrites
    instead of counting twice. value() averages over the trajectories that reported (the quench path reports one).

    Args:
        observables: the off-diagonal observables of the config.
        sequences: the SequenceData samples given to the engine (one per trajectory), for Energy's Hamiltonian.
        times_ns: the engine's output times (ns).
        config: the ChebConfig (passed to pulser's Observable.apply).
        due: id(observable) -> the engine output indices at which it is evaluated.
    """

    def __init__(self, observables, sequences, times_ns, config, due):
        self.observables, self.sequences, self.times_ns, self.config = observables, sequences, times_ns, config
        self.due = {id(obs): set(due[id(obs)]) for obs in observables}
        self.n_total = len(sequences[0].qubit_ids)
        self.values: dict = {id(obs): {} for obs in observables}  # id(obs) -> {(trajectory, j): value}
        self.evaluations = 0  # observable evaluations (one per observable, trajectory and due output)
        self._models: dict = {}
        self._hams: dict = {}

    def __call__(self, j, traj_ids, psi, active):
        todo = [obs for obs in self.observables if j in self.due[id(obs)]]
        if not todo:  # an output time of other observables only
            return
        need_h = any(isinstance(obs, _ENERGY) for obs in todo)
        index = noisy_cheb._embedding(np.asarray(active), psi.device)
        for row, k in enumerate(traj_ids):
            v = psi[row].to(torch.complex128)
            v = v / torch.linalg.vector_norm(v)
            full = torch.zeros(2**self.n_total, dtype=torch.complex128, device=psi.device)
            full[index] = v  # inactive (bad, leaked) atoms in their 0 bit, as the engines' probabilities
            state = StateVector(full, gpu=full.is_cuda)
            hv = self._apply_h(k, j, active, v) if need_h else None
            for obs in todo:
                self.values[id(obs)][(k, j)] = self._evaluate(obs, state, v, hv)
                self.evaluations += 1

    def value(self, obs, j: int):
        """Trajectory average of obs at output j (the state itself for StateResult)."""
        vals = [v for (k, jj), v in sorted(self.values[id(obs)].items()) if jj == j]
        if isinstance(obs, StateResult):
            return vals[0]
        return torch.stack([torch.as_tensor(v).cpu() for v in vals]).mean(0)

    def _evaluate(self, obs, state, v, hv):
        if isinstance(obs, Energy):
            return torch.vdot(v, hv).real
        if isinstance(obs, EnergySecondMoment):
            return torch.vdot(hv, hv).real
        if isinstance(obs, EnergyVariance):
            return torch.vdot(hv, hv).real - torch.vdot(v, hv).real ** 2
        return obs.apply(config=self.config, state=state, hamiltonian=None)  # Expectation, Fidelity, StateResult

    def _apply_h(self, k, j, active, v):
        """H v on the active atoms, H the Hermitian Hamiltonian (no -iG) of trajectory k's drive piece ending at
        output j (the first piece at t = 0): emu-sv's Hamiltonian of the step just completed."""
        sd = self.sequences[k]
        key = (id(sd.omega), id(sd.delta), id(sd.phi), sd.bad_atoms)  # the engines' own sample identity
        if key not in self._models:
            self._models[key] = noisy_cheb._Model(sd, v.device)
        model = self._models[key]
        t = self.times_ns[j]
        p = 0 if t <= 0 else next(q for q, piece in enumerate(model.pieces) if piece[0] + 1e-9 < t <= piece[1] + 1e-9)
        hkey = (key, p, tuple(bool(a) for a in active))
        if hkey not in self._hams:
            if len(self._hams) >= 64:  # bounded: shot-to-shot runs have one H per sample and piece
                self._hams.clear()
            idx = np.nonzero(active)[0]
            _, _, omega, delta, phi, mat = model.pieces[p]
            # noisy_cheb._build with G = 0: H~ = (H - c) / w, c real
            self._hams[hkey] = noisy_cheb._build(
                omega[idx], delta[idx], phi[idx], mat[:, idx][:, :, idx], model.ops, np.zeros((2, 2), complex), v.device
            )
        h = self._hams[hkey]
        out = torch.empty_like(v)
        h.step(v, None, out)
        return h.w * out + h.center.real * v
