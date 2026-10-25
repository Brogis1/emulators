"""noisy_cheb: exact quantum-trajectory emulation of pulser sequences with emu-mps' noise models, Chebyshev propagation.

It consumes the SequenceData that emu-base's PulserData produces for emu-mps (one per trajectory: drive samples,
interaction matrix, bad atoms, Lindblad operators), so every pulser noise channel arrives exactly as in emu-mps:
shot-to-shot noise (amplitude, laser waist, detuning, high-frequency detuning, Doppler, register, SPAM state
preparation) is already sampled into each SequenceData; Lindblad noise (dephasing, depolarizing, relaxation, eff_noise,
leakage) comes as local operators L_k; readout errors (SPAM p_false_pos / p_false_neg) are a classical channel on the
probabilities (`readout`), as emu-mps applies them when sampling bitstrings.

Hamiltonian (emu-mps conventions, local basis index 0 = ground / "u", atom 0 = most significant bit):
    H = sum_i Omega_i / 2 (cos phi_i X_i + sin phi_i Y_i) - sum_i delta_i n_i + sum_k sum_{i<j} M_k[i,j] O_k(i) O_k(j)
with (M, O) = (J, n) for Rydberg, (C3, sqrt(2) X/2), (C3, sqrt(2) Y/2), (C6, |0><0|) for XY, and (Jxx, X), (Jyy, Y),
(Jzz, Z) for a 3 x n x n Heisenberg stack. Omega, delta, phi are piecewise constant on the SequenceData time grid.

Method (Monte Carlo wave function, as emu-mps, without time step or truncation error):
1. Between jumps psi evolves under H_eff = H - i G, G = 1/2 sum_i sum_k L_k(i)^dag L_k(i), and jumps when
   ||psi(t)||^2 reaches a uniform random threshold r; the jump (qubit i, operator k) is drawn with weight
   ||L_k(i) psi||^2, then psi is normalised and a new r drawn.
2. H_eff is not Hermitian, but with H~ = (H_eff - c) / w the Jacobi-Anger series
       e^{-i H_eff t} psi = e^{-i c t} sum_k (2 - [k = 0]) (-i)^k J_k(w t) T_k(H~) psi
   still holds (it is an identity of entire functions). The numerical range of H~ lies in the rectangle
   [-1, 1] x i[-beta, beta] (Gershgorin interval of the Hermitian part; G's local spectrum, centred: the constant part
   of G is the decay e^{-Im(c) t}, exact), hence in the ellipse of foci +/-1 and parameter rho(beta), where
   ||T_k(H~)|| <= (1 + sqrt 2) rho^k (Crouzeix). The series is cut where |J_k(w t)| rho^k < 1e-16 and the computed
   ||T_k psi|| are checked against that bound. For Pauli noise (L^dag L proportional to 1) beta = 0 and rho = 1.
3. One series per segment gives psi at all its probe times (the coefficients alone depend on t): the segment
   [t0, t1] (t1 = next output time or piece boundary, at most (PROBES_MAX - PROBES_MIN) / w long) is probed at
   M + 1 Chebyshev-Lobatto times, M = PROBES_MIN + w (t1 - t0). The norm is non-increasing and smooth; if it falls
   below r, the jump time is the root of its polynomial interpolant (bracketed by the probes), and psi at that time
   is one short series from the probe before it. stats["jump_norm_dev"] = max |N(t_jump) - r| (measured ~1e-13).
4. Leakage (qutrit basis g, r, x): an operator into x makes the atom dark (nothing drives, couples to or leaves x);
   the trajectory continues with that atom removed and reported as leaked. Operators must either stay in {g, r} or
   map {g, r} into x, and none may leave x (NotImplementedError otherwise).

Requires emu-base / emu-mps (PulserData, MPSConfig) for `simulate`; the engine itself only needs torch and numpy.
Self-check: python3 noisy_cheb.py (single trajectories against dense quantum jumps with the same random numbers,
trajectory averages against the dense Lindblad equation; 2-3 atoms, CPU).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Sequence

import numpy as np
import torch

import rydberg_cheb_fast as ryd

CROUZEIX = 1 + 2 ** 0.5  # ||f(A)|| <= CROUZEIX max_{W(A)} |f|
PROBES_MIN, PROBES_MAX = 16, 64  # probes per segment: PROBES_MIN + w (t1 - t0), segments capped accordingly
GROWTH_SLACK = 1e-6  # relative slack of the check ||T_k psi|| <= CROUZEIX rho^k ||psi||
BLOCK = 32  # ring length of the series (vectors)


@dataclass
class NoisyResults:
    """Trajectory results at the output times (ns). probabilities: mean over trajectories of |psi(t)|^2 / ||psi||^2
    over the 2^n basis states of all atoms (bad and leaked atoms in bit 0), on the CPU; occupations, leaked:
    per trajectory (n_traj, T, n) <n_i> and leaked flags; correlations: mean <n_i n_j> (T, n, n)."""
    times_ns: np.ndarray
    probabilities: torch.Tensor
    occupations: torch.Tensor
    correlations: torch.Tensor
    leaked: torch.Tensor
    stats: dict = field(default_factory=dict)


def simulate(sequence, noise_model=None, n_trajectories: int = 1, eval_fractions: Sequence[float] = (1.0,),
             dt: int = 10, seed: int = 0, device: str | torch.device = "cpu", prefer_device_noise_model: bool = False,
             replace_sequence_data=None, initial_state: str | None = None) -> NoisyResults:
    """Emulate a pulser sequence like MPSBackend.run: same PulserData sampling, one trajectory per SequenceData.

    Args:
        sequence: pulser Sequence. noise_model: pulser NoiseModel (None: noiseless).
        n_trajectories: as MPSConfig.n_trajectories. eval_fractions: output times as fractions of the duration
            (as emu-mps' evaluation_times). dt: time step of the drive samples (ns), as MPSConfig.dt.
        seed: seed of the trajectories' random numbers (the noise samples come from PulserData, i.e. pulser's RNG).
        prefer_device_noise_model: use the device's default noise model, as MPSConfig.
        replace_sequence_data: optional function SequenceData -> SequenceData applied to each sample (e.g. a
            Heisenberg 3 x n x n interaction stack, as examples/emu_mps/heisenberg.ipynb).
        initial_state: basis state as a bitstring, atom 0 first, 0 = ground ("g" / "u"); default all 0.
    """
    import logging
    import warnings

    from emu_base import PulserData
    from emu_mps import MPSConfig

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # MPSConfig without observables
        config = MPSConfig(dt=dt, noise_model=noise_model, n_trajectories=n_trajectories,
                           prefer_device_noise_model=prefer_device_noise_model, optimize_qubit_ordering=False,
                           log_level=logging.WARNING)
    data = PulserData(sequence=sequence, config=config, dt=dt)
    sequences = data.get_sequences()
    if replace_sequence_data is not None:
        sequences = map(replace_sequence_data, sequences)
    return run_sequence_data(sequences, eval_fractions, seed=seed, device=device, initial_state=initial_state)


def run_sequence_data(sequences, eval_fractions: Sequence[float] = (1.0,), seed: int = 0,
                      device: str | torch.device = "cpu", initial_state: str | None = None) -> NoisyResults:
    """One quantum trajectory per SequenceData of the iterable `sequences` (see the module docstring)."""
    device = torch.device(device)
    rng = np.random.default_rng(seed)
    stats = dict(trajectories=0, jumps=0, leaks=0, segments=0, matvecs=0, jump_norm_dev=0.0, max_growth=0.0)
    cache: dict = {}
    mean = occ_all = leak_all = corr = None
    times_ns = None
    for sd in sequences:
        key = (id(sd.omega), id(sd.delta), id(sd.phi), sd.bad_atoms)
        if key not in cache:  # emu-base repeats the same tensors for Lindblad-only noise: build once
            cache.clear()
            cache[key] = _Model(sd, device)
        model = cache[key]
        if times_ns is None:
            times_ns = model.output_times(eval_fractions)
        P, leaked = model.trajectory(times_ns, rng, stats, initial_state)
        n = model.n_total
        occ, second = ryd.moments(P, n)
        if mean is None:
            mean, corr, occ_all, leak_all = torch.zeros_like(P), torch.zeros_like(second), [], []
        mean += P
        corr += second
        occ_all.append(occ.cpu())
        leak_all.append(leaked)
        stats["trajectories"] += 1
    n_traj = stats["trajectories"]
    return NoisyResults(times_ns=times_ns, probabilities=(mean / n_traj).cpu(), occupations=torch.stack(occ_all),
                        correlations=(corr / n_traj).cpu(), leaked=torch.stack(leak_all), stats=stats)


def readout(probabilities: torch.Tensor, p_false_pos: float, p_false_neg: float) -> torch.Tensor:
    """Measured bitstring distribution (..., 2^n) from P (..., 2^n): every atom independently reads 0 as 1 with
    probability p_false_pos and 1 as 0 with p_false_neg (pulser SPAM, as emu-mps' bitstring sampling)."""
    n = probabilities.shape[-1].bit_length() - 1
    confusion = torch.tensor([[1 - p_false_pos, p_false_neg], [p_false_pos, 1 - p_false_neg]], dtype=probabilities.dtype)
    out = probabilities.reshape(-1, *([2] * n))
    for i in range(n):  # measured bit b' <- true bit b with confusion[b', b]
        out = torch.movedim(torch.tensordot(out, confusion, dims=([i + 1], [1])), -1, i + 1)
    return out.reshape(probabilities.shape)


# ---------------------------------------------------------------- the model of one SequenceData

class _Model:
    """Pieces of constant H, the noise operators and the trajectory loop of one SequenceData."""

    def __init__(self, sd, device: torch.device):
        self.device = device
        self.target_times = np.asarray(sd.target_times, float)  # ns
        self.n_total = len(sd.qubit_ids)
        self.good = ~np.asarray(sd.bad_atoms, bool) if sd.state_prep_error > 0 else np.ones(self.n_total, bool)
        omega, delta, phi = (np.asarray(x.real if torch.is_tensor(x) else x, float) for x in (sd.omega, sd.delta, sd.phi))
        if any(torch.is_tensor(x) and x.is_complex() and float(x.imag.abs().max()) > 0 for x in (sd.omega, sd.delta, sd.phi)):
            raise ValueError("omega, delta, phi must be real")
        mats = [sd.interaction_matrix(0.5 * (a + b)) for a, b in zip(self.target_times, self.target_times[1:])]
        mats = [np.asarray(m.real if torch.is_tensor(m) else m, float) for m in mats]
        mats = [m[None] if m.ndim == 2 else m for m in mats]
        self.ops = _interaction_ops(sd.hamiltonian_type, mats[0].shape[0])
        if len(self.ops) == 3 and mats[0].shape[0] == 2:  # pulser XY stack: XX and YY share C3
            mats = [np.stack([m[0], m[0], m[1]]) for m in mats]
        # pieces: maximal runs of steps with the same drive and interaction matrix
        self.pieces = []  # (t_start, t_end (ns), omega, delta, phi, matrix)
        for s in range(len(omega)):
            same = self.pieces and all(np.array_equal(a, b) for a, b in zip(
                (omega[s], delta[s], phi[s], mats[s]), self.pieces[-1][2:]))
            if same:
                self.pieces[-1][1] = self.target_times[s + 1]
            else:
                self.pieces.append([self.target_times[s], self.target_times[s + 1], omega[s], delta[s], phi[s], mats[s]])
        self.lindblad = [np.asarray(L, complex) for L in sd.lindblad_ops]
        self.inside, self.leak = _split_lindblad(self.lindblad)
        g_loc = sum((L[:2, :2].conj().T @ L[:2, :2] + (L[2:, :2].conj().T @ L[2:, :2] if len(L) > 2 else 0))
                    for L in self.lindblad) if self.lindblad else np.zeros((2, 2))
        self.g_loc = 0.5 * np.asarray(g_loc, complex)  # G = sum_i g_loc(i)
        self.jump_ldl = [L[:2, :2].conj().T @ L[:2, :2] for L in self.inside] + \
                        [L[2:, :2].conj().T @ L[2:, :2] for L in self.leak]
        self._hams: dict = {}

    def output_times(self, fractions: Sequence[float]) -> np.ndarray:
        """Target times nearest to the fractions of the duration (emu-mps evaluates observables on that grid)."""
        total = self.target_times[-1]
        return np.array([self.target_times[np.argmin(np.abs(self.target_times - f * total))] for f in fractions])

    def ham(self, p: int, active: tuple[bool, ...]) -> SimpleNamespace:
        key = (p, active)
        if key not in self._hams:
            if len(self._hams) > 64:
                self._hams.clear()
            idx = np.nonzero(active)[0]
            _, _, omega, delta, phi, mat = self.pieces[p]
            self._hams[key] = _build(omega[idx], delta[idx], phi[idx], mat[:, idx][:, :, idx], self.ops, self.g_loc,
                                     self.device)
        return self._hams[key]

    def initial_index(self, initial_state: str | None, active: np.ndarray) -> int:
        """Index, in the active atoms' basis, of the initial basis state (bitstring over all atoms, default 0...0)."""
        if initial_state is None:
            return 0
        if len(initial_state) != self.n_total or set(initial_state) - {"0", "1"}:
            raise ValueError(f"initial_state must be a bitstring of {self.n_total} characters 0 / 1")
        return int("".join(b for b, a in zip(initial_state, active) if a) or "0", 2)

    def trajectory(self, out_times: np.ndarray, rng: np.random.Generator, stats: dict,
                   initial_state: str | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """(P (T, 2^n_total), leaked (T, n_total)) of one trajectory."""
        active = self.good.copy()  # atoms still in the simulation (not bad, not leaked)
        leaked = np.zeros(self.n_total, bool)
        T = len(out_times)
        P = torch.zeros(T, 2 ** self.n_total, dtype=torch.float64, device=self.device)
        leak_out = torch.zeros(T, self.n_total, dtype=torch.bool)
        psi = torch.zeros(2 ** int(active.sum()), dtype=torch.complex128, device=self.device)
        psi[self.initial_index(initial_state, active)] = 1.0
        r = rng.random()
        t, j = 0.0, 0  # current time (ns), next output index

        def record():
            nonlocal j
            while j < T and abs(out_times[j] - t) < 1e-9:
                p = psi.abs().square()
                P[j, _embedding(active, self.device)] = p / p.sum()
                leak_out[j] = torch.as_tensor(leaked)
                j += 1

        record()
        for p, (ta, tb, *_) in enumerate(self.pieces):
            while t < tb - 1e-9:
                h = self.ham(p, tuple(active))
                next_out = out_times[j] if j < T else np.inf
                t1 = min(tb, next_out if next_out > t + 1e-9 else np.inf, t + 1000 * (PROBES_MAX - PROBES_MIN) / h.w)
                delta_us = (t1 - t) / 1000
                m = int(PROBES_MIN + math.ceil(h.w * delta_us))
                nodes = delta_us * (1 - np.cos(np.pi * np.arange(m + 1) / m)) / 2  # Chebyshev-Lobatto, increasing
                states = _propagate(h, psi, nodes, stats)  # (m + 1, dim)
                norms = states.abs().square().sum(1).cpu().numpy()
                stats["segments"] += 1
                if norms[-1] > r:
                    psi, t = states[-1].clone(), t1
                    del states
                    record()
                    continue
                k = int(np.nonzero(norms <= r)[0][0])  # first probe below the threshold (k >= 1: norms[0] > r)
                tau = _interpolated_root(nodes, norms, r, k - 1)
                psi = _propagate(h, states[k - 1].clone(), np.array([tau - nodes[k - 1]]), stats)[0]
                del states
                stats["jump_norm_dev"] = max(stats["jump_norm_dev"], abs(float(psi.abs().square().sum()) - r))
                t += 1000 * tau
                psi = self._jump(psi, active, leaked, rng, stats)
                r = rng.random()
                record()
        return P, leak_out

    def _jump(self, psi, active, leaked, rng, stats):
        """Draw (qubit, operator) with weight ||L psi||^2, apply it, normalise; leakage removes the atom."""
        n = int(active.sum())
        rho = [_reduced(psi, i, n) for i in range(n)]  # 2 x 2 per active qubit
        ops = [(i, k) for i in range(n) for k in range(len(self.jump_ldl))]
        weights = np.array([max(float(np.real(np.trace(self.jump_ldl[k] @ rho[i]))), 0.0) for i, k in ops])
        i, k = ops[int(np.searchsorted(np.cumsum(weights) / weights.sum(), rng.random(), side="right").clip(0, len(ops) - 1))]
        stats["jumps"] += 1
        v = psi.view(2 ** i, 2, -1)
        if k < len(self.inside):
            L = torch.as_tensor(self.inside[k][:2, :2], device=psi.device)
            psi = torch.einsum("bc,acd->abd", L, v).reshape(-1)
        else:  # leakage into x: the remaining atoms carry sum_b L[x, b] psi(..b..)
            L = torch.as_tensor(self.leak[k - len(self.inside)], device=psi.device)
            psi = (L[2, 0] * v[:, 0] + L[2, 1] * v[:, 1]).reshape(-1)
            atom = np.nonzero(active)[0][i]
            active[atom], leaked[atom] = False, True
            stats["leaks"] += 1
        return psi / psi.norm()


def _interaction_ops(kind, stack: int) -> list[np.ndarray]:
    """Local operators O_k of the interaction terms M_k[i,j] O_k(i) O_k(j) (emu-mps' interaction_ops)."""
    from emu_base import HamiltonianType
    X = np.array([[0, 1], [1, 0]], complex)
    Y = np.array([[0, -1j], [1j, 0]])
    if kind == HamiltonianType.Rydberg:
        return [np.diag([0.0, 1.0]).astype(complex)]
    if stack == 3:  # Heisenberg [Jxx, Jyy, Jzz]
        return [X, Y, np.diag([1.0, -1.0]).astype(complex)]
    # pulser XY [C3, C6], expanded to [C3, C3, C6] by _Model as in emu-mps
    return [2 ** -0.5 * X, 2 ** -0.5 * Y, np.diag([1.0, 0.0]).astype(complex)]


def _split_lindblad(ops: list[np.ndarray]) -> tuple[list, list]:
    """(operators inside {g, r}, operators {g, r} -> x); NotImplementedError for anything else."""
    inside, leak = [], []
    for L in ops:
        if len(L) == 2 or (np.allclose(L[2, :], 0) and np.allclose(L[:, 2], 0)):
            inside.append(L)
        elif np.allclose(L[:, 2], 0) and np.allclose(L[:2, :], 0):
            leak.append(L)
        else:
            raise NotImplementedError("Lindblad operators must stay in {g, r} or map {g, r} into x, never leave x")
    return inside, leak


def _embedding(active: np.ndarray, device: torch.device) -> torch.Tensor:
    """Indices in the full 2^n_total basis of the active atoms' basis states (inactive atoms in bit 0)."""
    n_total = len(active)
    pos = n_total - 1 - np.nonzero(active)[0]  # bit of each active atom
    n = len(pos)
    local = np.arange(2 ** n)
    full = np.zeros(2 ** n, np.int64)
    for a, b in enumerate(pos):
        full |= ((local >> (n - 1 - a)) & 1) << b
    return torch.as_tensor(full, device=device)


def _reduced(psi: torch.Tensor, i: int, n: int) -> np.ndarray:
    v = psi.view(2 ** i, 2, -1)
    return torch.einsum("abc,adc->bd", v, v.conj()).cpu().numpy()


def _interpolated_root(nodes: np.ndarray, values: np.ndarray, target: float, k: int) -> float:
    """t in [nodes[k], nodes[k + 1]] where the Chebyshev-Lobatto interpolant of values equals target (bisection)."""
    m = len(nodes) - 1
    w = (-1.0) ** np.arange(m + 1)
    w[0] *= 0.5
    w[-1] *= 0.5

    def p(t):
        d = t - nodes
        hit = np.nonzero(d == 0)[0]
        if hit.size:
            return values[hit[0]]
        q = w / d
        return float(q @ values / q.sum())

    lo, hi = nodes[k], nodes[k + 1]
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mid in (lo, hi):
            break
        if p(mid) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------- H~, step and series

def _build(omega, delta, phi, mat, ops, g_loc, device) -> SimpleNamespace:
    """H~ = (H_eff - c) / w for one piece: step, w, complex center c, rho (see the module docstring)."""
    n = len(omega)
    dim = 2 ** n
    bits = [b.to(torch.float64) for b in ryd._occupations(n, device)]
    diag_re = torch.zeros(dim, dtype=torch.float64, device=device)
    same = np.zeros((n, n))
    differ = np.zeros((n, n))
    for M, O in zip(mat, ops):
        M = np.triu(M, 1)
        if np.allclose(O, np.diag(np.diag(O))):
            d0, d1 = np.diag(O).real
            for i, j in zip(*np.nonzero(M)):
                diag_re += M[i, j] * (d0 + (d1 - d0) * bits[i]) * (d0 + (d1 - d0) * bits[j])
        else:
            if not (abs(O[0, 0]) == abs(O[1, 1]) == 0 and np.isclose(O[1, 0] ** 2, O[0, 1] ** 2)
                    and abs((O[1, 0] ** 2).imag) < 1e-12):
                raise NotImplementedError("interaction operators must be diagonal or X / Y-like")
            same += M * (O[1, 0] ** 2).real  # 00 <-> 11
            differ += M * abs(O[1, 0]) ** 2  # 01 <-> 10
    for i in range(n):
        diag_re -= delta[i] * bits[i]
    pairs = np.argwhere(np.abs(same) + np.abs(differ) > 0)
    hop = np.array([[same[i, j], differ[i, j]] for i, j in pairs]).reshape(-1, 2)
    up = omega / 2 * np.exp(1j * phi)  # <1| H |0>
    dn = omega / 2 * np.exp(-1j * phi)  # <0| H |1>

    radius = float(np.abs(omega).sum() / 2 + (np.abs(hop).max(1).sum() if len(hop) else 0.0))
    e_min, e_max = float(diag_re.min()) - radius, float(diag_re.max()) + radius
    lam = np.linalg.eigvalsh(g_loc)
    g_mid, beta_w = 0.5 * (lam[0] + lam[1]), 0.5 * n * (lam[1] - lam[0])
    a_loc = g_loc - g_mid * np.eye(2)  # centred local G: spectrum +/- (lam1 - lam0) / 2
    center = complex(0.5 * (e_max + e_min), -n * g_mid)
    w = max(0.5 * (e_max - e_min), beta_w, 1e-12)
    beta = beta_w / w
    s = 0.5 * (beta ** 2 + math.sqrt(beta ** 4 + 4 * beta ** 2))
    rho = math.sqrt(1 + s) + math.sqrt(s)  # ellipse of foci +/-1 around [-1, 1] x i[-beta, beta]

    diag = ((diag_re - center.real) / w).to(torch.complex128)
    for i in range(n):  # -i (centred G) on the diagonal
        diag += -1j * (a_loc[0, 0] + (a_loc[1, 1] - a_loc[0, 0]) * bits[i]) / w
    t = dict(dtype=torch.complex128, device=device)
    up_s = torch.as_tensor((up - 1j * a_loc[1, 0]) / w, **t)
    dn_s = torch.as_tensor((dn - 1j * a_loc[0, 1]) / w, **t)
    hop_s = torch.as_tensor(hop / w, dtype=torch.float64, device=device)
    step = _make_step(diag, up_s, dn_s, hop_s, pairs, n)
    return SimpleNamespace(step=step, w=w, center=center, rho=rho, n=n,
                           diag=diag, up=up_s, dn=dn_s, hop=hop_s, pairs=pairs)  # tensors: for noisy_cheb_fast


if ryd.triton is not None:
    import triton.language as tl

    @ryd.triton.jit
    def _step_kernel(cur_ptr, prev_ptr, diag_ptr, up_ptr, dn_ptr, bits_ptr, hop_ptr, out_ptr, dim, n_pairs,
                     N_BITS: tl.constexpr, RECURSION: tl.constexpr, BLOCK: tl.constexpr):
        """Complex H~ (interleaved re, im float64): out = 2 H~ cur - prev (RECURSION) or H~ cur, BLOCK basis states.
        H~ = diag + sum_i (up_i |1><0| + dn_i |0><1|) + pair flips (bits[p], amplitude hop[p, 0] equal / hop[p, 1]
        different bits); atom i <-> bit N_BITS - 1 - i."""
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        cr = tl.load(cur_ptr + 2 * offs)
        ci = tl.load(cur_ptr + 2 * offs + 1)
        dr = tl.load(diag_ptr + 2 * offs)
        di = tl.load(diag_ptr + 2 * offs + 1)
        acc_r = dr * cr - di * ci
        acc_i = dr * ci + di * cr
        for i in tl.static_range(N_BITS):
            shift = N_BITS - 1 - i
            src = offs ^ (1 << shift)
            target_one = ((offs >> shift) & 1) == 1
            kr = tl.where(target_one, tl.load(up_ptr + 2 * i), tl.load(dn_ptr + 2 * i))
            ki = tl.where(target_one, tl.load(up_ptr + 2 * i + 1), tl.load(dn_ptr + 2 * i + 1))
            sr = tl.load(cur_ptr + 2 * src)
            si = tl.load(cur_ptr + 2 * src + 1)
            acc_r += kr * sr - ki * si
            acc_i += kr * si + ki * sr
        for p in range(n_pairs):
            bi = tl.load(bits_ptr + 2 * p)
            bj = tl.load(bits_ptr + 2 * p + 1)
            a = tl.where(((offs >> bi) ^ (offs >> bj)) & 1 == 1, tl.load(hop_ptr + 2 * p + 1), tl.load(hop_ptr + 2 * p))
            src = offs ^ ((1 << bi) | (1 << bj))
            acc_r += a * tl.load(cur_ptr + 2 * src)
            acc_i += a * tl.load(cur_ptr + 2 * src + 1)
        if RECURSION:
            acc_r = 2.0 * acc_r - tl.load(prev_ptr + 2 * offs)
            acc_i = 2.0 * acc_i - tl.load(prev_ptr + 2 * offs + 1)
        tl.store(out_ptr + 2 * offs, acc_r)
        tl.store(out_ptr + 2 * offs + 1, acc_i)


def _make_step(diag, up, dn, hop, pairs, n):
    """step(cur, prev, out) on (2^n,) complex: out = 2 H~ cur - prev, or H~ cur if prev is None. Triton kernel on GPU
    for 2^n >= TILE (torch fallback, with a warning, if it fails to compile)."""
    dim = 2 ** n
    device = diag.device
    index = torch.arange(dim, device=device)
    differ = [((index >> (n - 1 - i)) ^ (index >> (n - 1 - j))) & 1 == 1 for i, j in pairs.tolist()]

    def torch_step(cur, prev, out):
        y = diag * cur
        for i in range(n):  # target bit 0 gets dn * (source bit 1), target bit 1 gets up * (source bit 0)
            v, f = y.view(2 ** i, 2, -1), cur.view(2 ** i, 2, -1)
            v[:, 0].add_(f[:, 1], alpha=dn[i].item())
            v[:, 1].add_(f[:, 0], alpha=up[i].item())
        for p, (i, j) in enumerate(pairs.tolist()):
            flipped = cur.view(2 ** i, 2, 2 ** (j - i - 1), 2, -1).flip((1, 3)).reshape(dim)
            y.addcmul_(torch.where(differ[p], hop[p, 1], hop[p, 0]), flipped)
        if prev is None:
            out.copy_(y)
        else:
            torch.sub(y.mul_(2), prev, out=out)

    if not (diag.is_cuda and dim >= ryd.TILE and ryd.USE_TRITON):
        return torch_step
    bits = torch.as_tensor(n - 1 - pairs, dtype=torch.int32, device=device).reshape(-1, 2).contiguous()
    hop = hop.reshape(-1, 2).contiguous()
    real = [torch.view_as_real(x.contiguous()) for x in (diag, up, dn)]
    failed = False

    def step(cur, prev, out):
        nonlocal failed
        if not failed:
            try:
                _step_kernel[(dim // ryd.TILE,)](torch.view_as_real(cur), torch.view_as_real(cur if prev is None else prev),
                                                 *real, bits, hop, torch.view_as_real(out), dim, len(pairs), N_BITS=n,
                                                 RECURSION=prev is not None, BLOCK=ryd.TILE, num_warps=ryd.WARPS)
                return
            except Exception as error:  # first launches compile: fall back, do not crash
                import warnings
                warnings.warn(f"Triton kernel failed ({error!r}): torch step instead (slow)", stacklevel=2)
                failed = True
        torch_step(cur, prev, out)
    return step


def _propagate(h: SimpleNamespace, psi: torch.Tensor, offsets_us: np.ndarray, stats: dict) -> torch.Tensor:
    """States (m, 2^n) e^{-i H_eff s} psi at the offsets s (us) of one piece."""
    alphas = h.w * np.asarray(offsets_us, float)
    coef = torch.as_tensor(ryd._coefficients(alphas, h.rho), device=psi.device)
    coef = torch.view_as_complex(coef.view(len(coef), len(alphas), 2).contiguous())  # (K, m)
    n_terms, dim = coef.shape[0], psi.numel()
    acc = torch.zeros(dim, len(alphas), dtype=torch.complex128, device=psi.device)
    block = min(BLOCK, n_terms)
    ring = torch.empty(block, dim, dtype=torch.complex128, device=psi.device)
    ring[0] = psi
    norm0 = float(psi.norm())
    if n_terms > 1:
        h.step(ring[0], None, ring[1])
    k0, done, i = 0, 0, 2

    def flush(upto):
        acc.addmm_(ring[done:upto].T, coef[k0 + done:k0 + upto])
        k = k0 + upto - 1  # check ||T_k psi|| <= CROUZEIX rho^k ||psi||
        growth = float(ring[upto - 1].norm()) / (CROUZEIX * h.rho ** k * norm0)
        stats["max_growth"] = max(stats["max_growth"], growth)
        if not growth <= 1 + GROWTH_SLACK:
            raise FloatingPointError(f"Chebyshev vectors exceed the ellipse bound ({growth:.3e}): spectral bounds wrong")

    while k0 + i < n_terms:
        if i == block:
            flush(block)
            ring[0].copy_(ring[block - 2])
            ring[1].copy_(ring[block - 1])
            k0, done, i = k0 + block - 2, 2, 2
            continue
        h.step(ring[i - 1], ring[i - 2], ring[i])
        i += 1
    flush(min(i, n_terms - k0))
    stats["matvecs"] += n_terms - 1
    phase = np.exp(-1j * np.longdouble(h.center.real) * np.asarray(offsets_us, np.longdouble)).astype(complex)
    decay = np.exp(h.center.imag * np.asarray(offsets_us, float))  # center.imag = -n g_mid <= 0
    return acc.T * torch.as_tensor(phase * decay, device=psi.device)[:, None]


# ---------------------------------------------------------------- self-check

def _dense_ops(n: int):
    from functools import reduce

    def op(single, i):
        return reduce(np.kron, [single if k == i else np.eye(len(single)) for k in range(n)])
    return op


def _dense_hamiltonian(model: _Model, p: int, active) -> np.ndarray:
    """H of piece p for the active atoms, from Kronecker products (independent of the bit tricks)."""
    idx = np.nonzero(active)[0]
    n = len(idx)
    op = _dense_ops(n)
    _, _, omega, delta, phi, mat = model.pieces[p]
    X = np.array([[0, 1], [1, 0]], complex)
    Y = np.array([[0, -1j], [1j, 0]])
    N = np.diag([0.0, 1.0])
    H = sum(omega[a] / 2 * (np.cos(phi[a]) * op(X, q) + np.sin(phi[a]) * op(Y, q)) - delta[a] * op(N, q)
            for q, a in enumerate(idx))
    for M, O in zip(mat, model.ops):
        for qi, ai in enumerate(idx):
            for qj, aj in enumerate(idx):
                if qi < qj:
                    H = H + M[ai, aj] * op(O, qi) @ op(O, qj)
    return H


def _dense_trajectory(model: _Model, out_times: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One quantum-jump trajectory with dense expm and brentq jump times, consuming rng like _Model.trajectory
    (threshold, then operator choice). Leakage not supported here."""
    from scipy.linalg import expm
    from scipy.optimize import brentq
    active = model.good.copy()
    n = int(active.sum())
    op = _dense_ops(n)
    G = sum(op(model.g_loc, q) for q in range(n))
    psi = np.zeros(2 ** n, complex)
    psi[0] = 1
    r, t, j, out = rng.random(), 0.0, 0, []
    emb = _embedding(active, torch.device("cpu")).numpy()

    def record():
        nonlocal j
        while j < len(out_times) and abs(out_times[j] - t) < 1e-9:
            full = np.zeros(2 ** model.n_total)
            full[emb] = np.abs(psi) ** 2 / np.linalg.norm(psi) ** 2
            out.append(full)
            j += 1

    record()
    for p, (ta, tb, *_) in enumerate(model.pieces):
        Heff = _dense_hamiltonian(model, p, active) - 1j * G
        while t < tb - 1e-9:
            t1 = min(tb, out_times[j] if j < len(out_times) and out_times[j] > t + 1e-9 else np.inf)
            def U(s, Heff=Heff):
                return expm(-1j * Heff * s / 1000)
            end = U(t1 - t) @ psi
            if np.linalg.norm(end) ** 2 > r:
                psi, t = end, t1
                record()
                continue
            s = brentq(lambda s: np.linalg.norm(U(s) @ psi) ** 2 - r, 0, t1 - t, xtol=1e-13, rtol=1e-15)
            psi, t = U(s) @ psi, t + s
            rho = [np.einsum("abc,adc->bd", psi.reshape(2 ** q, 2, -1), psi.conj().reshape(2 ** q, 2, -1))
                   for q in range(n)]
            ops = [(q, k) for q in range(n) for k in range(len(model.inside))]
            weights = np.array([max(np.real(np.trace(model.jump_ldl[k] @ rho[q])), 0) for q, k in ops])
            q, k = ops[int(np.searchsorted(np.cumsum(weights) / weights.sum(), rng.random(), side="right").clip(0, len(ops) - 1))]
            psi = op(model.inside[k][:2, :2], q) @ psi
            psi /= np.linalg.norm(psi)
            r = rng.random()
            record()
    return np.array(out)


def _dense_lindblad(model: _Model, out_times: np.ndarray) -> np.ndarray:
    """Diagonal of rho(t) (T, 2^n_total) of the Lindblad equation (dense superoperators; qubits only)."""
    from scipy.linalg import expm
    active = model.good
    n = int(active.sum())
    op = _dense_ops(n)
    d = 2 ** n
    eye = np.eye(d)
    Ls = [op(L[:2, :2], q) for L in model.lindblad for q in range(n)]
    diss = sum(np.kron(L, L.conj()) - 0.5 * np.kron(L.conj().T @ L, eye) - 0.5 * np.kron(eye, (L.conj().T @ L).T)
               for L in Ls) if Ls else 0
    rho = np.zeros((d, d), complex)
    rho[0, 0] = 1
    emb = _embedding(active, torch.device("cpu")).numpy()
    t, j, out = 0.0, 0, []

    def record():
        nonlocal j
        while j < len(out_times) and abs(out_times[j] - t) < 1e-9:
            full = np.zeros(2 ** model.n_total)
            full[emb] = np.diag(rho).real
            out.append(full)
            j += 1

    record()
    for p, (ta, tb, *_) in enumerate(model.pieces):
        H = _dense_hamiltonian(model, p, active)
        gen = -1j * (np.kron(H, eye) - np.kron(eye, H.T)) + diss
        while t < tb - 1e-9:
            t1 = min(tb, out_times[j] if j < len(out_times) and out_times[j] > t + 1e-9 else np.inf)
            rho = (expm(gen * (t1 - t) / 1000) @ rho.ravel()).reshape(d, d)
            t = t1
            record()
    return np.array(out)


def _fake_sequence_data(n, omega, delta, phi, matrix, lindblad_ops, kind, steps=8, dt=250, bad_atoms=None):
    """A SequenceData by hand: piecewise-constant drive (steps, n) or constant (n,)."""
    from emu_base import HamiltonianType, SequenceData
    c128 = torch.complex128
    def expand(x):
        return torch.as_tensor(np.broadcast_to(x, (steps, n)).copy(), dtype=c128)
    return SequenceData(omega=expand(omega), delta=expand(delta), phi=expand(phi),
                        interaction_matrix=lambda t: torch.as_tensor(matrix), qubit_ids=tuple(f"q{i}" for i in range(n)),
                        bad_atoms=tuple(bad_atoms or (False,) * n), lindblad_ops=[torch.as_tensor(L, dtype=c128) for L in lindblad_ops],
                        state_prep_error=0.1 if bad_atoms else 0.0, target_times=[dt * k for k in range(steps + 1)],
                        eigenstates=["g", "r"] if kind == HamiltonianType.Rydberg else ["u", "d"],
                        hamiltonian_type=kind, has_stochastic_noise=bool(bad_atoms))


if __name__ == "__main__":
    import itertools
    import warnings

    from emu_base import HamiltonianType
    warnings.filterwarnings("ignore", message="long double is float64")
    rng0 = np.random.default_rng(0)
    relax, deph = np.sqrt(2.0) * np.array([[0, 1], [0, 0]], complex), np.sqrt(1.0 / 2) * np.diag([1.0, -1.0])
    weird = 1.5 * np.array([[0.3, 0.5 + 0.2j], [0.1j, -0.2]])  # non-Hermitian, not unitary: general eff_noise
    n = 3
    J = np.triu(rng0.uniform(1, 5, (n, n)), 1)
    J = J + J.T
    cases = {
        "rydberg relaxation + dephasing, piecewise drive": _fake_sequence_data(
            n, rng0.uniform(1, 4, (8, n)), rng0.uniform(-2, 2, (8, n)), 0.0, J[None], [relax, deph], HamiltonianType.Rydberg),
        "rydberg eff_noise (general), phase": _fake_sequence_data(
            n, 3.0, 1.0, rng0.uniform(0, 2 * np.pi, n), J[None], [weird], HamiltonianType.Rydberg),
        "heisenberg relaxation": _fake_sequence_data(
            n, 2.0, 0.5, 0.0, np.stack([J, 0.7 * J, 1.3 * J]) / 4, [relax], HamiltonianType.XY),
        "rydberg bad atom, relaxation": _fake_sequence_data(
            n, 3.0, 1.0, 0.0, J[None], [relax], HamiltonianType.Rydberg, bad_atoms=(False, True, False)),
    }
    for name, sd in cases.items():
        model = _Model(sd, torch.device("cpu"))
        out_times = model.output_times(np.linspace(0, 1, 9))
        # 1. single trajectories, same random numbers: exact agreement
        worst = 0.0
        stats = dict(trajectories=0, jumps=0, leaks=0, segments=0, matvecs=0, jump_norm_dev=0.0, max_growth=0.0)
        for seed in range(4):
            P, _ = model.trajectory(out_times, np.random.default_rng(seed), stats)
            dense = _dense_trajectory(model, out_times, np.random.default_rng(seed))
            worst = max(worst, float(np.abs(P.numpy() - dense).max()))
        print(f"{name}: single trajectories vs dense quantum jumps (same random numbers): max |dP| = {worst:.1e}, "
              f"jumps {stats['jumps']}, jump_norm_dev {stats['jump_norm_dev']:.1e}, max growth {stats['max_growth']:.2f}")
        assert worst < 1e-8 and stats["jumps"] >= 4, (name, worst, stats["jumps"])
        # 2. trajectory average vs Lindblad
        n_traj = 3000
        res = run_sequence_data(itertools.repeat(sd, n_traj), np.linspace(0, 1, 9), seed=1)
        exact = _dense_lindblad(model, out_times)
        occ_exact = torch.as_tensor(exact) @ ryd.moments(torch.eye(2 ** n, dtype=torch.float64), n)[0]
        mean, err = res.occupations.mean(0), res.occupations.std(0) / n_traj ** 0.5
        z = float(((mean - occ_exact).abs() / err.clamp(min=1e-3)).max())
        print(f"    {n_traj} trajectories vs dense Lindblad: max |<n_i> error| = {float((mean - occ_exact).abs().max()):.3f}, "
              f"max error / std. error = {z:.1f}, max |P error| = {float((res.probabilities - torch.as_tensor(exact)).abs().max()):.3f}")
        assert z < 5, (name, z)
    print("self-check OK")
