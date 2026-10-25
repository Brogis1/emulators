"""heisenberg_cheb_fast: exact, batched quench dynamics of the Heisenberg model on GPU, fast Chebyshev propagation.

Same method, memory planner, Lanczos guard and accuracy checks as rydberg_cheb_fast (read its docstring), whose helpers
are imported, for R independent, constant Hamiltonians (convention of examples/emu_mps/heisenberg.py)

    H = sum_i Omega_i / 2 X_i  -  sum_i delta_i n_i  +  sum_{i<j} Jxx_ij X_i X_j + Jyy_ij Y_i Y_j + Jzz_ij Z_i Z_j

(rad/us, times in us, n = (1 - Z) / 2) from |0...0>; same outputs as rydberg_cheb_fast.evolve (atom 0 = most
significant bit). H is real symmetric: Z_i Z_j is diagonal, and X_i X_j + Y_i Y_j flips the bits of atoms i and j with
amplitude Jxx - Jyy when they are equal (00 <-> 11), Jxx + Jyy when they differ (01 <-> 10). Only the pairs where some
realization has Jxx or Jyy nonzero are applied: a nearest-neighbour chain costs n - 1 pair flips per step, all-to-all
n (n - 1) / 2. A laser phase phi would add Y_i terms and make H complex: not supported (it is a gauge only when it
commutes with the couplings, e.g. uniform phi and Jxx = Jyy).

Noise (evolve_noisy, evolve_noisy_rydberg). Pulser's dephasing (Lindblad operator sqrt(g/2) Z_i) and depolarizing
(sqrt(g/4) X_i, Y_i, Z_i) noise, by quantum trajectories. For a Pauli operator L^dag L is proportional to the identity,
so the no-jump Hamiltonian H - i/2 sum L^dag L is H plus a constant: between jumps the state evolves under the noiseless
H (one Chebyshev series per segment) and the jumps do not depend on the state. Each trajectory is a Poisson process of
Pauli kicks (per qubit: Z at rate g_deph / 2 + g_depol / 4, X and Y at g_depol / 4); the trajectory average converges
to the Lindblad solution as 1 / sqrt(n_traj), with no time-step or truncation error. This is the unravelling that the
Monte Carlo wave-function method of emu-mps reduces to for these operators. Relaxation (L = sqrt(g) sigma^-) is NOT
supported: L^dag L = g n makes the no-jump Hamiltonian non-Hermitian (complex spectrum), outside the method. Shot-to-
shot noise (amplitude, detuning, positions) needs no trajectories: one row of `evolve` per sample.
The Rydberg Hamiltonian of rydberg_cheb_fast is a Heisenberg one: with n = (1 - Z) / 2,
sum_{i<j} J_ij n_i n_j = sum_{i<j} J_ij / 4 Z_i Z_j + sum_i (sum_j J_ij / 2) n_i - sum_{i<j} J_ij / 4 (rydberg_as_heisenberg),
equal up to a constant, i.e. a global phase: probabilities are identical.
Self-check: python3 heisenberg_cheb_fast.py (dense matrix exponentials and Lindblad superoperators, CPU).
"""
from __future__ import annotations

import warnings
from types import SimpleNamespace
from typing import Literal, Sequence

import numpy as np
import torch

import rydberg_cheb_fast as ryd
from rydberg_cheb_fast import GROWTH_TOL, HOST_FRACTION, MEMORY_FRACTION, NORM_TOL, TILE, WARPS, moments, triton

__all__ = ["evolve", "evolve_noisy", "evolve_noisy_rydberg", "rydberg_as_heisenberg", "moments"]

if triton is not None:
    import triton.language as tl

    @triton.jit
    def _step_kernel(cur_ptr, prev_ptr, diag_ptr, w_ptr, bits_ptr, hop_ptr, out_ptr, dim, n_pairs,
                     N_BITS: tl.constexpr, RECURSION: tl.constexpr, BLOCK: tl.constexpr, TILES: tl.constexpr):
        """rydberg_cheb_fast._step_kernel plus the pair flips: pair p flips bits (bits[p, 0], bits[p, 1]) with
        amplitude hop[r, p, 0] if they are equal, hop[r, p, 1] if they differ."""
        pid = tl.program_id(0)
        r = pid // TILES
        offs = (pid % TILES) * BLOCK + tl.arange(0, BLOCK)
        base = r.to(tl.int64) * dim
        acc = tl.load(diag_ptr + base + offs) * tl.load(cur_ptr + base + offs)
        for i in tl.static_range(N_BITS):
            acc += tl.load(w_ptr + r * N_BITS + i) * tl.load(cur_ptr + base + (offs ^ (1 << (N_BITS - 1 - i))))
        for p in range(n_pairs):
            bi = tl.load(bits_ptr + 2 * p)
            bj = tl.load(bits_ptr + 2 * p + 1)
            same = tl.load(hop_ptr + 2 * (r * n_pairs + p))
            differ = tl.load(hop_ptr + 2 * (r * n_pairs + p) + 1)
            amplitude = tl.where(((offs >> bi) ^ (offs >> bj)) & 1 == 1, differ, same)
            acc += amplitude * tl.load(cur_ptr + base + (offs ^ ((1 << bi) | (1 << bj))))
        if RECURSION:
            acc = 2.0 * acc - tl.load(prev_ptr + base + offs)
        tl.store(out_ptr + base + offs, acc)


def evolve(J: np.ndarray, omega: np.ndarray, delta: np.ndarray, times_us: Sequence[float],
           device: str | torch.device = "cuda", stats: dict | None = None,
           output: Literal["probabilities", "state"] = "probabilities",
           bounds: Literal["gershgorin", "lanczos"] = "gershgorin", memory_bytes: float | None = None) -> torch.Tensor:
    """Probabilities |psi_r(t)|^2 (or amplitudes psi_r(t)) of R Heisenberg quenches from |0...0>, at increasing times.

    Args:
        J: (R, 3, n, n) couplings [Jxx, Jyy, Jzz] in rad/us, symmetric with zero diagonal (upper triangle read).
        omega, delta, times_us, device, stats, output, bounds, memory_bytes: as in rydberg_cheb_fast.evolve.

    Returns, Raises: as rydberg_cheb_fast.evolve.
    """
    times = _validated_times(times_us)
    J, omega, delta = _validated(J, omega, delta)
    if output not in ("probabilities", "state") or bounds not in ("gershgorin", "lanczos"):
        raise ValueError('output must be "probabilities" or "state", bounds "gershgorin" or "lanczos"')
    device = torch.device(device)
    if device.type == "cuda" and device.index is not None and device.index != torch.cuda.current_device():
        with torch.cuda.device(device):  # Triton launches on the current device, not on the tensors' device
            return evolve(J, omega, delta, times, device, stats, output, bounds, memory_bytes)
    n_real, n = omega.shape
    dim, rows = 2 ** n, n_real * 2 ** n
    ham = _hamiltonian(J, omega, delta, device, bounds)
    step, center, half_width = ham.step, ham.center, ham.half_width

    out_vectors = len(times) * (1 if output == "probabilities" else 2)
    budget, host_free = _budget(device, memory_bytes)
    host_vectors = HOST_FRACTION * host_free / (8 * rows) if device.type == "cuda" and host_free is not None else 0.0
    segments, block, on_host = ryd._plan(times, half_width, budget / (8 * rows), out_vectors, host_vectors)
    if len(segments) > 3:
        warnings.warn(f"{len(segments)} series restarted from |0...0> (memory short: device budget {budget / 1e9:.3g} GB,"
                      f" output on host {on_host}): several times slower than one series", stacklevel=2)

    dtype = torch.float64 if output == "probabilities" else torch.complex128
    result, matvecs, max_norm2, norm_dev = None, 0, 0.0, 0.0
    for j0, j1 in segments:
        acc, n_terms, norm2 = ryd._series(step, n_real, dim, half_width * times[j0:j1], block, device)
        matvecs += n_terms - 1
        max_norm2 = float("nan") if norm2 != norm2 else max(max_norm2, norm2)
        if not max_norm2 <= 1 + GROWTH_TOL:
            del acc
            break
        if result is None:
            result = torch.empty(len(times), n_real, dim, dtype=dtype, device="cpu" if on_host else device)
        z = acc.view(rows, j1 - j0, 2)  # (Re, Im) of sum_k c_k(t_j) T_k, without the phase e^{-i c t_j}
        psi = torch.view_as_complex(z)
        phases = ryd._phases(center, times[j0:j1])
        for j in range(j1 - j0):
            p = z[:, j, 0].square().addcmul_(z[:, j, 1], z[:, j, 1])
            dev = float((p.view(n_real, dim).sum(1) - 1).abs().max())
            norm_dev = max(norm_dev, dev if dev == dev else float("inf"))
            target = result[j0 + j].view(rows)
            if output == "probabilities":
                target.copy_(p)
            elif on_host:
                target.copy_(psi[:, j] * phases[j])
            else:
                torch.mul(psi[:, j], phases[j], out=target)
            del p
        del acc, z, psi

    if not max_norm2 <= 1 + GROWTH_TOL:
        if bounds == "gershgorin":
            raise FloatingPointError(f"Chebyshev vectors grew (max ||T_k||^2 = {max_norm2:.3e}) with Gershgorin bounds")
        rejected = {"lanczos_rejected": max_norm2, "rejected_matvecs": ham.lanczos_matvecs + matvecs}
        del result, ham, step
        out = evolve(J, omega, delta, times, device, stats, output, "gershgorin", memory_bytes)
        if stats is not None:
            stats.update(rejected)
        return out
    if on_host:
        result = result.to(device)
    if stats is not None:
        stats.update(matvecs=matvecs, lanczos_matvecs=ham.lanczos_matvecs, half_width=half_width, center=center,
                     norm_dev=norm_dev, max_norm2=max_norm2, bounds_used=bounds, segments=len(segments), block=block,
                     output_on_host=on_host, budget_bytes=budget, pairs=ham.pairs,
                     longdouble_eps=float(np.finfo(np.longdouble).eps))
    if norm_dev > NORM_TOL:
        raise FloatingPointError(f"max |sum P - 1| = {norm_dev:.1e} > {NORM_TOL}")
    return result


def evolve_noisy(J: np.ndarray, omega: np.ndarray, delta: np.ndarray, times_us: Sequence[float],
                 dephasing_rate: float = 0.0, depolarizing_rate: float = 0.0, n_traj: int = 100, seed: int = 0,
                 initial_state: int = 0, device: str | torch.device = "cuda",
                 bounds: Literal["gershgorin", "lanczos"] = "gershgorin", memory_bytes: float | None = None,
                 stats: dict | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Heisenberg quench with pulser dephasing / depolarizing noise, by n_traj quantum trajectories (module docstring).

    Args:
        J: (3, n, n) couplings [Jxx, Jyy, Jzz]; omega, delta: (n,); one Hamiltonian, units as in `evolve`.
        times_us: output times, as in `evolve`.
        dephasing_rate, depolarizing_rate: pulser NoiseModel rates in 1/us (Lindblad operators sqrt(g/2) Z_i and
            sqrt(g/4) X_i, Y_i, Z_i on every qubit).
        n_traj, seed: number of trajectories and seed of their (numpy) random kicks.
        initial_state: basis state |x> the trajectories start from (integer x, atom 0 = most significant bit).
        device, bounds, memory_bytes: as in `evolve`.
        stats: optional dict, filled with "matvecs", "lanczos_matvecs", "kicks", "segments", "norm_dev" (max |sum P - 1|
            over trajectories and times), "half_width", "center", "bounds_used", "block", "pairs", and after a rejected
            Lanczos interval "lanczos_rejected".

    Returns:
        (mean probabilities (T, 2^n) float64 on `device`, the trajectory average of |psi(t)|^2, i.e. the diagonal of
        the Lindblad density matrix up to statistical error; occupations <n_i> of every trajectory (n_traj, T, n) on
        the CPU, for error bars: std(0) / sqrt(n_traj)).
    """
    times = _validated_times(times_us)
    J, omega, delta = _validated(np.asarray(J)[None], np.asarray(omega)[None], np.asarray(delta)[None])
    if not (dephasing_rate >= 0 and depolarizing_rate >= 0 and n_traj >= 1):
        raise ValueError("rates must be non-negative and n_traj >= 1")
    if bounds not in ("gershgorin", "lanczos"):
        raise ValueError('bounds must be "gershgorin" or "lanczos"')
    device = torch.device(device)
    if device.type == "cuda" and device.index is not None and device.index != torch.cuda.current_device():
        with torch.cuda.device(device):
            return evolve_noisy(J[0], omega[0], delta[0], times, dephasing_rate, depolarizing_rate, n_traj, seed,
                                initial_state, device, bounds, memory_bytes, stats)
    n = omega.shape[1]
    dim, T = 2 ** n, len(times)
    if not 0 <= initial_state < dim:
        raise ValueError(f"initial_state must be in [0, 2^n) = [0, {dim})")
    ham = _hamiltonian(J, omega, delta, device, bounds)
    lanczos_matvecs = ham.lanczos_matvecs
    budget, _ = _budget(device, memory_bytes)
    # vectors of 2^n float64: mean and trajectory P (2T), segment output and accumulators (4 (T + 1)), psi and kicks
    # (4), diagonal (1), temporaries; the rest is the ring
    block = int(min(ryd.BLOCK_MAX, budget / (8 * dim) - 2 * T - 4 * (T + 1) - 5 - ryd.TEMP_VECTORS))
    if block < ryd.BLOCK_MIN:
        raise MemoryError(f"{T} output times do not fit in {budget / 1e9:.3g} GB beside the series: split the times")
    counters = dict(matvecs=0, kicks=0, segments=0, norm_dev=0.0)

    def propagate(psi, offsets):
        """States (m, 2^n) at t0 + offsets from psi (2^n,) at t0: two real series (Re psi, Im psi), H being real."""
        nonlocal ham
        while True:
            out = torch.zeros(len(offsets), dim, dtype=torch.complex128, device=device)
            norm2 = 0.0
            for part, factor in ((psi.real, 1.0), (psi.imag, 1j)):
                if not bool(part.any()):
                    continue
                acc, n_terms, norm2 = ryd._series(ham.step, 1, dim, ham.half_width * offsets, block, device,
                                                  start=part[None])
                counters["matvecs"] += n_terms - 1
                if not norm2 <= 1 + GROWTH_TOL:
                    break
                out += factor * torch.view_as_complex(acc.view(dim, len(offsets), 2)).T
                del acc
            if norm2 <= 1 + GROWTH_TOL:
                break
            if ham.bounds == "gershgorin":
                raise FloatingPointError(f"Chebyshev vectors grew (max ||T_k||^2 = {norm2:.3e}) with Gershgorin bounds")
            counters["lanczos_rejected"] = norm2
            del out, ham
            ham = _hamiltonian(J, omega, delta, device, "gershgorin")
        counters["segments"] += 1
        phases = torch.tensor(ryd._phases(ham.center, offsets), dtype=torch.complex128, device=device)
        return out.mul_(phases[:, None])

    # kick rates per qubit of X, Y, Z: dephasing sqrt(g/2) Z -> g/2; depolarizing sqrt(g/4) X, Y, Z -> g/4 each
    pauli_rates = np.array([depolarizing_rate / 4, depolarizing_rate / 4, dephasing_rate / 2 + depolarizing_rate / 4])
    rng = np.random.default_rng(seed)
    mean = torch.zeros(T, dim, dtype=torch.float64, device=device)
    occupations = torch.empty(n_traj, T, n, dtype=torch.float64)
    for r in range(n_traj):
        kicks = _draw_kicks(rng, pauli_rates, n, times[-1])
        counters["kicks"] += len(kicks)
        P = torch.empty(T, dim, dtype=torch.float64, device=device)
        psi = torch.zeros(dim, dtype=torch.complex128, device=device)
        psi[initial_state] = 1.0
        t0, j = 0.0, 0
        for t_kick, qubit, pauli in kicks + [(np.inf, 0, 0)]:
            j1 = j + int(np.searchsorted(times[j:], t_kick, side="right"))  # outputs before the kick
            offsets = np.append(times[j:j1], t_kick)[:j1 - j + (t_kick < np.inf)] - t0
            if not len(offsets):
                break
            states = propagate(psi, offsets)
            P[j:j1] = states[:j1 - j].abs().square()
            if j1 > j:
                dev = float((P[j:j1].sum(1) - 1).abs().max())
                counters["norm_dev"] = max(counters["norm_dev"], dev if dev == dev else float("inf"))
            j = j1
            if t_kick == np.inf:
                break
            psi = states[-1].clone()
            del states
            _apply_pauli(psi, qubit, pauli)
            t0 = t_kick
        mean += P
        occupations[r] = moments(P, n)[0].cpu()
        del P
    mean /= n_traj
    if stats is not None:
        stats.update(counters, lanczos_matvecs=lanczos_matvecs, half_width=ham.half_width, center=ham.center,
                     bounds_used=ham.bounds, block=block, pairs=ham.pairs)
    if counters["norm_dev"] > NORM_TOL:
        raise FloatingPointError(f"max |sum P - 1| = {counters['norm_dev']:.1e} > {NORM_TOL}")
    return mean, occupations


def rydberg_as_heisenberg(J: np.ndarray, omega: np.ndarray, delta: np.ndarray) -> tuple[np.ndarray, np.ndarray,
                                                                                         np.ndarray]:
    """(J3, omega, delta') of the Heisenberg H equal to the Rydberg H of rydberg_cheb_fast plus sum_{i<j} J_ij / 4:
    Jxx = Jyy = 0, Jzz = J / 4, delta'_i = delta_i - sum_j J_ij / 2. J (..., n, n) -> J3 (..., 3, n, n)."""
    J, delta = np.asarray(J, float), np.asarray(delta, float)
    zero = np.zeros_like(J)
    return np.stack([zero, zero, J / 4], -3), np.asarray(omega, float), delta - J.sum(-1) / 2


def evolve_noisy_rydberg(J: np.ndarray, omega: np.ndarray, delta: np.ndarray, times_us: Sequence[float],
                         **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
    """evolve_noisy for the Rydberg H of rydberg_cheb_fast: J (n, n) = C6 / r^6 (rad/us), omega, delta (n,).
    Pauli noise is the same in the pulser (r, g) basis and here (Z and Y change sign, which a Lindblad term ignores)."""
    return evolve_noisy(*rydberg_as_heisenberg(J, omega, delta), times_us, **kwargs)


# ---------------------------------------------------------------- helpers

def _validated_times(times_us: Sequence[float]) -> np.ndarray:
    times = np.asarray(times_us, float)
    if not (times.ndim == 1 and times.size and times[0] >= 0 and np.all(np.diff(times) >= 0)):
        raise ValueError("times_us must be non-negative and non-decreasing")
    return times


def _validated(J, omega, delta) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Real, finite float arrays J (R, 3, n, n) symmetric, omega and delta (R, n); ValueError otherwise."""
    if any(np.iscomplexobj(x) for x in (J, omega, delta)):
        raise ValueError("J, omega and delta must be real (no laser phase: see the module docstring)")
    J, omega, delta = (np.asarray(x, float) for x in (J, omega, delta))
    if omega.ndim != 2 or delta.shape != omega.shape or J.shape != (omega.shape[0], 3, omega.shape[1], omega.shape[1]):
        raise ValueError("expected J (R, 3, n, n), omega (R, n), delta (R, n)")
    if not (np.isfinite(J).all() and np.isfinite(omega).all() and np.isfinite(delta).all()):
        raise ValueError("J, omega and delta must be finite")
    if not np.allclose(J, np.swapaxes(J, -1, -2)):
        raise ValueError("J must be symmetric")
    return J, omega, delta


def _budget(device: torch.device, memory_bytes: float | None) -> tuple[float, float | None]:
    """(bytes the solver may use on `device`, free host bytes or None), as in rydberg_cheb_fast.evolve."""
    if device.type == "cuda":
        one = torch.ones(1, 1, dtype=torch.float64, device=device)
        one.mm(one)  # creates the cuBLAS handle and workspace now: cublasCreate fails once the budget is taken
        del one
        torch.cuda.empty_cache()
    host_free = ryd._host_free_bytes()
    if memory_bytes is not None:
        return memory_bytes, host_free
    if device.type == "cuda":
        return MEMORY_FRACTION * torch.cuda.mem_get_info(device)[0], host_free
    return (HOST_FRACTION * host_free if host_free is not None else float("inf")), host_free


def _hamiltonian(J: np.ndarray, omega: np.ndarray, delta: np.ndarray, device: torch.device,
                 bounds: str) -> SimpleNamespace:
    """H~ = (H - c) / w of R realizations: step, center c, half_width w (rad/us), bounds, lanczos_matvecs, pairs.
    `step` keeps its tensors alive."""
    n_real, n = omega.shape
    dim = 2 ** n
    # Pair flips (i < j) present in some realization; amplitudes (R, P, 2): [Jxx - Jyy (equal bits), Jxx + Jyy].
    pairs = np.argwhere(np.triu(np.abs(J[:, 0]) + np.abs(J[:, 1]), 1).max(0) > 0)
    J_t, omega_t, delta_t = (torch.as_tensor(x, device=device) for x in (J, omega, delta))
    pi, pj = (torch.as_tensor(pairs[:, k], device=device) for k in (0, 1))
    jxx, jyy = J_t[:, 0, pi, pj], J_t[:, 1, pi, pj]
    hop = torch.stack([jxx - jyy, jxx + jyy], -1)

    # Diagonal E(x) and Gershgorin bounds (each pair flip puts one entry, |Jxx -/+ Jyy|, in every row).
    diagonal = torch.empty(n_real, dim, dtype=torch.float64, device=device)
    _fill_diagonal(diagonal, J_t[:, 2], delta_t)
    radius = (0.5 * omega_t).abs().sum(1) + hop.abs().amax(2).sum(1)
    e_min = float((diagonal.min(1).values - radius).min())
    e_max = float((diagonal.max(1).values + radius).max())
    center, half_width = 0.5 * (e_max + e_min), 0.5 * (e_max - e_min)
    if half_width == 0:
        raise ValueError("H is proportional to the identity: psi(t) = e^{-i c t}|0...0>")
    diagonal.sub_(center).div_(half_width)  # in place from here on: the scaled diagonal of H~
    scaled_omega = (0.5 * omega_t / half_width).contiguous()
    scaled_hop = (hop / half_width).contiguous()
    step = _make_step(diagonal, scaled_omega, scaled_hop, pairs, n)

    lanczos_matvecs = 0
    if bounds == "lanczos":
        lo, hi, lanczos_matvecs = ryd._lanczos_interval(step, n_real, dim, device)
        margin = ryd.LANCZOS_MARGIN * 0.5 * (hi - lo)
        lo, hi = max(lo - margin, -1.0), min(hi + margin, 1.0)  # never wider than Gershgorin
        center, half_width = center + half_width * 0.5 * (hi + lo), half_width * 0.5 * (hi - lo)
        _fill_diagonal(diagonal, J_t[:, 2], delta_t)  # rebuilt, in place (`step` keeps its pointers): H~ rounded once
        diagonal.sub_(center).div_(half_width)
        scaled_omega.copy_(0.5 * omega_t / half_width)
        scaled_hop.copy_(hop / half_width)
    return SimpleNamespace(step=step, center=center, half_width=float(half_width), bounds=bounds,
                           lanczos_matvecs=lanczos_matvecs, pairs=len(pairs))


def _draw_kicks(rng: np.random.Generator, pauli_rates: np.ndarray, n: int, t_max: float) -> list[tuple[float, int, int]]:
    """Kicks (time, qubit, Pauli 0/1/2 = X/Y/Z) of one trajectory: Poisson process on [0, t_max] of rate
    n sum(pauli_rates), uniform qubit, Pauli with probabilities proportional to pauli_rates. Sorted by time."""
    total = pauli_rates.sum()
    if total == 0:
        return []
    count = rng.poisson(n * total * t_max)
    times = np.sort(rng.uniform(0.0, t_max, count))
    qubits = rng.integers(n, size=count)
    paulis = rng.choice(3, size=count, p=pauli_rates / total)
    return [(float(t), int(q), int(p)) for t, q, p in zip(times, qubits, paulis)]


def _apply_pauli(psi: torch.Tensor, qubit: int, pauli: int) -> None:
    """psi (2^n,) <- P_qubit psi in place, P = X, Y, Z for pauli = 0, 1, 2 (atom i <-> bit n - 1 - i); Y = i X Z."""
    v = psi.view(2 ** qubit, 2, -1)
    if pauli >= 1:
        v[:, 1].neg_()
    if pauli <= 1:
        v.copy_(v.flip(1))
    if pauli == 1:
        psi.mul_(1j)


def _fill_diagonal(diagonal: torch.Tensor, Jzz: torch.Tensor, delta: torch.Tensor) -> None:
    """diagonal (R, 2^n) <- E(x) = sum_{i<j} Jzz_ij z_i z_j - sum_i delta_i x_i (z = 1 - 2 x), in place."""
    n = delta.shape[1]
    occupied = ryd._occupations(n, diagonal.device)
    diagonal.zero_()
    for i in range(n):
        diagonal -= delta[:, i:i + 1] * occupied[i]
        for j in range(i):
            diagonal += Jzz[:, j, i:i + 1] * (1 - 2 * (occupied[i] ^ occupied[j]).double())


def _make_step(scaled_diagonal: torch.Tensor, scaled_omega: torch.Tensor, scaled_hop: torch.Tensor,
               pairs: np.ndarray, n: int):
    """step(cur, prev, out) of rydberg_cheb_fast._make_step, with the pair flips of scaled_hop (R, P, 2)."""
    n_real, dim = scaled_diagonal.shape
    device = scaled_diagonal.device
    bits = torch.as_tensor(n - 1 - pairs, dtype=torch.int32, device=device).contiguous()  # (P, 2) bit positions

    def torch_step(cur, prev, out):
        # ponytail: recomputes the pair parities each step (~3 extra temporaries beyond the planner's); fallback only
        index = torch.arange(dim, device=device)
        y = scaled_diagonal * cur
        for i in range(n):
            y.addcmul_(scaled_omega[:, i:i + 1], cur.view(n_real, 2 ** i, 2, -1).flip(2).reshape(n_real, dim))
        for p, (i, j) in enumerate(pairs.tolist()):
            flipped = cur.view(n_real, 2 ** i, 2, 2 ** (j - i - 1), 2, -1).flip((2, 4)).reshape(n_real, dim)
            differ = ((index >> (n - 1 - i)) ^ (index >> (n - 1 - j))) & 1 == 1
            y.addcmul_(torch.where(differ, scaled_hop[:, p, 1:], scaled_hop[:, p, :1]), flipped)
        if prev is None:
            out.copy_(y)
        else:
            torch.sub(y.mul_(2), prev, out=out)

    if not (scaled_diagonal.is_cuda and dim >= TILE and ryd.USE_TRITON):
        return torch_step
    if triton is None:
        warnings.warn("triton unavailable: torch step on GPU (slow)", stacklevel=3)
        return torch_step
    failed = False

    def step(cur, prev, out):
        nonlocal failed
        if not failed:
            try:
                _step_kernel[(n_real * (dim // TILE),)](cur, cur if prev is None else prev, scaled_diagonal,
                                                        scaled_omega, bits, scaled_hop, out, dim, len(pairs), N_BITS=n,
                                                        RECURSION=prev is not None, BLOCK=TILE, TILES=dim // TILE,
                                                        num_warps=WARPS)
                return
            except Exception as error:  # first launches compile: fall back, do not crash
                warnings.warn(f"Triton kernel failed ({error!r}): torch step instead (slow)", stacklevel=2)
                failed = True
        torch_step(cur, prev, out)
    return step


# ---------------------------------------------------------------- self-check

_PAULIS = [np.array([[0, 1], [1, 0]], complex), np.array([[0, -1j], [1j, 0]]), np.diag([1.0, -1.0]).astype(complex)]


def _op(single: np.ndarray, i: int, n: int) -> np.ndarray:
    from functools import reduce
    return reduce(np.kron, [single if k == i else np.eye(2) for k in range(n)])


def _dense_hamiltonian(J: np.ndarray, omega: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """H from Kronecker products of Paulis (as heisenberg.py), independent of the bit tricks above."""
    n = len(omega)
    H = sum(omega[i] / 2 * _op(_PAULIS[0], i, n) - delta[i] * _op(np.diag([0.0, 1.0]), i, n) for i in range(n))
    return H + sum(J[a, i, j] * _op(P, i, n) @ _op(P, j, n)
                   for a, P in enumerate(_PAULIS) for i in range(n) for j in range(i + 1, n))


def _dense_reference(J: np.ndarray, omega: np.ndarray, delta: np.ndarray, times_us: Sequence[float]) -> np.ndarray:
    """psi(t) by dense matrix exponentials (small n only)."""
    from scipy.linalg import expm
    H = _dense_hamiltonian(J, omega, delta)
    psi0 = np.zeros(len(H), complex)
    psi0[0] = 1
    return np.array([expm(-1j * H * t) @ psi0 for t in times_us])


def _dense_lindblad_occupations(H: np.ndarray, dephasing_rate: float, depolarizing_rate: float,
                                times_us: Sequence[float], initial_state: int = 0) -> np.ndarray:
    """<n_i>(t) (T, n) of the Lindblad equation with pulser's dephasing / depolarizing operators, by exponentials of
    the dense superoperator (row-major vec: vec(A rho B) = (A kron B^T) vec(rho)); n <= 5."""
    from scipy.linalg import expm
    d = len(H)
    n = d.bit_length() - 1
    eye = np.eye(d)
    gen = -1j * (np.kron(H, eye) - np.kron(eye, H.T))
    for P, rate in ((0, depolarizing_rate / 4), (1, depolarizing_rate / 4), (2, dephasing_rate / 2 + depolarizing_rate / 4)):
        for i in range(n):
            L = _op(_PAULIS[P], i, n)  # rate * (L rho L^dag - rho), L^dag L = 1
            gen += rate * (np.kron(L, L.conj()) - np.eye(d * d))
    rho0 = np.zeros((d, d), complex)
    rho0[initial_state, initial_state] = 1
    occ = ((np.arange(d)[:, None] >> (n - 1 - np.arange(n))) & 1).astype(float)
    return np.array([np.diag((expm(gen * t) @ rho0.ravel()).reshape(d, d)).real @ occ for t in times_us])


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n, times = 7, np.array([0.0, 0.2, 0.715, 0.715, 1.536])
    chain = np.diag(np.ones(n - 1), 1)
    chain = chain + chain.T
    all_to_all = rng.normal(0, 1, (3, n, n))
    J = np.stack([np.stack([1.0 * chain, 0.7 * chain, 1.3 * chain]),  # XYZ chain, as heisenberg.py
                  np.stack([chain, chain, chain]),  # isotropic chain
                  np.triu(all_to_all, 1) + np.swapaxes(np.triu(all_to_all, 1), 1, 2)])  # every pair, Jxx != Jyy
    n_real = len(J)
    omega, delta = rng.uniform(-2, 2, (n_real, n)), rng.uniform(-1, 1, (n_real, n))
    reference = np.stack([_dense_reference(J[r], omega[r], delta[r], times) for r in range(n_real)], 1)
    for kwargs in ({}, {"bounds": "lanczos"}, {"memory_bytes": 8 * n_real * 2 ** n * 19}):  # 19 vectors: segments
        stats: dict = {}
        P = evolve(J, omega, delta, times, device="cpu", stats=stats, **kwargs).numpy()
        psi = evolve(J, omega, delta, times, device="cpu", output="state", bounds=kwargs.get("bounds", "gershgorin"))
        psi = psi.numpy()
        error, error_state = np.abs(P - np.abs(reference) ** 2).max(), np.abs(psi - reference).max()
        print(kwargs, f"max |P - dense expm| = {error:.1e}; max |psi - dense expm| = {error_state:.1e};",
              {k: stats[k] for k in ("matvecs", "segments", "pairs", "norm_dev", "bounds_used")})
        assert error < 1e-12 and error_state < 1e-12, (kwargs, error, error_state)

    # Rydberg as Heisenberg: same probabilities as rydberg_cheb_fast.evolve.
    positions = np.column_stack([np.arange(n) * 6.0, rng.normal(0, 0.5, n)])
    J_ryd = np.stack([ryd.couplings(positions + rng.normal(0, 0.1, positions.shape)) for _ in range(n_real)])
    om_ryd, de_ryd = rng.uniform(-1, 1, (n_real, n)) * 0.3 * J_ryd.max(), rng.uniform(-5, 5, (n_real, n))
    error = (ryd.evolve(J_ryd, om_ryd, de_ryd, times, device="cpu")
             - evolve(*rydberg_as_heisenberg(J_ryd, om_ryd, de_ryd), times, device="cpu")).abs().max()
    print(f"rydberg_as_heisenberg: max |P_rydberg_cheb_fast - P_heisenberg| = {error:.1e}")
    assert error < 1e-12

    # One noisy trajectory against dense exponentials with the same kicks (same seed => same kicks).
    n = 5
    Jh, om, de = J[2][:, :n, :n], omega[2, :n], delta[2, :n]
    noise = dict(dephasing_rate=0.8, depolarizing_rate=0.6)
    stats = {}
    for bounds in ("gershgorin", "lanczos"):
        P, _ = evolve_noisy(Jh, om, de, times, **noise, n_traj=1, seed=3, initial_state=5, device="cpu",
                            bounds=bounds, stats=stats)
        rates = np.array([noise["depolarizing_rate"] / 4] * 2 + [noise["dephasing_rate"] / 2 + noise["depolarizing_rate"] / 4])
        kicks = _draw_kicks(np.random.default_rng(3), rates, n, times[-1])
        from scipy.linalg import expm
        H = _dense_hamiltonian(Jh, om, de)
        psi, t0, dense_P = np.eye(2 ** n, dtype=complex)[5], 0.0, []
        events = sorted([(t, 1, j) for j, t in enumerate(times)] + [(t, 0, k) for k, (t, _, _) in enumerate(kicks)],
                        key=lambda e: (e[0], -e[1]))  # outputs before a kick at the same time
        for t, is_output, k in events:
            psi, t0 = expm(-1j * H * (t - t0)) @ psi, t
            if is_output:
                dense_P.append(np.abs(psi) ** 2)
            else:
                psi = _op(_PAULIS[kicks[k][2]], kicks[k][1], n) @ psi
        error = np.abs(P.numpy() - np.array(dense_P)).max()
        print(f"one trajectory ({bounds}, {len(kicks)} kicks): max |P - dense expm with the same kicks| = {error:.1e}",
              {k: stats[k] for k in ("matvecs", "segments", "norm_dev", "bounds_used")})
        assert len(kicks) >= 2 and error < 1e-12, (len(kicks), error)

    # Trajectory average against the dense Lindblad equation (Heisenberg and Rydberg), statistical.
    n, n_traj, times = 4, 2000, np.linspace(0, 1.5, 6)
    J_ryd4 = ryd.couplings(np.column_stack([np.arange(n) * 7.0, np.zeros(n)]))
    cases = {"heisenberg": (J[0][:, :n, :n], omega[0, :n], delta[0, :n]),
             "rydberg": rydberg_as_heisenberg(J_ryd4, np.full(n, 6.0), np.full(n, 2.0))}
    for name, (Jc, om, de) in cases.items():
        _, occ = evolve_noisy(Jc, om, de, times, **noise, n_traj=n_traj, seed=1, initial_state=0b0101, device="cpu")
        exact = _dense_lindblad_occupations(_dense_hamiltonian(Jc, om, de), **noise, times_us=times,
                                            initial_state=0b0101)
        mean, err = occ.mean(0).numpy(), occ.std(0).numpy() / np.sqrt(n_traj)
        z = np.abs(mean - exact) / np.maximum(err, 1e-3)
        print(f"{name}: {n_traj} trajectories vs dense Lindblad, max |<n_i> error| = {np.abs(mean - exact).max():.3f}, "
              f"max error / std. error = {z.max():.1f}")
        assert z.max() < 5, (name, z.max())
    print("self-check OK")
