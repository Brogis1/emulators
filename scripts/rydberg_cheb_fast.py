"""rydberg_cheb_fast: exact, batched quench dynamics of Rydberg-atom arrays on GPU, fast Chebyshev propagation.

Replacement for rydberg_cheb.evolve: same arguments, conventions and results to round-off (differences listed below),
faster on one A100-40GB (quench to 4 us, 19 output times, probabilities, R = 1): N = 12, 5x; N = 20, 10x (0.19 s);
N = 24, 9x (3.6 s); N = 26, 8x (21 s; 27 s for states); the 185 MUTAG graphs with N <= 27, 8.6x (335 s against
2891 s; bounds="lanczos": 11x, 262 s). N = 27: 4.5x (106 s, 3 series, output waiting on the host) against
rydberg_cheb called twice, PROVIDED the job has ~26 GB of free host memory (cgroup included) for the 20 GB output.
Without it the output stays on the GPU, with one series per output time (a warning is raised above 3 series), and one
call is SLOWER than rydberg_cheb (601 s against 483 s, N = 27 FM); split the times into two calls instead, as for
rydberg_cheb (10 + 9 times: 89 s, measured with no host memory). On Slurm, request it explicitly (e.g. --mem=48G for
N <= 27): the default job allocation is usually far smaller. Entry point: `evolve(J, omega, delta, times_us)`.
It evolves R independent Hamiltonians at once (each with its own J, Omega, delta: noise batches, parameter scans) of

    H = sum_i Omega_i / 2 X_i  -  sum_i delta_i n_i  +  sum_{i<j} J_ij n_i n_j        (rad/us; times in us)

from |0...0> (all atoms in the ground state) and returns the exact probabilities |psi(t)|^2 of the 2^n basis states
(atom 0 = most significant bit), or with output="state" the complex amplitudes psi(t) (global phase included).
`moments` turns probabilities into <n_i> and <n_i n_j>; `couplings` builds J = C6 / r^6 from positions.
J, Omega and delta must be real (complex inputs raise ValueError): the method below needs a real H. A laser phase phi_i
(term Omega_i / 2 (e^{i phi_i} |1><0|_i + h.c.)) is a diagonal gauge here: with U = exp(i sum_i phi_i n_i),
H_phi = U H U^dagger and U|0...0> = |0...0>, so psi_phi(t)[x] = e^{i sum_i phi_i x_i} psi(t)[x]; the caller applies that
factor to the amplitudes (probabilities are unchanged).

Method (every change is exact in exact arithmetic; the Hamiltonian is CONSTANT, no pulse shape, no decoherence):
1. One real Chebyshev series for all output times. With H~ = (H - c) / w (spectrum mapped into [-1, 1]), H~ is real
   symmetric and |0...0> is real, so the vectors T_k = T_k(H~)|0...0> are real (float64) and
       psi(t) = e^{-i c t} sum_k c_k(t) T_k,    c_k(t) = (2 - [k = 0]) (-i)^k J_k(w t)    (Jacobi-Anger),
   where only the coefficients depend on t. The T_k pass through a ring buffer; every `block` steps one matrix product
   adds them to 2T real accumulators (Re, Im of every psi(t_j)). The series stops where |J_k(w t_max)| < 1e-16.
2. One fused GPU kernel (Triton) per step: T_{k+1} = 2 (d T_k + sum_i w_i X_i T_k) - T_{k-1}, one pass over memory.
   Without Triton (not installed, or failing at its first launch: a warning is raised), on CPU, or for n < 8, a torch
   implementation does the same arithmetic; on GPU it runs at ~0.6x the speed of rydberg_cheb (N = 12-16, measured).
3. Spectral interval: Gershgorin bounds by default (rigorous: diagonal -/+ sum_i |Omega_i| / 2, shared by the batch).
   bounds="lanczos" (opt-in) takes the extreme Ritz values of 60 Lanczos steps widened by 1 % of the half-width, which
   cuts the number of terms by 11-31 % (FM ~30 %, AFM ~12 %, measured N = 12-27), under the guard described below.
   H~ is then rebuilt from the raw diagonal with the final (c, w), so it is rounded once, as with Gershgorin.
The Bessel values J_k come from Miller's backward recurrence in long double (scipy.special.jv errs by up to 4e-14 at the
arguments w t ~ 1000-3000 of a long series); the global phase e^{-i c t} is also evaluated in long double.

Guard of the Lanczos interval. For |x| <= 1, |T_k(x)| <= 1, hence ||T_k|0...0>|| <= 1 for every k when the interval
contains every eigenvalue that |0...0> overlaps. An eigenvalue outside it (x = +/- cosh s) gives a component a cosh(k s)
that grows with k; round-off seeds such components even when a = 0. Such a component is not wrong in itself: the
Jacobi-Anger series converges to e^{-i w t x} for every real x. Its errors are (i) the truncated tail, of order
|J_K| ||T_K|| ~ 1e-16 ||T_K||, (ii) the floating-point summation, of order eps sum_k |c_k| ||T_k|| with
sum_k |c_k| = O(sqrt(w t)) ~ 1e2, and (iii) the propagation of the recurrence's own round-off: an error seeded at step m
is carried as U_{k-m-1}(x), amplified by up to ~1/sin(theta) inside the interval (x = cos theta) and ~1/sinh(s) just
outside it, a term a valid interval shares near its edges. All three are those of a valid interval as long as the
computed ||T_k|| stay O(1). The guard measures ||T_k||^2 of every realization at the end of every block and of the
series, and rejects the interval if it exceeds 1 + GROWTH_TOL (1e-9). With a valid interval that cannot happen
(round-off <= 1e-13), so a rejection proves that the interval misses part of the spectrum reached by the computed
vectors, round-off included (possibly only sectors |0...0> does not overlap: the cost is a needless restart); the run
then restarts with Gershgorin bounds (stats["lanczos_rejected"]). When the guard is silent, the outside components grow
(essentially) monotonically in k, so the final check bounds them at every k: max_k ||T_k||^2 <= 2 + GROWTH_TOL and the
error stays that of a valid interval within a factor ~sqrt(2). Since T_k^2 = (1 + T_2k) / 2, ||T_k||^2 ~ 1/2 inside
the interval: the guard only sees outside components of squared norm ~0.5 or more, which the argument tolerates.
This is an argument, not a proof, checked numerically: intervals made too narrow on purpose (n = 8 in the tests;
N = 12 with weights 1e-5 and 1e-36 at the edge; N = 20-22 with |0...0> near the ground state, delta < 0, where every
margin from -5 % to -0.01 % was rejected). The guard does not check that the interval contains the whole spectrum (it
does not need to). The 1 % margin is a choice, not a bound; it makes rejections rare and keeps term (iii) small: in the
N = 20-22, delta < 0 runs, probabilities moved by up to 2e-13 from the Gershgorin run with margins 0 and 0.1 %
(guard silent), by <= 6e-14 with 1 %. Do not lower LANCZOS_MARGIN.

Memory: float64 vectors of R 2^n entries ("vectors" below): the diagonal (1), 2T accumulators, a ring of 8-64 vectors,
the output (T vectors, 2T with output="state") and a few temporaries. When the single series does not fit in the
budget (default MEMORY_FRACTION of the free GPU memory, on CPU HOST_FRACTION of the free host / cgroup memory, or
`memory_bytes`), the output times are split into segments,
each a series restarted from |0...0>, and, when the GPU cannot hold the output beside the series, the finished
segments wait in host memory (checked against the free host / cgroup memory) and the output is copied back at the end.
A cost model picks the split. The returned tensor always lives on `device` and must fit there on its own: N = 27 at 19
times fits for probabilities (20 GB) on a 40 GB A100, not for states (41 GB). stats reports the plan.

Accuracy (measured; t <= 4 us; dense checks at n <= 8 and t <= 2 us; one MUTAG graph per size, FM and AFM recipes,
R > 1 only at N <= 16; probabilities also on all 185 MUTAG graphs N <= 27, FM, R = 1): probabilities within 5e-15 of
dense matrix exponentials and within 8e-14 of rydberg_cheb.evolve (whose own |sum P - 1| reaches 3e-13 there, against
9e-15 here); amplitudes within 2e-14 of dense exponentials, within 2e-13 of rydberg_cheb raw and
6e-14 once the global phase of every (t, r) is aligned. The global phase is accurate to ~eps |c| t (~1e-13 rad at
|c t| ~ 1e3: rounded center, rescaling; raw gaps up to 1e-12 between the two bounds at N = 22, delta < 0, where |c| is
larger), as in rydberg_cheb, not better in general: against an extended-precision
reference (N = 12 and 16, FM) this code is at 7e-14 and 4e-14, rydberg_cheb at 5e-14 and 9e-14.
The run stops if max |sum_x P(x) - 1| exceeds NORM_TOL = 1e-12 (FloatingPointError); measured <= 9e-15.

Differences from rydberg_cheb.evolve: keyword arguments bounds and memory_bytes; group_size accepted and ignored;
NORM_TOL = 1e-12 instead of 1e-10 (can raise where rydberg_cheb passes); MemoryError (output too large for the device)
instead of torch.OutOfMemoryError; ValueError on complex, non-finite or malformed inputs and on H proportional to the
identity; more stats keys, and with bounds="lanczos" or several segments "half_width" and "matvecs" describe the final
interval and all segments; the result is always contiguous on `device`.

Requires numpy, scipy, torch; triton (shipped with CUDA builds of torch) for the fast kernel. x86-64 Linux long double
(80-bit) or better is expected for the Bessel values and the global phase; with a float64 long double a warning is
raised (amplitudes then accurate to ~1e-13 only; stats["longdouble_eps"] records the precision used).
Author: Mathieu Garrigues (Pasqal). Self-check: python3 rydberg_cheb_fast.py (dense matrix exponentials, 7 atoms, CPU).
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import torch
from scipy.linalg import eigh_tridiagonal

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU-only installs: torch implementation of the step
    triton = None

C6_RYD60 = 865723.02  # rad um^6 / us, Rydberg level 60 (public pulser AnalogDevice value)
BESSEL_TOL = 1e-16  # Chebyshev series truncated where |J_k(w t_max)| < BESSEL_TOL
NORM_TOL = 1e-12  # the run stops if max |sum_x P(x) - 1| exceeds this
GROWTH_TOL = 1e-9  # guard: max ||T_k||^2 above 1 + GROWTH_TOL means the interval misses part of the spectrum
LANCZOS_STEPS, LANCZOS_MARGIN = 60, 0.01  # Lanczos steps; margin added on each side, relative to the Ritz half-width
MEMORY_FRACTION = 0.9  # share of the free GPU memory used by default
HOST_FRACTION = 0.8  # share of the free host memory the output may occupy while it waits there
USE_TRITON = triton is not None  # set to False to force the torch step (tests, or a broken Triton install)
TILE, WARPS = 256, 8  # Triton: basis states and warps per program (within 2-7 % of the best setting, N = 14-26)
BLOCK_MIN, BLOCK_MAX = 8, 64  # ring-buffer length bounds
TEMP_VECTORS = 3  # temporaries: torch step (2), output conversion (<= 3)
HOST_COPY_FACTOR = 150  # cost of one byte through PCIe (pageable) in bytes of GPU memory traffic (planner model)

if triton is not None:
    @triton.jit
    def _step_kernel(cur_ptr, prev_ptr, diag_ptr, w_ptr, out_ptr, dim, N_BITS: tl.constexpr,
                     RECURSION: tl.constexpr, BLOCK: tl.constexpr, TILES: tl.constexpr):
        """out = 2 H~ cur - prev (RECURSION) or H~ cur, for BLOCK basis states of one realization. H~ = diag +
        sum_i w_i X_i, atom i <-> bit N_BITS - 1 - i. Flat grid of R * TILES programs (TILES = dim / BLOCK): CUDA caps
        the second and third grid axes at 65535, which R or TILES alone may exceed."""
        pid = tl.program_id(0)
        r = pid // TILES
        offs = (pid % TILES) * BLOCK + tl.arange(0, BLOCK)
        base = r.to(tl.int64) * dim
        acc = tl.load(diag_ptr + base + offs) * tl.load(cur_ptr + base + offs)
        for i in tl.static_range(N_BITS):
            acc += tl.load(w_ptr + r * N_BITS + i) * tl.load(cur_ptr + base + (offs ^ (1 << (N_BITS - 1 - i))))
        if RECURSION:
            acc = 2.0 * acc - tl.load(prev_ptr + base + offs)
        tl.store(out_ptr + base + offs, acc)


def evolve(J: np.ndarray, omega: np.ndarray, delta: np.ndarray, times_us: Sequence[float],
           device: str | torch.device = "cuda", group_size: int = 6, stats: dict | None = None,
           output: Literal["probabilities", "state"] = "probabilities",
           bounds: Literal["gershgorin", "lanczos"] = "gershgorin", memory_bytes: float | None = None,
           on_state=None) -> torch.Tensor:
    """Probabilities |psi_r(t)|^2 (or amplitudes psi_r(t)) of R quenches from |0...0>, at increasing times.

    Args:
        J: (R, n, n) couplings in rad/us, symmetric with zero diagonal (only the upper triangle is read).
        omega: (R, n) Rabi frequencies Omega_i in rad/us (any sign).
        delta: (R, n) detunings delta_i in rad/us.
        times_us: non-negative, non-decreasing output times in us (repeats and t = 0 allowed).
        device: "cuda" or "cpu".
        group_size: accepted for compatibility with rydberg_cheb.evolve and ignored (no dense blocks here).
        stats: optional dict, filled with "matvecs" (H applications of the series, all segments), "lanczos_matvecs",
            "half_width" and "center" (rad/us), "norm_dev" (max |sum P - 1|), "max_norm2" (max ||T_k||^2 checked),
            "bounds_used", "segments" (number of series), "block" (ring length), "output_on_host" (bool), "budget_bytes"
            (device memory the plan could use), "longdouble_eps", and, after a rejected Lanczos interval,
            "lanczos_rejected" (max ||T_k||^2 seen) and "rejected_matvecs" (work discarded).
        output: "probabilities" (default) or "state".
        bounds: "gershgorin" (default, rigorous) or "lanczos" (tighter, guarded; see the module docstring).
        memory_bytes: bytes the solver may use on `device` beyond the Hamiltonian diagonal (default on GPU:
            MEMORY_FRACTION of the free memory; on CPU: HOST_FRACTION of the free host / cgroup memory, no limit if
            unknown).
        on_state: optional hook on_state(j, psi), called for every output index j with psi(t_j) (R, 2^n) complex128
            (exact global phase, read-only) while the series still holds it, whatever `output`: observables that
            need the amplitudes at a few times without returning them all (experimental/emu_cheb). After a rejected
            Lanczos interval it is called again for the rerun's times. None (default): the code path is unchanged.

    Returns:
        Tensor (T, R, 2^n) on `device`, contiguous, basis index with atom 0 as the most significant bit: float64
        probabilities |psi|^2, or complex128 amplitudes psi (exact global phase) with output="state".

    Raises:
        ValueError on malformed, complex or non-finite inputs and on H proportional to the identity; MemoryError if
            the output alone does not fit in the budget (split the times);
        FloatingPointError if max |sum P - 1| > NORM_TOL or the Chebyshev vectors grow with Gershgorin bounds.
    """
    times = np.asarray(times_us, float)
    if not (times.ndim == 1 and times.size and times[0] >= 0 and np.all(np.diff(times) >= 0)):
        raise ValueError("times_us must be non-negative and non-decreasing")
    if any(np.iscomplexobj(x) for x in (J, omega, delta)):
        raise ValueError("J, omega and delta must be real (laser phases: see the module docstring)")
    J, omega, delta = (np.asarray(x, float) for x in (J, omega, delta))
    if omega.ndim != 2 or delta.shape != omega.shape or J.shape != omega.shape + omega.shape[-1:]:
        raise ValueError("expected J (R, n, n), omega (R, n), delta (R, n)")
    if not (np.isfinite(J).all() and np.isfinite(omega).all() and np.isfinite(delta).all()):
        raise ValueError("J, omega and delta must be finite")
    if not np.allclose(J, np.swapaxes(J, -1, -2)):
        raise ValueError("J must be symmetric")
    if output not in ("probabilities", "state") or bounds not in ("gershgorin", "lanczos"):
        raise ValueError('output must be "probabilities" or "state", bounds "gershgorin" or "lanczos"')
    device = torch.device(device)
    if device.type == "cuda" and device.index is not None and device.index != torch.cuda.current_device():
        with torch.cuda.device(device):  # Triton launches on the current device, not on the tensors' device
            return evolve(J, omega, delta, times, device, group_size, stats, output, bounds, memory_bytes, on_state)
    n_real, n = omega.shape
    dim, rows = 2 ** n, n_real * 2 ** n

    # Diagonal E(x) (same construction as rydberg_cheb) and Gershgorin bounds.
    J_t, omega_t, delta_t = (torch.as_tensor(x, device=device) for x in (J, omega, delta))
    diagonal = torch.empty(n_real, dim, dtype=torch.float64, device=device)
    _fill_diagonal(diagonal, J_t, delta_t)
    radius = (0.5 * omega_t).abs().sum(1)
    e_min = float((diagonal.min(1).values - radius).min())
    e_max = float((diagonal.max(1).values + radius).max())
    center, half_width = 0.5 * (e_max + e_min), 0.5 * (e_max - e_min)
    if half_width == 0:
        raise ValueError("H is proportional to the identity (Omega = 0, constant diagonal): psi(t) = e^{-i c t}|0...0>")
    diagonal.sub_(center).div_(half_width)  # in place from here on: the scaled diagonal of H~
    scaled_omega = (0.5 * omega_t / half_width).contiguous()
    step = _make_step(diagonal, scaled_omega, n)

    lanczos_matvecs = 0
    if bounds == "lanczos":
        lo, hi, lanczos_matvecs = _lanczos_interval(step, n_real, dim, device)
        margin = LANCZOS_MARGIN * 0.5 * (hi - lo)
        lo, hi = max(lo - margin, -1.0), min(hi + margin, 1.0)  # never wider than Gershgorin
        center, half_width = center + half_width * 0.5 * (hi + lo), half_width * 0.5 * (hi - lo)
        _fill_diagonal(diagonal, J_t, delta_t)  # rebuilt, in place (`step` keeps its pointers): H~ rounded once
        diagonal.sub_(center).div_(half_width)
        scaled_omega.copy_(0.5 * omega_t / half_width)

    out_vectors = len(times) * (1 if output == "probabilities" else 2)
    if device.type == "cuda":
        one = torch.ones(1, 1, dtype=torch.float64, device=device)
        one.mm(one)  # creates the cuBLAS handle and workspace now: cublasCreate fails once the budget is taken
        del one
        torch.cuda.empty_cache()  # the construction's cached blocks (fragments) go back to the driver
    host_free = _host_free_bytes()  # None if unknown
    budget = memory_bytes
    if budget is None:
        if device.type == "cuda":
            budget = MEMORY_FRACTION * torch.cuda.mem_get_info(device)[0]
        else:
            budget = HOST_FRACTION * host_free if host_free is not None else float("inf")
    host_vectors = HOST_FRACTION * host_free / (8 * rows) if device.type == "cuda" and host_free is not None else 0.0
    segments, block, on_host = _plan(times, half_width, budget / (8 * rows), out_vectors, host_vectors)
    if len(segments) > 3:
        warnings.warn(f"{len(segments)} series restarted from |0...0> (memory short: device budget {budget / 1e9:.3g} GB,"
                      f" output on host {on_host}): several times slower than one series; give the job more host "
                      "memory or split the output times", stacklevel=2)

    dtype = torch.float64 if output == "probabilities" else torch.complex128
    result, matvecs, max_norm2, norm_dev = None, 0, 0.0, 0.0
    for j0, j1 in segments:
        acc, n_terms, norm2 = _series(step, n_real, dim, half_width * times[j0:j1], block, device)
        matvecs += n_terms - 1
        if norm2 != norm2:  # NaN counts as growth
            max_norm2 = float("nan")
        else:
            max_norm2 = max(max_norm2, norm2)
        if not max_norm2 <= 1 + GROWTH_TOL:
            del acc
            break
        if result is None:  # allocated after the first series (its ring is free by now)
            result = torch.empty(len(times), n_real, dim, dtype=dtype, device="cpu" if on_host else device)
        z = acc.view(rows, j1 - j0, 2)  # (Re, Im) of sum_k c_k(t_j) T_k, without the phase e^{-i c t_j}
        psi = torch.view_as_complex(z)
        phases = _phases(center, times[j0:j1])
        for j in range(j1 - j0):
            p = z[:, j, 0].square().addcmul_(z[:, j, 1], z[:, j, 1])  # |psi|^2 as rydberg_cheb: re^2 + im^2
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
            if on_state is not None:  # opt-in hook: psi(t_j) of every realization, only while it is in `acc`
                on_state(j0 + j, (psi[:, j] * phases[j]).view(n_real, dim))
        del acc, z, psi

    if not max_norm2 <= 1 + GROWTH_TOL:
        if bounds == "gershgorin":
            raise FloatingPointError(f"Chebyshev vectors grew (max ||T_k||^2 = {max_norm2:.3e}) with Gershgorin bounds")
        del result, diagonal, scaled_omega, step
        rejected = {"lanczos_rejected": max_norm2, "rejected_matvecs": lanczos_matvecs + matvecs}
        out = evolve(J, omega, delta, times, device, group_size, stats, output, "gershgorin", memory_bytes, on_state)
        if stats is not None:
            stats.update(rejected)
        return out
    if on_host:
        result = result.to(device)
    if stats is not None:
        stats.update(matvecs=matvecs, lanczos_matvecs=lanczos_matvecs, half_width=half_width, center=center,
                     norm_dev=norm_dev, max_norm2=max_norm2, bounds_used=bounds, segments=len(segments), block=block,
                     output_on_host=on_host, budget_bytes=budget, longdouble_eps=float(np.finfo(np.longdouble).eps))
    if norm_dev > NORM_TOL:
        raise FloatingPointError(f"max |sum P - 1| = {norm_dev:.1e} > {NORM_TOL}")
    return result


def moments(probabilities: torch.Tensor, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(<n_i> (..., n), <n_i n_j> (..., n, n)) from probabilities (..., 2^n) of `evolve` (one read of P, by GEMMs over
    slices of basis states; temporary of 2^min(n, 17) x n^2 values of P's dtype, 0.76 GB at n = 27 in float64)."""
    lead = probabilities.shape[:-1]
    P = probabilities.reshape(-1, 2 ** n)
    second = torch.zeros(P.shape[0], n * n, dtype=P.dtype, device=P.device)
    shifts = n - 1 - torch.arange(n, device=P.device)
    chunk = 2 ** min(n, 17)
    for x0 in range(0, 2 ** n, chunk):
        occ = ((torch.arange(x0, x0 + chunk, device=P.device)[:, None] >> shifts) & 1).to(P.dtype)
        second.addmm_(P[:, x0:x0 + chunk], (occ[:, :, None] * occ[:, None, :]).view(chunk, n * n))
    second = second.view(*lead, n, n)
    return torch.diagonal(second, dim1=-2, dim2=-1).clone(), second


def couplings(positions: np.ndarray, c6: float = C6_RYD60) -> np.ndarray:
    """J_ij = C6 / r_ij^6 (rad/us) from positions (n, 2) or (n, 3) in um; zero diagonal."""
    positions = np.asarray(positions, float)
    r = np.linalg.norm(positions[:, None] - positions[None], axis=-1)
    with np.errstate(divide="ignore"):
        return np.where(r > 0, c6 / r ** 6, 0.0)


# ---------------------------------------------------------------- helpers of evolve

def _occupations(n: int, device: torch.device) -> list[torch.Tensor]:
    """n boolean vectors (2^n,): occupation of atom i in each basis state (atom 0 = most significant bit)."""
    index = torch.arange(2 ** n, device=device)
    return [((index >> (n - 1 - i)) & 1).bool() for i in range(n)]


def _fill_diagonal(diagonal: torch.Tensor, J: torch.Tensor, delta: torch.Tensor) -> None:
    """diagonal (R, 2^n) <- E(x) = sum_{i<j} J_ij x_i x_j - sum_i delta_i x_i, in place (same order as rydberg_cheb)."""
    n = delta.shape[1]
    occupied = _occupations(n, diagonal.device)
    diagonal.zero_()
    for i in range(n):
        diagonal -= delta[:, i:i + 1] * occupied[i]
        for j in range(i):
            diagonal += J[:, j, i:i + 1] * (occupied[i] & occupied[j])


def _make_step(scaled_diagonal: torch.Tensor, scaled_omega: torch.Tensor, n: int):
    """step(cur, prev, out): out = 2 H~ cur - prev, or out = H~ cur if prev is None. Batches (R, 2^n), contiguous.
    The returned function reads scaled_diagonal and scaled_omega at call time (they may be rescaled in place)."""
    n_real, dim = scaled_diagonal.shape

    def torch_step(cur, prev, out):  # X_i flips the axis of bit n - 1 - i
        y = scaled_diagonal * cur
        for i in range(n):
            y.addcmul_(scaled_omega[:, i:i + 1], cur.view(n_real, 2 ** i, 2, -1).flip(2).reshape(n_real, dim))
        if prev is None:
            out.copy_(y)
        else:
            torch.sub(y.mul_(2), prev, out=out)

    if not (scaled_diagonal.is_cuda and dim >= TILE and USE_TRITON):
        return torch_step
    if triton is None:
        warnings.warn("triton unavailable: torch step on GPU, ~0.6x the speed of rydberg_cheb", stacklevel=3)
        return torch_step
    failed = False

    def step(cur, prev, out):
        nonlocal failed
        if not failed:
            try:
                _step_kernel[(n_real * (dim // TILE),)](cur, cur if prev is None else prev, scaled_diagonal,
                                                        scaled_omega, out, dim, N_BITS=n, RECURSION=prev is not None,
                                                        BLOCK=TILE, TILES=dim // TILE, num_warps=WARPS)
                return
            except Exception as error:  # first launches compile (C compiler, cache directory): fall back, do not crash
                warnings.warn(f"Triton kernel failed ({error!r}): torch step instead, ~0.6x the speed of rydberg_cheb",
                              stacklevel=2)
                failed = True
        torch_step(cur, prev, out)
    return step


def _lanczos_interval(step, n_real: int, dim: int, device: torch.device) -> tuple[float, float, int]:
    """[min, max] over r of the extreme Ritz values of H~_r (random start, LANCZOS_STEPS steps, no reorthogonalisation)
    and the number of H applications. Ritz values lie inside the spectrum: the caller widens the interval and the series
    guard rejects it if it misses part of the spectrum seen by |0...0>."""
    gen = torch.Generator(device=device).manual_seed(0)
    v = torch.randn(n_real, dim, dtype=torch.float64, device=device, generator=gen)
    v /= v.norm(dim=1, keepdim=True)
    v_prev, u = torch.zeros_like(v), torch.empty_like(v)
    beta = torch.zeros(n_real, 1, dtype=torch.float64, device=device)
    alphas, betas = [], []
    for _ in range(min(LANCZOS_STEPS, dim)):
        step(v, None, u)
        u -= beta * v_prev
        a = torch.linalg.vecdot(u, v).unsqueeze(1)
        u -= a * v
        beta = u.norm(dim=1, keepdim=True)
        alphas.append(a)
        betas.append(beta)
        if float(beta.min()) < 1e-12:  # invariant subspace: the Ritz values are eigenvalues
            break
        v_prev, v = v, u / beta
    a = torch.cat(alphas, 1).cpu().numpy()
    b = torch.cat(betas, 1).cpu().numpy()
    ritz = [eigh_tridiagonal(a[r], b[r, :-1], eigvals_only=True) for r in range(n_real)]
    return min(e[0] for e in ritz), max(e[-1] for e in ritz), len(alphas)


def _bessel(alphas: np.ndarray, rho: float = 1.0) -> np.ndarray:
    """(K, T) J_k(alpha_j), k < K, cut where max_j |J_k(alpha_j)| rho^k < BESSEL_TOL (K >= 2), by Miller's backward
    recurrence in long double normalised by J_0 + 2 sum_k J_2k = 1. rho > 1: Chebyshev series of a non-Hermitian
    matrix whose numerical range lies in the ellipse of foci +/-1 and parameter rho (|T_k| <= rho^k there)."""
    a_max = float(np.max(alphas))
    k_max = int(a_max + 40 * (a_max ** (1 / 3) + 1)) + 10  # beyond the cut (rydberg_cheb's search range)
    while True:
        out = _bessel_upto(alphas, k_max)
        weight = rho ** np.arange(k_max + 1.0)
        if rho == 1 or (np.abs(out[-1]) * weight[-1]).max() < BESSEL_TOL:
            break
        k_max = int(1.3 * k_max) + 10  # rho > 1: the weighted tail decays later
    above = np.nonzero(np.abs(out).max(1) * weight > BESSEL_TOL)[0]
    return out[:max(int(above[-1]) + 1 if above.size else 0, 2)]


def _bessel_upto(alphas: np.ndarray, k_max: int) -> np.ndarray:
    """(k_max + 1, T) J_k(alpha_j) for k <= k_max (Miller's backward recurrence, see _bessel)."""
    a_all = np.asarray(alphas, np.longdouble)
    a_max = float(a_all.max())
    out = np.zeros((k_max + 1, len(a_all)))
    out[0, a_all == 0] = 1.0  # J_k(0) = [k = 0]
    a = a_all[a_all > 0]
    if a.size:
        info = np.finfo(np.longdouble)
        if info.eps > 1e-18:
            warnings.warn("long double is float64 here: Bessel coefficients accurate to ~4e-14 and global phase to "
                          "~1e-13 only (amplitudes ~1e-13)", stacklevel=3)
        big = np.sqrt(info.max)
        start = k_max + 60 + int(20 * (a_max ** (1 / 3) + 1))
        j = np.zeros((start + 2, a.size), np.longdouble)
        j[start] = 1
        for k in range(start, 0, -1):
            j[k - 1] = (2 * k / a) * j[k] - j[k + 1]
            over = np.abs(j[k - 1]) > big
            if over.any():  # rescale before overflow
                j[k - 1:, over] /= big
        j /= j[0] + 2 * j[2::2].sum(0)
        out[:, a_all > 0] = j[:k_max + 1].astype(float)
    return out


def _coefficients(alphas: np.ndarray, rho: float = 1.0) -> np.ndarray:
    """(K, 2T): columns (Re, Im) of c_k(t_j) = (2 - [k = 0]) (-i)^k J_k(alpha_j) (rho: see _bessel)."""
    bessel = _bessel(alphas, rho)
    k = np.arange(len(bessel))
    c = np.where(k == 0, 1.0, 2.0)[:, None] * np.array([1, -1j, -1, 1j])[k % 4][:, None] * bessel
    return np.stack([c.real, c.imag], -1).reshape(len(bessel), 2 * len(alphas))


def _phases(center: float, times: np.ndarray) -> list[complex]:
    """e^{-i c t} with c t evaluated in long double (|c t| reaches ~1e3 rad: float64 would lose ~1e-13)."""
    angle = np.longdouble(center) * np.asarray(times, np.longdouble)
    return [complex(float(np.cos(x)), -float(np.sin(x))) for x in angle]


def _series(step, n_real: int, dim: int, alphas: np.ndarray, block: int,
            device: torch.device, start: torch.Tensor | None = None) -> tuple[torch.Tensor, int, float]:
    """Accumulators (R 2^n, 2T) of sum_k c_k(t_j) T_k (Re, Im interleaved), number of terms K, and the largest
    ||T_k||^2 checked (end of every block and last vector; stops early above 1 + GROWTH_TOL). T_0 = start (real,
    (R, 2^n), norm <= 1 for the guard), default |0...0>."""
    coef = torch.as_tensor(_coefficients(alphas), device=device)
    n_terms = coef.shape[0]
    acc = torch.zeros(n_real * dim, coef.shape[1], dtype=torch.float64, device=device)
    block = min(block, n_terms)  # short series: smaller ring
    ring = torch.empty(block, n_real, dim, dtype=torch.float64, device=device)  # ring[i] = T_{k0 + i}
    if start is None:
        ring[0].zero_()
        ring[0, :, 0] = 1.0  # T_0 = |0...0>
    else:
        ring[0].copy_(start)
    step(ring[0], None, ring[1])  # T_1 = H~ T_0
    k0, done, i, norm2 = 0, 0, 2, 0.0

    def flush(upto):
        nonlocal norm2
        acc.addmm_(ring[done:upto].view(upto - done, -1).T, coef[k0 + done:k0 + upto])
        g = float(torch.linalg.vector_norm(ring[upto - 1], dim=-1).square().max())
        norm2 = max(norm2, g) if g == g else float("nan")

    while k0 + i < n_terms:
        if i == block:
            flush(block)
            if not norm2 <= 1 + GROWTH_TOL:
                break
            ring[0].copy_(ring[block - 2])
            ring[1].copy_(ring[block - 1])
            k0, done, i = k0 + block - 2, 2, 2
            continue
        step(ring[i - 1], ring[i - 2], ring[i])
        i += 1
    else:
        flush(min(i, n_terms - k0))
    del ring
    return acc, n_terms, norm2


def _plan(times: np.ndarray, half_width: float, budget: float, out_vectors: int,
          host_vectors: float) -> tuple[list[tuple[int, int]], int, bool]:
    """(segments [(j0, j1)], ring length, output waits on the host) minimising a memory-traffic model.

    Units: float64 vectors of R 2^n entries (budget: on the device, beyond the diagonal; host_vectors: on the host).
    A segment of the times [j0, j1) is one series from |0...0> to t_{j1 - 1} with 2 (j1 - j0) accumulators. Model of the
    traffic per amplitude: steps x (40 + 32 T_seg / (block - 2)) bytes (step kernel and diagonal; accumulator update
    every block - 2 steps), plus the round trip of the output through PCIe if it waits on the host.
    ponytail: equal-count segments, restarted from |0...0>; restarting from the complex state needs two real series.
    """
    T = len(times)
    best = None
    for n_seg in range(1, T + 1):
        segments = [(int(s[0]), int(s[-1]) + 1) for s in np.array_split(np.arange(T), n_seg)]
        t_seg = max(j1 - j0 for j0, j1 in segments)
        alphas = half_width * times[[j1 - 1 for _, j1 in segments]]
        steps = float(np.sum(alphas + 11 * np.cbrt(alphas) + 2))
        for on_host in (False, True):
            if on_host and out_vectors > host_vectors:
                continue
            work = budget - TEMP_VECTORS - (out_vectors if n_seg > 1 and not on_host else 0)
            block = int(min(BLOCK_MAX, work - 2 * t_seg))
            final = 2 * t_seg + TEMP_VECTORS + (out_vectors if not on_host else 0)  # conversion of the last segment
            if block < BLOCK_MIN or final > budget or out_vectors > budget:
                continue
            cost = steps * (40 + 32 * t_seg / (block - 2)) + (16 * HOST_COPY_FACTOR * out_vectors if on_host else 0)
            if best is None or cost < best[0]:
                best = (cost, segments, block, on_host)
    if best is None:
        raise MemoryError(f"{out_vectors} output vectors and the series do not fit in {budget:.1f} vectors of the "
                          "device budget: split the output times")
    return best[1], best[2], best[3]


def _host_free_bytes() -> float | None:
    """Free host memory: MemAvailable, capped by the memory cgroup (v1 or v2, every ancestor) of this process; None if
    unknown (no /proc/meminfo)."""
    try:
        meminfo = Path("/proc/meminfo").read_text().split()
        free = float(meminfo[meminfo.index("MemAvailable:") + 1]) * 1024
    except (OSError, ValueError):
        return None
    try:
        lines = Path("/proc/self/cgroup").read_text().splitlines()
    except OSError:
        return free
    for line in lines:
        _, controllers, path = line.split(":", 2)
        if controllers == "":
            root, limit_file, usage_file = Path("/sys/fs/cgroup"), "memory.max", "memory.current"
        elif "memory" in controllers.split(","):
            root = Path("/sys/fs/cgroup/memory")
            limit_file, usage_file = "memory.limit_in_bytes", "memory.usage_in_bytes"
        else:
            continue
        parts = [p for p in path.split("/") if p]
        for depth in range(len(parts), -1, -1):
            folder = root.joinpath(*parts[:depth])
            try:
                limit = (folder / limit_file).read_text().strip()
                usage = float((folder / usage_file).read_text())
            except (OSError, ValueError):
                continue
            if limit != "max" and float(limit) < 2 ** 60:
                free = min(free, float(limit) - usage)
    return max(free, 0.0)


# ---------------------------------------------------------------- self-check

def _dense_reference(J: np.ndarray, omega: np.ndarray, delta: np.ndarray, times_us: Sequence[float]) -> np.ndarray:
    """psi(t) by dense matrix exponentials of the full 2^n x 2^n Hamiltonian (independent check, small n only)."""
    from scipy.linalg import expm
    n = len(omega)
    index = np.arange(2 ** n)
    occ = (index[:, None] >> (n - 1 - np.arange(n))[None, :]) & 1
    H = np.diag(np.einsum("ki,ij,kj->k", occ, np.triu(J, 1), occ) - occ @ delta).astype(complex)
    for a in range(n):
        H[index, index ^ (1 << (n - 1 - a))] += omega[a] / 2
    psi0 = np.zeros(2 ** n, complex)
    psi0[0] = 1
    return np.array([expm(-1j * H * t) @ psi0 for t in times_us])


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n, n_real, times = 7, 3, np.array([0.0, 0.2, 0.715, 0.715, 1.536])
    positions = np.column_stack([np.arange(n) * 6.0, rng.normal(0, 0.5, n)])
    J0 = couplings(positions)
    J = np.stack([couplings(positions + rng.normal(0, 0.1, positions.shape)) for _ in range(n_real)])
    omega, delta = rng.uniform(-1, 1, (n_real, n)) * 0.3 * J0.max(), rng.uniform(-5, 5, (n_real, n))
    reference = np.stack([_dense_reference(J[r], omega[r], delta[r], times) for r in range(n_real)], 1)
    for kwargs in ({}, {"bounds": "lanczos"}, {"memory_bytes": 8 * n_real * 2 ** n * 19}):  # 19 vectors: 5 segments
        stats: dict = {}
        P = evolve(J, omega, delta, times, device="cpu", stats=stats, **kwargs).numpy()
        psi = evolve(J, omega, delta, times, device="cpu", output="state", bounds=kwargs.get("bounds", "gershgorin"))
        psi = psi.numpy()
        error, error_state = np.abs(P - np.abs(reference) ** 2).max(), np.abs(psi - reference).max()
        print(kwargs, f"max |P - dense expm| = {error:.1e}; max |psi - dense expm| = {error_state:.1e};",
              {k: stats[k] for k in ("matvecs", "segments", "norm_dev", "bounds_used")})
        assert error < 1e-12 and error_state < 1e-12, (kwargs, error, error_state)
    print("self-check OK")
