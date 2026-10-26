"""noisy_cheb_fast: optimized engine for noisy_cheb (same physics; noisy_cheb stays the validated reference).

Optimizations:
1. Batched trajectories. Trajectories of the same SequenceData (all of them for Lindblad-only noise, e.g. FRESNEL's
   model) evolve as the rows of one (R, 2^n) block: one kernel launch per Chebyshev step for all rows, H read once
   per tile. R is set by the memory budget.
2. Jump detection at segment ends. The no-jump norm is non-increasing, so ||psi(t1)||^2 > r proves that no jump
   happened in [t0, t1]: segments run between output times / piece boundaries with a single accumulator per row.
   Only rows that jump pay for probes: Chebyshev-Lobatto probes over the segment (bracketed recursively when it is
   long), the root of the norm interpolant, one short series to the jump time (as noisy_cheb).
3. Coefficient cache: segments of equal length share their Bessel tables.
4. Optional single precision (single=True, complex64): for statistics-level runs (round-off ~1e-6, far below the
   statistical error); complex128 by default.
Random numbers: one numpy stream per trajectory (SeedSequence(seed).spawn), consumed as noisy_cheb's
_Model.trajectory (threshold; per jump: operator choice, new threshold). In complex128 a trajectory is therefore the
same as noisy_cheb's with that stream, whatever the batch: the self-check verifies it, plus the Lindblad limit.
5. Interaction part reused across drive steps: a time-dependent drive (pulse shapes, local pulses, DMM) changes only
   the detunings, Rabi frequencies and phases between pieces; the interaction diagonal and pair amplitudes are built
   once per coupling matrix, and all pieces share one spectral interval (the union of theirs), so that segments of
   equal length share their Bessel tables too.

Usage: as noisy_cheb (same arguments and results), plus single / max_batch / memory_bytes:
    res = noisy_cheb_fast.simulate(seq, noise_model, n_trajectories=5000, eval_fractions=fractions, device="cuda")
Batching only helps when samples are identical (Lindblad-only noise); for shot-to-shot noise use noisy_cheb_turbo.
Self-check: python -m emu_cheb.solvers.noisy_cheb_fast (CPU, 3 atoms).
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Sequence

import numpy as np
import torch

from . import noisy_cheb as ref
from . import rydberg_cheb_fast as ryd

BLOCK = 32  # ring length (vectors per row)
ROW_VECTORS = BLOCK + ref.PROBES_MAX + 8  # complex vectors per row at the peak (ring, probe accumulators, states)


def simulate(sequence, noise_model=None, n_trajectories: int = 1, eval_fractions: Sequence[float] = (1.0,),
             dt: int = 10, seed: int = 0, device: str | torch.device = "cpu", prefer_device_noise_model: bool = False,
             replace_sequence_data=None, single: bool = False, max_batch: int | None = None,
             memory_bytes: float | None = None, initial_state: str | None = None) -> ref.NoisyResults:
    """As noisy_cheb.simulate, with the fast engine (single, max_batch, memory_bytes: see run_sequence_data)."""
    import logging
    import warnings

    from emu_base import PulserData
    from ..config import ChebConfig  # only its PulserData options are used

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        config = ChebConfig(dt=dt, noise_model=noise_model, n_trajectories=n_trajectories,
                            prefer_device_noise_model=prefer_device_noise_model, log_level=logging.WARNING)
    sequences = PulserData(sequence=sequence, config=config, dt=dt).get_sequences()
    if replace_sequence_data is not None:
        sequences = map(replace_sequence_data, sequences)
    return run_sequence_data(sequences, eval_fractions, seed, device, single, max_batch, memory_bytes, initial_state)


def run_sequence_data(sequences, eval_fractions: Sequence[float] = (1.0,), seed: int = 0,
                      device: str | torch.device = "cpu", single: bool = False, max_batch: int | None = None,
                      memory_bytes: float | None = None, initial_state: str | None = None,
                      on_state=None) -> ref.NoisyResults:
    """One trajectory per SequenceData, batched over consecutive identical samples.

    Args:
        single: complex64 instead of complex128.
        max_batch: cap on the rows of a batch (default: memory budget only).
        memory_bytes: memory the batch may use (default: 40 % of the free GPU memory; 2 GB on CPU).
        initial_state: as noisy_cheb.simulate.
        on_state: optional hook on the states at every output time, as noisy_cheb.run_sequence_data.
    """
    device = torch.device(device)
    dtype = torch.complex64 if single else torch.complex128
    sds = list(sequences)
    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(len(sds))]
    stats = dict(trajectories=0, jumps=0, leaks=0, segments=0, matvecs=0, row_matvecs=0, jump_norm_dev=0.0,
                 max_growth=0.0, batches=0)
    if memory_bytes is None:
        if device.type == "cuda":
            # return the caching allocator's free blocks first (as rydberg_cheb_fast): mem_get_info does not count
            # them, and a run following another one would otherwise get smaller batches (found: turbo 2x slower)
            torch.cuda.empty_cache()
            memory_bytes = 0.4 * torch.cuda.mem_get_info(device)[0]  # headroom for allocator fragmentation
        else:
            memory_bytes = 2e9
    out = None
    i = 0
    while i < len(sds):
        key = (id(sds[i].omega), id(sds[i].delta), id(sds[i].phi), sds[i].bad_atoms)
        model = ref._Model(sds[i], device)
        times_ns = model.output_times(eval_fractions)
        dim = 2 ** int(model.good.sum())
        r_max = max(1, int(memory_bytes / (ROW_VECTORS * dim * (8 if single else 16))))
        if max_batch:
            r_max = min(r_max, max_batch)
        j = i + 1
        while j < len(sds) and j - i < r_max and (id(sds[j].omega), id(sds[j].delta), id(sds[j].phi),
                                                     sds[j].bad_atoms) == key:
            j += 1
        batch = _Batch(model, times_ns, streams[i:j], dtype, stats, initial_state)
        batch.on_state, batch.ids = on_state, list(range(i, j))
        part = batch.run()
        out = part if out is None else {k: (out[k] + part[k] if k in ("P", "corr") else torch.cat([out[k], part[k]]))
                                        for k in out}
        stats["trajectories"] += j - i
        stats["batches"] += 1
        i = j
    n_traj = stats["trajectories"]
    return ref.NoisyResults(times_ns=times_ns, probabilities=(out["P"] / n_traj).cpu(), occupations=out["occ"],
                            correlations=(out["corr"] / n_traj).cpu(), leaked=out["leaked"], stats=stats)


class _Batch:
    """R trajectories of one _Model, advanced segment by segment (all rows reach every segment end together)."""

    on_state = None  # optional hook on the states at the output times (noisy_cheb.run_sequence_data), set by the caller
    ids = None  # index into the caller's samples of every row (for on_state)

    def __init__(self, model, out_times, rngs, dtype, stats, initial_state=None):
        self.model, self.out_times, self.rngs, self.dtype, self.stats = model, out_times, rngs, dtype, stats
        self.initial_state = initial_state
        self.device = model.device
        self.n_total = model.n_total
        self._hams: dict = {}

    def ham(self, p, active):
        """H~ of piece p (noisy_cheb._build's H~, with (center, w) shared by all pieces of this set of active atoms:
        the union of their intervals) and its batched step in self.dtype. The interaction part (diagonal, pair
        flips) is built once per coupling matrix; a piece only adds -sum delta_i n_i, Omega and phi."""
        key = (p, active)
        if key not in self._hams:
            if len(self._hams) >= ref._cache_entries(48 * 2 ** int(sum(active))):
                self._hams.clear()
            self._hams[key] = self._build(p, active, self._common_bounds(active))
        return self._hams[key]

    def _interaction(self, p, active):
        """(bits, interaction diagonal, pairs, hop) of piece p's coupling matrix, cached per (matrix, active)."""
        idx = np.nonzero(active)[0]
        mat = self.model.pieces[p][5][:, idx][:, :, idx]
        key = (mat.tobytes(), active)
        cache = self.__dict__.setdefault("_inter", {})
        if key not in cache:
            if len(cache) >= ref._cache_entries(16 * 2 ** len(idx)):
                cache.clear()
            n = len(idx)
            index = torch.arange(2 ** n, device=self.device)
            bit = lambda i: ref._occupation(index, n, i)  # noqa: E731  (on the fly: n float64 vectors not stored)
            diag = torch.zeros(2 ** n, dtype=torch.float64, device=self.device)
            same, differ = np.zeros((n, n)), np.zeros((n, n))
            for M, O in zip(mat, self.model.ops):  # as noisy_cheb._build
                M = np.triu(M, 1)
                if np.allclose(O, np.diag(np.diag(O))):
                    d0, d1 = np.diag(O).real
                    for i, j in zip(*np.nonzero(M)):
                        diag += M[i, j] * (d0 + (d1 - d0) * bit(i)) * (d0 + (d1 - d0) * bit(j))
                else:
                    same += M * (O[1, 0] ** 2).real
                    differ += M * abs(O[1, 0]) ** 2
            pairs = np.argwhere(np.abs(same) + np.abs(differ) > 0)
            hop = np.array([[same[i, j], differ[i, j]] for i, j in pairs]).reshape(-1, 2)
            cache[key] = (index, diag, pairs, hop)
        return cache[key]

    def _diagonal(self, p, active):
        """Real diagonal of H (interactions - sum_i delta_i n_i) and the Gershgorin radius of piece p."""
        idx = np.nonzero(active)[0]
        _, _, omega, delta, phi, _ = self.model.pieces[p]
        index, diag, pairs, hop = self._interaction(p, active)
        diag = diag.clone()
        for i, a in enumerate(idx):
            diag -= delta[a] * ref._occupation(index, len(idx), i)
        radius = float(np.abs(omega[idx]).sum() / 2 + (np.abs(hop).max(1).sum() if len(hop) else 0.0))
        return diag, radius

    def _common_bounds(self, active):
        """(center, w, rho) covering every piece for these active atoms (computed once)."""
        cache = self.__dict__.setdefault("_bounds", {})
        if active not in cache:
            e_min, e_max = np.inf, -np.inf
            for p in range(len(self.model.pieces)):
                diag, radius = self._diagonal(p, active)
                e_min, e_max = min(e_min, float(diag.min()) - radius), max(e_max, float(diag.max()) + radius)
            n = int(sum(active))
            lam = np.linalg.eigvalsh(self.model.g_loc)
            g_mid, beta_w = 0.5 * (lam[0] + lam[1]), 0.5 * n * (lam[1] - lam[0])
            w = max(0.5 * (e_max - e_min), beta_w, 1e-12)
            beta = beta_w / w
            s = 0.5 * (beta ** 2 + math.sqrt(beta ** 4 + 4 * beta ** 2))
            cache[active] = (complex(0.5 * (e_max + e_min), -n * g_mid), w, math.sqrt(1 + s) + math.sqrt(s))
        return cache[active]

    def _build(self, p, active, bounds):
        """noisy_cheb._build with the given (center, w, rho) and the cached interaction part."""
        from types import SimpleNamespace
        center, w, rho = bounds
        idx = np.nonzero(active)[0]
        n = len(idx)
        _, _, omega, delta, phi, _ = self.model.pieces[p]
        index, _, pairs, hop = self._interaction(p, active)
        diag_re, _ = self._diagonal(p, active)
        lam = np.linalg.eigvalsh(self.model.g_loc)
        a_loc = self.model.g_loc - 0.5 * (lam[0] + lam[1]) * np.eye(2)
        diag = ((diag_re - center.real) / w).to(torch.complex128)
        for i in range(n):
            diag += -1j * (a_loc[0, 0] + (a_loc[1, 1] - a_loc[0, 0]) * ref._occupation(index, n, i)) / w
        up = (omega[idx] / 2 * np.exp(1j * phi[idx]) - 1j * a_loc[1, 0]) / w
        dn = (omega[idx] / 2 * np.exp(-1j * phi[idx]) - 1j * a_loc[0, 1]) / w
        c = dict(dtype=self.dtype, device=self.device)
        real = torch.float32 if self.dtype == torch.complex64 else torch.float64
        step = _make_batched_step(diag.to(**c), torch.as_tensor(up, **c), torch.as_tensor(dn, **c),
                                  torch.as_tensor(hop / w, dtype=real, device=self.device), pairs, n)
        return SimpleNamespace(w=w, center=center, rho=rho, n=n), step

    def run(self) -> dict:
        """Advance all rows from 0 to the end; returns sums over rows of P and <n_i n_j>, per-row <n_i> and leaks."""
        model, R, T = self.model, len(self.rngs), len(self.out_times)
        n_total = self.n_total
        self.thresholds = np.array([g.random() for g in self.rngs])
        self.active = [model.good.copy() for _ in range(R)]
        self.leaked = np.zeros((R, n_total), bool)
        psi = torch.zeros(R, 2 ** int(model.good.sum()), dtype=self.dtype, device=self.device)
        psi[:, model.initial_index(self.initial_state, model.good)] = 1.0
        # rows are grouped by their set of active atoms (leakage shrinks it): all rows of a group share H~ and a
        # state dimension, so they evolve as one block
        groups = {tuple(model.good): (list(range(R)), psi)}  # active mask -> (rows, states), all at time t
        self.res = dict(P=torch.zeros(T, 2 ** n_total, dtype=torch.float64, device=self.device),
                        corr=torch.zeros(T, n_total, n_total, dtype=torch.float64, device=self.device),
                        occ=torch.zeros(R, T, n_total, dtype=torch.float64),
                        leaked=torch.zeros(R, T, n_total, dtype=torch.bool))
        # segment ends common to every row: output times and piece boundaries
        ends = sorted({float(x) for x in self.out_times} | {float(p[1]) for p in model.pieces})
        t, j = 0.0, 0
        j = self._record(groups, t, j)
        for t1 in ends:
            if t1 <= t + 1e-9:
                continue
            p = next(k for k, piece in enumerate(model.pieces) if piece[0] <= t + 1e-9 < piece[1])
            groups = self._segment(groups, p, t, t1)
            t = t1
            j = self._record(groups, t, j)
        return self.res

    def _record(self, groups, t, j):
        while j < len(self.out_times) and abs(self.out_times[j] - t) < 1e-9:
            for active, (rows, psi) in groups.items():
                p = psi.abs().square().to(torch.float64)
                full = torch.zeros(len(rows), 2 ** self.n_total, dtype=torch.float64, device=self.device)
                full[:, ref._embedding(np.array(active), self.device)] = p / p.sum(1, keepdim=True)
                occ, second = ryd.moments(full, self.n_total)
                self.res["P"][j] += full.sum(0)
                self.res["corr"][j] += second.sum(0)
                self.res["occ"][rows, j] = occ.cpu()
                self.res["leaked"][rows, j] = torch.as_tensor(self.leaked[rows])
                if self.on_state is not None:  # opt-in hook: the states themselves, while they are at out_times[j]
                    self.on_state(j, [self.ids[int(r)] for r in rows], psi, np.array(active))
            j += 1
        return j

    def _segment(self, groups, p, t, t1):
        """All rows from t to t1 within piece p; rows that jump continue from their jump time.

        `pending` holds rows still short of t1 with their own current time (all equal to t at first; a row that jumped
        at t + tau restarts from there). Each pass propagates a group to t1 in one batched series: rows still above
        their threshold at t1 are done (the norm is non-increasing, so they did not jump on the way); the others get
        their jump located, applied, and go back to `pending` (possibly in another group after a leak)."""
        done: dict = {}
        pending = {a: (rows, psi, np.full(len(rows), t)) for a, (rows, psi) in groups.items()}
        while pending:
            active, (rows, psi, t_rows) = pending.popitem()
            h, step = self.ham(p, active)
            end = self._propagate(h, step, psi, ((t1 - t_rows) / 1000)[:, None])[:, 0]  # (R_g, dim)
            norms = end.abs().square().sum(1).double().cpu().numpy()
            self.stats["segments"] += 1
            jumped = norms <= self.thresholds[rows]
            if (~jumped).any():
                keep = np.nonzero(~jumped)[0]
                self._add(done, active, [rows[k] for k in keep], end[keep])
            if not jumped.any():
                continue
            idx = np.nonzero(jumped)[0]
            jrows = [rows[k] for k in idx]
            tau, psi_tau = self._locate(h, step, psi[idx], (t1 - t_rows[idx]) / 1000, self.thresholds[jrows])
            for k, row in enumerate(jrows):
                act = self.active[row]
                new = self.model._jump(psi_tau[k].to(torch.complex128), act, self.leaked[row], self.rngs[row],
                                       self.stats).to(self.dtype)
                self.thresholds[row] = self.rngs[row].random()
                self._add(pending, tuple(act), [row], new[None], t_rows[idx[k]] + 1000 * tau[k])
        return {a: (rows, torch.cat(states)) for a, (rows, states) in done.items()}

    @staticmethod
    def _add(store, active, rows, states, t_row=None):
        if t_row is None:  # done: (rows, [state blocks])
            old_rows, old = store.get(active, ([], []))
            store[active] = (old_rows + rows, old + [states])
        else:  # pending: (rows, states, times)
            if active in store:
                old_rows, old, old_t = store[active]
                store[active] = (old_rows + rows, torch.cat([old, states]), np.append(old_t, t_row))
            else:
                store[active] = (rows, states, np.array([t_row]))

    def _locate(self, h, step, psi, window_us, thresholds):
        """Jump times tau (us, from each row's start) and states there, for rows whose norm falls below its threshold
        within its window: probes, recursive bracketing when the window is long, interpolated root."""
        J = len(thresholds)
        offset = np.zeros(J)
        tau = np.zeros(J)
        start = psi
        bracket_start = [None] * J  # state at the probe before the root
        left = np.zeros(J)  # time of that probe from the row start
        todo = np.arange(J)
        window = np.asarray(window_us, float).copy()
        while len(todo):
            # probes over each row's window (one batched series); a window too long for PROBES_MAX probes is only
            # bracketed (the norm is monotone, so the bracket is exact) and refined on the next pass
            m = int(min(ref.PROBES_MAX, ref.PROBES_MIN + math.ceil(h.w * window[todo].max())))
            nodes = window[todo, None] * (1 - np.cos(np.pi * np.arange(m + 1) / m)) / 2
            states = self._propagate(h, step, start, nodes)  # (len(todo), m + 1, dim)
            norms = states.abs().square().sum(2).double().cpu().numpy()
            refine, new_start = [], []
            for a, row in enumerate(todo):
                below = np.nonzero(norms[a] <= thresholds[row])[0]
                k = int(below[0]) if below.size else m  # round-off: the end check said "below"
                k = max(k, 1)
                if h.w * window[row] <= ref.PROBES_MAX - ref.PROBES_MIN:
                    tau[row] = offset[row] + ref._interpolated_root(nodes[a], norms[a], thresholds[row], k - 1)
                    # a copy, not a view: a view would keep the whole (rows, probes, 2^n) block alive into the
                    # next refinement pass (two blocks at once: out of memory at n = 20)
                    bracket_start[row] = states[a, k - 1].clone()
                    left[row] = offset[row] + nodes[a, k - 1]
                else:  # long window: continue from the probe before the crossing, on the bracket
                    refine.append(row)
                    new_start.append(states[a, k - 1])
                    offset[row] += nodes[a, k - 1]
                    window[row] = nodes[a, k] - nodes[a, k - 1]
            todo = np.array(refine, int)
            start = torch.stack(new_start) if new_start else None
            del states
        psi_tau = self._propagate(h, step, torch.stack(bracket_start), (tau - left)[:, None])[:, 0]
        dev = np.abs(psi_tau.abs().square().sum(1).double().cpu().numpy() - thresholds).max()
        self.stats["jump_norm_dev"] = max(self.stats["jump_norm_dev"], float(dev))
        return tau, psi_tau

    def _propagate(self, h, step, psi, offsets_us):
        """States (R, m, dim) at per-row offsets (R, m) (us) from psi (R, dim) under one piece."""
        # noisy_cheb._propagate for R rows: the ring holds (BLOCK, R, dim); with offsets shared by all rows (the
        # common case: every row goes to the same segment end) one coefficient table, cached, else one per row
        R, dim = psi.shape
        alphas = h.w * np.asarray(offsets_us, float)
        shared = bool(np.all(alphas == alphas[:1]))
        if shared:
            coef = torch.as_tensor(_coef_cached(h.rho, tuple(alphas[0])), device=psi.device).to(self.dtype)  # (K, m)
        else:
            c = ryd._coefficients(alphas.ravel(), h.rho)
            coef = torch.as_tensor(c[:, 0::2] + 1j * c[:, 1::2], device=psi.device).to(self.dtype).view(-1, *alphas.shape)
        n_terms, m = coef.shape[0], alphas.shape[1]
        # accumulated in place: (m, R, dim) with shared coefficients (one addmm_ per ring), (R, m, dim) otherwise
        # (one baddbmm_); an einsum would allocate a temporary as large as acc at every flush (2x the planned peak)
        acc = torch.zeros((m, R, dim) if shared else (R, m, dim), dtype=self.dtype, device=psi.device)
        block = min(BLOCK, n_terms)
        ring = torch.empty(block, R, dim, dtype=self.dtype, device=psi.device)
        ring[0] = psi
        norm0 = psi.norm(dim=1)
        if n_terms > 1:
            step(ring[0], None, ring[1])
        k0, done, i = 0, 0, 2
        def flush(upto):
            c, block_ = coef[k0 + done:k0 + upto], ring[done:upto]
            if shared:
                acc.view(m, R * dim).addmm_(c.T, block_.reshape(upto - done, R * dim))
            else:
                acc.baddbmm_(c.permute(1, 2, 0), block_.permute(1, 0, 2))
            k = k0 + upto - 1
            growth = float((ring[upto - 1].norm(dim=1) / (ref.CROUZEIX * h.rho ** k * norm0)).max())
            self.stats["max_growth"] = max(self.stats["max_growth"], growth)
            slack = 1e-3 if self.dtype == torch.complex64 else ref.GROWTH_SLACK
            if not growth <= 1 + slack:
                raise FloatingPointError(f"Chebyshev vectors exceed the ellipse bound ({growth:.3e})")

        while k0 + i < n_terms:
            if i == block:
                flush(block)
                ring[0].copy_(ring[block - 2])
                ring[1].copy_(ring[block - 1])
                k0, done, i = k0 + block - 2, 2, 2
                continue
            step(ring[i - 1], ring[i - 2], ring[i])
            i += 1
        flush(min(i, n_terms - k0))
        self.stats["matvecs"] += n_terms - 1
        self.stats["row_matvecs"] += (n_terms - 1) * R
        s = np.asarray(offsets_us, float)
        phase = np.exp(-1j * np.longdouble(h.center.real) * s.astype(np.longdouble)).astype(complex)
        factor = torch.as_tensor(phase * np.exp(h.center.imag * s), device=psi.device).to(self.dtype)
        states = acc.permute(1, 0, 2) if shared else acc  # (R, m, dim) view, no copy
        return states.mul_(factor[:, :, None])


@lru_cache(maxsize=256)
def _coef_cached(rho: float, alphas: tuple) -> np.ndarray:
    c = ryd._coefficients(np.array(alphas), rho)
    return c[:, 0::2] + 1j * c[:, 1::2]


# ---------------------------------------------------------------- batched step

if ryd.triton is not None:
    import triton.language as tl

    @ryd.triton.jit
    def _batched_kernel(cur_ptr, prev_ptr, diag_ptr, up_ptr, dn_ptr, bits_ptr, hop_ptr, out_ptr, dim, n_pairs,
                        N_BITS: tl.constexpr, RECURSION: tl.constexpr, BLOCK_SIZE: tl.constexpr,
                        TILES: tl.constexpr):
        """noisy_cheb._step_kernel for R rows sharing H~: flat grid of R * TILES programs."""
        pid = tl.program_id(0)
        r = pid // TILES
        offs = (pid % TILES) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        base = 2 * (r.to(tl.int64) * dim)
        cr = tl.load(cur_ptr + base + 2 * offs)
        ci = tl.load(cur_ptr + base + 2 * offs + 1)
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
            sr = tl.load(cur_ptr + base + 2 * src)
            si = tl.load(cur_ptr + base + 2 * src + 1)
            acc_r += kr * sr - ki * si
            acc_i += kr * si + ki * sr
        for p in range(n_pairs):
            bi = tl.load(bits_ptr + 2 * p)
            bj = tl.load(bits_ptr + 2 * p + 1)
            a = tl.where(((offs >> bi) ^ (offs >> bj)) & 1 == 1, tl.load(hop_ptr + 2 * p + 1), tl.load(hop_ptr + 2 * p))
            src = offs ^ ((1 << bi) | (1 << bj))
            acc_r += a * tl.load(cur_ptr + base + 2 * src)
            acc_i += a * tl.load(cur_ptr + base + 2 * src + 1)
        if RECURSION:
            acc_r = 2.0 * acc_r - tl.load(prev_ptr + base + 2 * offs)
            acc_i = 2.0 * acc_i - tl.load(prev_ptr + base + 2 * offs + 1)
        tl.store(out_ptr + base + 2 * offs, acc_r)
        tl.store(out_ptr + base + 2 * offs + 1, acc_i)


def _make_batched_step(diag, up, dn, hop, pairs, n):
    """step(cur, prev, out) on (R, 2^n) complex rows sharing H~ (noisy_cheb._make_step, batched)."""
    dim = 2 ** n
    device = diag.device
    differ = ref._parity_masks(pairs, n, device)  # built on the first torch step only
    up_c, dn_c = up.tolist(), dn.tolist()

    def torch_step(cur, prev, out):
        R = cur.shape[0]
        y = diag * cur
        for i in range(n):
            v, f = y.view(R, 2 ** i, 2, -1), cur.view(R, 2 ** i, 2, -1)
            v[:, :, 0].add_(f[:, :, 1], alpha=dn_c[i])
            v[:, :, 1].add_(f[:, :, 0], alpha=up_c[i])
        for p, (i, j) in enumerate(pairs.tolist()):
            flipped = cur.view(R, 2 ** i, 2, 2 ** (j - i - 1), 2, -1).flip((2, 4)).reshape(R, dim)
            y.addcmul_(torch.where(differ.get()[p], hop[p, 1], hop[p, 0]), flipped)
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
                tiles = dim // ryd.TILE
                _batched_kernel[(cur.shape[0] * tiles,)](
                    torch.view_as_real(cur), torch.view_as_real(cur if prev is None else prev), *real, bits, hop,
                    torch.view_as_real(out), dim, len(pairs), N_BITS=n, RECURSION=prev is not None,
                    BLOCK_SIZE=ryd.TILE, TILES=tiles, num_warps=ryd.WARPS)
                return
            except Exception as error:
                import warnings
                warnings.warn(f"Triton kernel failed ({error!r}): torch step instead (slow)", stacklevel=2)
                failed = True
        torch_step(cur, prev, out)
    return step


# ---------------------------------------------------------------- self-check

if __name__ == "__main__":
    import itertools
    import time
    import warnings

    from emu_base import HamiltonianType
    warnings.filterwarnings("ignore", message="long double is float64")
    rng0 = np.random.default_rng(0)
    relax, deph = np.sqrt(2.0) * np.array([[0, 1], [0, 0]], complex), np.sqrt(1.0 / 2) * np.diag([1.0, -1.0])
    n = 3
    J = np.triu(rng0.uniform(1, 5, (n, n)), 1)
    J = J + J.T
    fractions = np.linspace(0, 1, 9)
    cases = {
        "relaxation + dephasing, piecewise drive": ref._fake_sequence_data(
            n, rng0.uniform(1, 4, (8, n)), rng0.uniform(-2, 2, (8, n)), 0.0, J[None], [relax, deph],
            HamiltonianType.Rydberg),
        "general eff_noise, phase": ref._fake_sequence_data(
            n, 3.0, 1.0, rng0.uniform(0, 2 * np.pi, n), J[None], [1.5 * np.array([[0.3, 0.5 + 0.2j], [0.1j, -0.2]])],
            HamiltonianType.Rydberg),
    }
    for name, sd in cases.items():
        model = ref._Model(sd, torch.device("cpu"))
        out_times = model.output_times(fractions)
        n_traj = 12
        streams = [np.random.default_rng(s) for s in np.random.SeedSequence(5).spawn(n_traj)]
        reference = []
        for g in streams:
            st = dict(trajectories=0, jumps=0, leaks=0, segments=0, matvecs=0, jump_norm_dev=0.0, max_growth=0.0)
            reference.append(model.trajectory(out_times, g, st)[0])
        reference = torch.stack(reference)
        occ_ref = ryd.moments(reference, n)[0]
        for batch in (1, 5, None):  # same streams: identical trajectories whatever the batch
            res = run_sequence_data(itertools.repeat(sd, n_traj), fractions, seed=5, max_batch=batch)
            err = float((res.occupations - occ_ref).abs().max())
            print(f"{name}, batch {batch}: max |d<n>| vs noisy_cheb trajectory by trajectory = {err:.1e} "
                  f"(jumps {res.stats['jumps']}, batches {res.stats['batches']}, jump_norm_dev "
                  f"{res.stats['jump_norm_dev']:.1e})")
            assert err < 1e-8 and res.stats["jumps"] > 0, (name, batch, err)
        # Lindblad limit, single precision included
        exact = ref._dense_lindblad(model, out_times)
        occ_exact = torch.as_tensor(exact) @ ryd.moments(torch.eye(2 ** n, dtype=torch.float64), n)[0]
        for single in (False, True):
            t0 = time.perf_counter()
            res = run_sequence_data(itertools.repeat(sd, 3000), fractions, seed=1, single=single)
            sec = time.perf_counter() - t0
            mean, err = res.occupations.mean(0), res.occupations.std(0) / 3000 ** 0.5
            z = float(((mean - occ_exact).abs() / err.clamp(min=1e-3)).max())
            print(f"    3000 trajectories ({'complex64' if single else 'complex128'}, {sec / 3:.1f} ms/traj) vs dense "
                  f"Lindblad: max |<n_i> error| = {float((mean - occ_exact).abs().max()):.3f}, max z = {z:.1f}")
            assert z < 5, (name, single, z)
    print("self-check OK")
