"""noisy_cheb_turbo: further optimizations on top of noisy_cheb_fast (which stays unchanged); same physics.

1. Lanczos bounds. The Hermitian part of every distinct H~ (each drive piece, each sample of the batch) is bracketed
   by the extreme Ritz values of LANCZOS_STEPS Lanczos steps, widened by LANCZOS_MARGIN of the half-width and never
   wider than Gershgorin (only when the sequence has at most LANCZOS_MAX_PIECES pieces: otherwise the Lanczos work
   would exceed what it saves). Fewer Chebyshev terms (~ w t). A Ritz interval is not a proof: the ellipse check of
   every series (||T_k psi|| <= (1 + sqrt 2) rho^k ||psi||) stays on, and if it fails the whole batch is rerun with
   Gershgorin bounds and the same random streams (stats["lanczos_rejected"]).
2. Shot-to-shot batching. Samples with different H (amplitude, laser waist, detuning, Doppler, register noise, ...)
   share a batch when their time grid, pieces, atoms (bad-atom pattern) and Lindblad operators agree: the kernel
   reads a per-row diagonal, drive and pair amplitudes; (center, w) is the union over rows and pieces, so the Bessel
   tables stay shared. Samples that are the same object (Lindblad-only noise) keep noisy_cheb_fast's shared-H kernel.
3. Small-register kernel (GPU, 2^n <= SMALL_DIM): the whole Chebyshev series of a segment runs in ONE kernel launch,
   one program (thread block) per trajectory, the state staying in L1/L2 with a block barrier between steps, instead
   of one launch per Chebyshev step. vectorized=False (the first validated version): segments with at most SMALL_M
   output times and shared coefficients; jump location keeps the general path.
4. vectorized=True (default): jumps of all rows at once, and jump times by Newton iteration on the norm with its
   exact derivative instead of probe accumulators, so that segments and jump times both go through the small-register
   kernel (per-row coefficient tables); see _TurboBatchV.
Random numbers: one stream per trajectory, as noisy_cheb_fast: trajectories are the same as noisy_cheb_fast's (and
noisy_cheb's) up to round-off, whatever the batching (self-check; the small kernel is checked on GPU).

Usage: as noisy_cheb / noisy_cheb_fast (same arguments and results), plus
    bounds="lanczos" | "gershgorin"   spectral interval (Lanczos: fewer terms at larger n, guarded)
    small_kernel=True                 single-launch series kernel on GPU for 2^n <= SMALL_DIM
    vectorized=True                   batched jumps + Newton jump location (False: the first validated turbo)
    res = noisy_cheb_turbo.simulate(seq, fresnel.default_noise_model, n_trajectories=2000,
                                    eval_fractions=np.linspace(0, 1, 21), device="cuda")
This is the engine to use; noisy_cheb (reference) and noisy_cheb_fast stay as the validated yardsticks.
Self-check: python -m emu_cheb.solvers.noisy_cheb_turbo (CPU, 3 atoms; the GPU kernels are checked by test/emu_cheb/cluster/gpu_turbo_check.py).
"""
from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Sequence

import numpy as np
import torch

from . import noisy_cheb as ref
from . import noisy_cheb_fast as fast
from . import rydberg_cheb_fast as ryd

LANCZOS_STEPS, LANCZOS_MARGIN, LANCZOS_MAX_PIECES = 60, 0.01, 8
SMALL_DIM, SMALL_M = 4096, 4  # small-register kernel: 2^n <= SMALL_DIM, at most SMALL_M output times per segment
SMALL_ELEMS = 16384  # vectorized path: small kernel when (outputs, padded to a power of 2) x 2^n <= SMALL_ELEMS


def simulate(sequence, noise_model=None, n_trajectories: int = 1, eval_fractions: Sequence[float] = (1.0,),
             dt: int = 10, seed: int = 0, device: str | torch.device = "cpu", prefer_device_noise_model: bool = False,
             replace_sequence_data=None, single: bool = False, max_batch: int | None = None,
             memory_bytes: float | None = None, initial_state: str | None = None, bounds: str = "lanczos",
             small_kernel: bool = True, vectorized: bool = True) -> ref.NoisyResults:
    """As noisy_cheb_fast.simulate; bounds: "lanczos" (default) or "gershgorin"; small_kernel, vectorized: see the
    module docstring."""
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
    return run_sequence_data(sequences, eval_fractions, seed, device, single, max_batch, memory_bytes, initial_state,
                             bounds, small_kernel, vectorized)


def run_sequence_data(sequences, eval_fractions: Sequence[float] = (1.0,), seed: int = 0,
                      device: str | torch.device = "cpu", single: bool = False, max_batch: int | None = None,
                      memory_bytes: float | None = None, initial_state: str | None = None, bounds: str = "lanczos",
                      small_kernel: bool = True, vectorized: bool = True, on_state=None) -> ref.NoisyResults:
    """One trajectory per SequenceData; samples with compatible structure are batched even if their H differ.
    on_state: optional hook on the states at every output time, as noisy_cheb.run_sequence_data."""
    if bounds not in ("lanczos", "gershgorin"):
        raise ValueError('bounds must be "lanczos" or "gershgorin"')
    device = torch.device(device)
    dtype = torch.complex64 if single else torch.complex128
    sds = list(sequences)
    seeds = np.random.SeedSequence(seed).spawn(len(sds))
    stats = dict(trajectories=0, jumps=0, leaks=0, segments=0, matvecs=0, row_matvecs=0, jump_norm_dev=0.0,
                 max_growth=0.0, batches=0, lanczos_matvecs=0, lanczos_rejected=0, small_kernel_calls=0)
    if memory_bytes is None:
        if device.type == "cuda":
            # return the caching allocator's free blocks first (as rydberg_cheb_fast): mem_get_info does not count
            # them, and a run following another one would otherwise get smaller batches (found: turbo 2x slower)
            torch.cuda.empty_cache()
            memory_bytes = 0.4 * torch.cuda.mem_get_info(device)[0]  # headroom for allocator fragmentation
        else:
            memory_bytes = 2e9
    models, cache = [], {}
    for sd in sds:
        key = (id(sd.omega), id(sd.delta), id(sd.phi), sd.bad_atoms)
        if key not in cache:
            cache[key] = ref._Model(sd, device)
        models.append(cache[key])
    times_ns = models[0].output_times(eval_fractions)
    n_total, T = models[0].n_total, len(times_ns)
    groups: dict = {}
    for i, m in enumerate(models):
        groups.setdefault(_batch_key(m), []).append(i)
    out = dict(P=torch.zeros(T, 2 ** n_total, dtype=torch.float64, device=device),
               corr=torch.zeros(T, n_total, n_total, dtype=torch.float64, device=device),
               occ=torch.zeros(len(sds), T, n_total, dtype=torch.float64),
               leaked=torch.zeros(len(sds), T, n_total, dtype=torch.bool))
    for idxs in groups.values():
        dim = 2 ** int(models[idxs[0]].good.sum())
        per_row = fast.ROW_VECTORS + (0 if all(models[i] is models[idxs[0]] for i in idxs) else 3)  # + row H
        r_max = max(1, int(memory_bytes / (per_row * dim * (8 if single else 16))))
        if max_batch:
            r_max = min(r_max, max_batch)
        for c0 in range(0, len(idxs), r_max):
            chunk = idxs[c0:c0 + r_max]
            batch = _TurboBatchV if vectorized else _TurboBatch
            b = batch([models[i] for i in chunk], [seeds[i] for i in chunk], times_ns, dtype, stats,
                      initial_state, bounds, small_kernel)
            b.on_state, b.ids = on_state, chunk
            part = b.run()
            out["P"] += part["P"]
            out["corr"] += part["corr"]
            out["occ"][chunk] = part["occ"]
            out["leaked"][chunk] = part["leaked"]
            stats["batches"] += 1
    stats["trajectories"] = len(sds)
    return ref.NoisyResults(times_ns=times_ns, probabilities=(out["P"] / len(sds)).cpu(), occupations=out["occ"],
                            correlations=(out["corr"] / len(sds)).cpu(), leaked=out["leaked"], stats=stats)


def _batch_key(model) -> tuple:
    """Samples with the same key can share a batch (same time grid, pieces, atoms, operators)."""
    return (tuple(model.target_times), tuple((p[0], p[1]) for p in model.pieces), tuple(model.good), model.n_total,
            b"".join(np.ascontiguousarray(L).tobytes() for L in model.lindblad),
            b"".join(np.ascontiguousarray(O).tobytes() for O in model.ops))


class _TurboBatch(fast._Batch):
    """noisy_cheb_fast._Batch with per-row Hamiltonians, Lanczos bounds and the small-register series kernel."""

    def __init__(self, models, seeds, out_times, dtype, stats, initial_state, bounds, small_kernel):
        super().__init__(models[0], out_times, [np.random.default_rng(s) for s in seeds], dtype, stats, initial_state)
        self.models, self.seeds = models, seeds
        self.shared_h = all(m is models[0] for m in models)
        self.bounds_kind, self.small_kernel = bounds, small_kernel
        self._terms: dict = {}
        self._inter_rows: dict = {}

    def run(self) -> dict:
        if self.bounds_kind == "lanczos":
            try:
                return super().run()
            except FloatingPointError:  # the Lanczos interval missed part of the spectrum: redo with Gershgorin
                self.stats["lanczos_rejected"] += 1
                self.bounds_kind = "gershgorin"
                self.rngs = [np.random.default_rng(s) for s in self.seeds]
                self._hams.clear()
                self.__dict__.pop("_bounds", None)
        return super().run()

    # ---------------------------------------------------------------- H~ per row

    def ham(self, p, active):
        key = (p, active)
        if key not in self._hams:
            rows_h = 1 if self.shared_h else len(self.models)
            if len(self._hams) >= ref._cache_entries(32 * rows_h * 2 ** int(sum(active))):
                self._hams.clear()
            self._hams[key] = (self._build_rows(p, active, self._common_bounds(active)), None)
        return self._hams[key]

    def _terms_of(self, p, active):
        """Hermitian terms of piece p for every row (one row if H is shared): diag_re (R', 2^n) float64, omega, phi
        (R', n), hop (R', P, 2), pairs (P, 2), bits (n, 2^n)."""
        key = (p, active)
        if key in self._terms:
            return self._terms[key]
        models = [self.model] if self.shared_h else self.models
        idx = np.nonzero(active)[0]
        n = len(idx)
        if len(self._terms) >= ref._cache_entries(8 * len(models) * 2 ** n):
            self._terms.clear()
        omega = np.stack([m.pieces[p][2][idx] for m in models])
        delta = np.stack([m.pieces[p][3][idx] for m in models])
        phi = np.stack([m.pieces[p][4][idx] for m in models])
        mats = np.stack([m.pieces[p][5][:, idx][:, :, idx] for m in models])  # (R', K, n, n)
        bits = self._bits(n)
        ikey = (mats.tobytes(), active)
        if ikey not in self._inter_rows:
            if len(self._inter_rows) >= ref._cache_entries(8 * len(models) * 2 ** n):
                self._inter_rows.clear()
            diag = torch.zeros(len(models), 2 ** n, dtype=torch.float64, device=self.device)
            same, differ = np.zeros((len(models), n, n)), np.zeros((len(models), n, n))
            for k, O in enumerate(self.model.ops):  # as noisy_cheb._build, per row
                M = np.triu(mats[:, k], 1)
                if np.allclose(O, np.diag(np.diag(O))):
                    d0, d1 = np.diag(O).real
                    for i, j in zip(*np.nonzero(np.abs(M).max(0))):
                        f_ij = (d0 + (d1 - d0) * bits.row(i)) * (d0 + (d1 - d0) * bits.row(j))
                        diag += torch.as_tensor(M[:, i, j], device=self.device)[:, None] * f_ij[None]
                else:
                    same += M * (O[1, 0] ** 2).real
                    differ += M * abs(O[1, 0]) ** 2
            pairs = np.argwhere((np.abs(same) + np.abs(differ)).max(0) > 0)
            hop = np.stack([same[:, pairs[:, 0], pairs[:, 1]], differ[:, pairs[:, 0], pairs[:, 1]]], -1)
            self._inter_rows[ikey] = (diag, pairs, hop.reshape(len(models), -1, 2))
        diag_int, pairs, hop = self._inter_rows[ikey]
        # per piece only the detunings change the diagonal: one (R', n) x (n, 2^n) product for all rows
        diag = diag_int - bits.weighted(torch.as_tensor(delta, device=self.device))
        radius = np.abs(omega).sum(1) / 2 + (np.abs(hop).max(2).sum(1) if hop.shape[1] else 0.0)
        self._terms[key] = SimpleNamespace(diag=diag, omega=omega, phi=phi, hop=hop, pairs=pairs, bits=bits, n=n,
                                           radius=radius)
        return self._terms[key]

    def _bits(self, n):
        """Occupations of the n atoms: rows on the fly (never n float64 vectors kept per piece), and the products
        sum_i c[r, i] n_i for all rows: one (R', n) x (n, 2^n) matrix product while the matrix is small (2^n <= 2^16),
        otherwise a loop over the atoms."""
        cache = self.__dict__.setdefault("_bits_cache", {})
        if n not in cache:
            index = torch.arange(2 ** n, device=self.device)
            dense = (torch.stack([ref._occupation(index, n, i) for i in range(n)]) if 0 < n <= 16 else None)

            class Bits:
                def row(self_, i):
                    return dense[i] if dense is not None else ref._occupation(index, n, i)

                def weighted(self_, c):  # (R', n) -> (R', 2^n)
                    if n == 0:
                        return torch.zeros(c.shape[0], 1, dtype=torch.float64, device=c.device)
                    if dense is not None:
                        return c @ dense
                    out = torch.zeros(c.shape[0], 2 ** n, dtype=torch.float64, device=c.device)
                    for i in range(n):
                        out += c[:, i:i + 1] * ref._occupation(index, n, i)
                    return out
            cache[n] = Bits()
        return cache[n]

    def _common_bounds(self, active):
        cache = self.__dict__.setdefault("_bounds", {})
        if active in cache:
            return cache[active]
        pieces = range(len(self.model.pieces))
        e_min, e_max = np.inf, -np.inf
        for p in pieces:
            t = self._terms_of(p, active)
            e_min = min(e_min, float((t.diag.min(1).values.cpu().numpy() - t.radius).min()))
            e_max = max(e_max, float((t.diag.max(1).values.cpu().numpy() + t.radius).max()))
        if self.bounds_kind == "lanczos" and len(pieces) <= LANCZOS_MAX_PIECES:
            lo, hi = np.inf, -np.inf
            for p in pieces:
                a, b = self._lanczos(self._terms_of(p, active))
                lo, hi = min(lo, a), max(hi, b)
            margin = LANCZOS_MARGIN * 0.5 * (hi - lo)
            e_min, e_max = max(lo - margin, e_min), min(hi + margin, e_max)  # never wider than Gershgorin
        n = int(sum(active))
        lam = np.linalg.eigvalsh(self.model.g_loc)
        g_mid, beta_w = 0.5 * (lam[0] + lam[1]), 0.5 * n * (lam[1] - lam[0])
        w = max(0.5 * (e_max - e_min), beta_w, 1e-12)
        beta = beta_w / w
        s = 0.5 * (beta ** 2 + math.sqrt(beta ** 4 + 4 * beta ** 2))
        cache[active] = (complex(0.5 * (e_max + e_min), -n * g_mid), w, math.sqrt(1 + s) + math.sqrt(s))
        return cache[active]

    def _lanczos(self, t) -> tuple[float, float]:
        """min / max over rows of the extreme Ritz values of the Hermitian part (unscaled, rad/us)."""
        from scipy.linalg import eigh_tridiagonal
        R, dim = t.diag.shape
        up = torch.as_tensor(t.omega / 2 * np.exp(1j * t.phi), device=self.device)
        step = _rowwise_step(t.diag.to(torch.complex128), up, up.conj().resolve_conj(), torch.as_tensor(t.hop, device=self.device),
                             t.pairs, t.n)
        gen = torch.Generator(device=self.device).manual_seed(0)
        v = torch.complex(torch.randn(R, dim, dtype=torch.float64, device=self.device, generator=gen),
                          torch.randn(R, dim, dtype=torch.float64, device=self.device, generator=gen))
        v /= v.norm(dim=1, keepdim=True)
        v_prev, u = torch.zeros_like(v), torch.empty_like(v)
        beta = torch.zeros(R, 1, dtype=torch.float64, device=self.device)
        alphas, betas = [], []
        for _ in range(min(LANCZOS_STEPS, dim)):
            step(v, None, u)
            u -= beta * v_prev
            a = (v.conj() * u).sum(1, keepdim=True).real
            u -= a * v
            beta = u.norm(dim=1, keepdim=True)
            alphas.append(a)
            betas.append(beta)
            if float(beta.min()) < 1e-12:
                break
            v_prev, v = v, u / beta
        self.stats["lanczos_matvecs"] += len(alphas) * R
        a = torch.cat(alphas, 1).cpu().numpy()
        b = torch.cat(betas, 1).cpu().numpy()
        ritz = [eigh_tridiagonal(a[r], b[r, :-1], eigvals_only=True) for r in range(R)]
        return min(e[0] for e in ritz), max(e[-1] for e in ritz)

    def _build_rows(self, p, active, bounds):
        center, w, rho = bounds
        t = self._terms_of(p, active)
        lam = np.linalg.eigvalsh(self.model.g_loc)
        a_loc = self.model.g_loc - 0.5 * (lam[0] + lam[1]) * np.eye(2)
        diag = ((t.diag - center.real) / w).to(torch.complex128)
        for i in range(t.n):
            diag += -1j * (a_loc[0, 0] + (a_loc[1, 1] - a_loc[0, 0]) * t.bits.row(i)) / w
        up = (t.omega / 2 * np.exp(1j * t.phi) - 1j * a_loc[1, 0]) / w
        dn = (t.omega / 2 * np.exp(-1j * t.phi) - 1j * a_loc[0, 1]) / w
        c = dict(dtype=self.dtype, device=self.device)
        real = torch.float32 if self.dtype == torch.complex64 else torch.float64
        h = SimpleNamespace(w=w, center=center, rho=rho, n=t.n, pairs=t.pairs, shared=self.shared_h,
                            diag=diag.to(**c), up=torch.as_tensor(up, **c), dn=torch.as_tensor(dn, **c),
                            hop=torch.as_tensor(t.hop / w, dtype=real, device=self.device), step=None)
        if self.shared_h:
            h.step = fast._make_batched_step(h.diag[0], h.up[0], h.dn[0], h.hop[0], t.pairs, t.n)
        return h

    def _step(self, h, rows):
        if h.shared:
            return h.step
        sel = torch.as_tensor(rows, device=self.device)
        return _rowwise_step(h.diag[sel], h.up[sel], h.dn[sel], h.hop[sel], h.pairs, h.n)

    # ---------------------------------------------------------------- segments (noisy_cheb_fast with rows)

    def _segment(self, groups, p, t, t1):
        done: dict = {}
        pending = {a: (rows, psi, np.full(len(rows), t)) for a, (rows, psi) in groups.items()}
        while pending:
            active, (rows, psi, t_rows) = pending.popitem()
            h, _ = self.ham(p, active)
            end = self._propagate_rows(h, rows, psi, ((t1 - t_rows) / 1000)[:, None])[:, 0]
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
            tau, psi_tau = self._locate_rows(h, jrows, psi[idx], (t1 - t_rows[idx]) / 1000, self.thresholds[jrows])
            for k, row in enumerate(jrows):
                act = self.active[row]
                new = self.models[row]._jump(psi_tau[k].to(torch.complex128), act, self.leaked[row], self.rngs[row],
                                             self.stats).to(self.dtype)
                self.thresholds[row] = self.rngs[row].random()
                self._add(pending, tuple(act), [row], new[None], t_rows[idx[k]] + 1000 * tau[k])
        return {a: (rows, torch.cat(states)) for a, (rows, states) in done.items()}

    def _locate_rows(self, h, jrows, psi, window_us, thresholds):
        J = len(thresholds)
        offset, tau, left = np.zeros(J), np.zeros(J), np.zeros(J)
        start = psi
        bracket_start = [None] * J
        todo = np.arange(J)
        window = np.asarray(window_us, float).copy()
        while len(todo):
            m = int(min(ref.PROBES_MAX, ref.PROBES_MIN + math.ceil(h.w * window[todo].max())))
            nodes = window[todo, None] * (1 - np.cos(np.pi * np.arange(m + 1) / m)) / 2
            states = self._propagate_rows(h, [jrows[a] for a in todo], start, nodes)
            norms = states.abs().square().sum(2).double().cpu().numpy()
            refine, new_start = [], []
            for a, row in enumerate(todo):
                below = np.nonzero(norms[a] <= thresholds[row])[0]
                k = max(int(below[0]) if below.size else m, 1)
                if h.w * window[row] <= ref.PROBES_MAX - ref.PROBES_MIN:
                    tau[row] = offset[row] + ref._interpolated_root(nodes[a], norms[a], thresholds[row], k - 1)
                    # a copy, not a view: a view would keep the whole (rows, probes, 2^n) block alive into the
                    # next refinement pass (two blocks at once: out of memory at n = 20)
                    bracket_start[row] = states[a, k - 1].clone()
                    left[row] = offset[row] + nodes[a, k - 1]
                else:
                    refine.append(row)
                    new_start.append(states[a, k - 1])
                    offset[row] += nodes[a, k - 1]
                    window[row] = nodes[a, k] - nodes[a, k - 1]
            todo = np.array(refine, int)
            start = torch.stack(new_start) if new_start else None
            del states
        psi_tau = self._propagate_rows(h, jrows, torch.stack(bracket_start), (tau - left)[:, None])[:, 0]
        dev = np.abs(psi_tau.abs().square().sum(1).double().cpu().numpy() - thresholds).max()
        self.stats["jump_norm_dev"] = max(self.stats["jump_norm_dev"], float(dev))
        return tau, psi_tau

    def _propagate_rows(self, h, rows, psi, offsets_us):
        offsets_us = np.asarray(offsets_us, float)
        dim = psi.shape[1]
        if (self.small_kernel and psi.is_cuda and ryd.triton is not None and ryd.USE_TRITON and 32 <= dim <= SMALL_DIM
                and offsets_us.shape[1] <= SMALL_M and bool(np.all(offsets_us == offsets_us[:1]))):
            return self._propagate_small(h, rows, psi, offsets_us)
        return fast._Batch._propagate(self, h, self._step(h, rows), psi, offsets_us)

    def _propagate_small(self, h, rows, psi, offsets_us):
        """The whole series in one launch (one program per row): see _series_kernel. Offsets shared by the rows use
        one coefficient table, otherwise one table per row."""
        R, dim = psi.shape
        m = offsets_us.shape[1]
        mp = max(SMALL_M, 1 << (m - 1).bit_length())
        coef_row = not bool(np.all(offsets_us == offsets_us[:1]))
        if coef_row:
            c = ryd._coefficients((h.w * offsets_us).ravel(), h.rho)
            coef = (c[:, 0::2] + 1j * c[:, 1::2]).reshape(-1, R, m).transpose(1, 0, 2)  # (R, K, m)
            table = np.zeros((R, coef.shape[1], mp), complex)
            table[:, :, :m] = coef
        else:
            coef = fast._coef_cached(h.rho, tuple(h.w * offsets_us[0]))  # (K, m)
            table = np.zeros((coef.shape[0], mp), complex)
            table[:, :m] = coef
        n_terms = table.shape[-2]
        table = torch.as_tensor(table, device=psi.device).to(self.dtype)
        if h.shared:
            diag, up, dn, hop = h.diag[0], h.up[0], h.dn[0], h.hop[0]
        else:
            sel = torch.as_tensor(rows, device=self.device)
            diag, up, dn, hop = h.diag[sel], h.up[sel], h.dn[sel], h.hop[sel]
        bits = torch.as_tensor(h.n - 1 - h.pairs, dtype=torch.int32, device=psi.device).reshape(-1, 2).contiguous()
        out = torch.empty(R, mp, dim, dtype=self.dtype, device=psi.device)
        buf = torch.empty(R, 2, dim, dtype=self.dtype, device=psi.device)
        last = torch.empty(R, dtype=torch.float32 if self.dtype == torch.complex64 else torch.float64,
                           device=psi.device)
        psi = psi.contiguous()
        real = [torch.view_as_real(x.contiguous()) for x in (psi, buf, table, diag, up, dn)]
        _series_kernel[(R,)](*real, bits, hop.reshape(-1, 2).contiguous() if hop.numel() else hop.contiguous(),
                             torch.view_as_real(out), last, n_terms, len(h.pairs), N_BITS=h.n, DIM=dim, MP=mp,
                             ROW_H=not h.shared, COEF_ROW=coef_row, num_warps=min(16, max(1, mp * dim // 1024)),
                             num_stages=1)  # no software pipelining: it would prefetch T_k before it is stored
        self.stats["small_kernel_calls"] += 1
        self.stats["matvecs"] += n_terms - 1
        self.stats["row_matvecs"] += (n_terms - 1) * R
        growth = float((last.sqrt() / (ref.CROUZEIX * h.rho ** (n_terms - 1) * psi.norm(dim=1))).max())
        self.stats["max_growth"] = max(self.stats["max_growth"], growth)
        if not growth <= 1 + (1e-3 if self.dtype == torch.complex64 else ref.GROWTH_SLACK):
            raise FloatingPointError(f"Chebyshev vectors exceed the ellipse bound ({growth:.3e})")
        s = offsets_us
        phase = np.exp(-1j * np.longdouble(h.center.real) * s.astype(np.longdouble)).astype(complex)
        factor = torch.as_tensor(phase * np.exp(h.center.imag * s), device=psi.device).to(self.dtype)
        return out[:, :m].mul_(factor[:, :, None])


class _TurboBatchV(_TurboBatch):
    """_TurboBatch with vectorized jumps (vectorized=True): every jumping row of a group is handled at once.

    Jump time: safeguarded Newton on N(t) = ||psi(t)||^2 - r with the exact derivative dN/dt = -2 <psi|G|psi>
    (G = 1/2 sum L^dag L), bracketed by [segment start, segment end] (N is non-increasing), first guess from an
    exponential interpolation of N; every iteration is one batched single-output series from the segment start, so
    the small-register kernel takes it (per-row coefficient tables). No probe accumulators.
    Jump: reduced density matrices and weights ||L_k(i) psi||^2 of all rows in a few batched contractions; each row
    draws from its own stream in the same order as noisy_cheb (operator choice, then the new threshold)."""

    NEWTON_MAX = 60

    def _small_ok(self, psi, m=1):
        mp = max(SMALL_M, 1 << (m - 1).bit_length())
        return bool(self.small_kernel and psi.is_cuda and ryd.triton is not None and ryd.USE_TRITON
                    and 32 <= psi.shape[1] <= SMALL_DIM and mp * psi.shape[1] <= SMALL_ELEMS)

    def _propagate_rows(self, h, rows, psi, offsets_us):
        offsets_us = np.asarray(offsets_us, float)
        if self._small_ok(psi, offsets_us.shape[1]):
            return self._propagate_small(h, rows, psi, offsets_us)
        return fast._Batch._propagate(self, h, self._step(h, rows), psi, offsets_us)

    @staticmethod
    def _reduced_rows(psi, off_diagonal=True):
        """Reduced 2 x 2 density matrices of every qubit for every row (unnormalised psi): (rho00, rho11, rho01), each
        (R, n), rho01 = sum_x psi_x conj(psi_{x with bit i flipped}) over bit i = 0 (None if not requested).
        Populations come from |psi|^2 reduced over strided views (no copy of the state); a per-qubit einsum would copy
        and permute the whole state n times (3x slower than the reference engine's jump at n = 18)."""
        R, dim = psi.shape
        n = dim.bit_length() - 1
        p = psi.abs().square().double()
        total = p.sum(1)
        rho11 = torch.stack([p.view(R, 2 ** i, 2, -1)[:, :, 1].sum((1, 2)) for i in range(n)], 1)
        rho01 = None
        if off_diagonal:
            rho01 = torch.stack([(psi.view(R, 2 ** i, 2, -1)[:, :, 0] * psi.view(R, 2 ** i, 2, -1)[:, :, 1].conj())
                                 .sum((1, 2)).to(torch.complex128) for i in range(n)], 1)
        return total[:, None] - rho11, rho11, rho01

    @staticmethod
    def _trace_with(op, rho00, rho11, rho01):
        """Re Tr(op rho_i) for a 2 x 2 Hermitian op: (R, n)."""
        out = op[0, 0].real * rho00 + op[1, 1].real * rho11
        if rho01 is not None and abs(op[0, 1]) > 0:  # Tr(op rho) off-diagonal part: 2 Re(op_01 rho_10) = 2 Re(op_01 conj(rho_01))
            out = out + 2 * (op[0, 1] * rho01.conj()).real
        return out

    def _expect_g(self, psi):
        """<psi|G|psi> per row (G = sum_i g_loc(i)), real."""
        g = self.model.g_loc
        rho00, rho11, rho01 = self._reduced_rows(psi, off_diagonal=abs(g[0, 1]) > 0)
        return self._trace_with(g, rho00, rho11, rho01).sum(1)

    def _segment(self, groups, p, t, t1):
        done: dict = {}
        pending = {a: (rows, psi, np.full(len(rows), t)) for a, (rows, psi) in groups.items()}
        while pending:
            active, (rows, psi, t_rows) = pending.popitem()
            h, _ = self.ham(p, active)
            end = self._propagate_rows(h, rows, psi, ((t1 - t_rows) / 1000)[:, None])[:, 0]
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
            if self._small_ok(psi):  # Newton: more terms, but each iteration is one launch of the small kernel
                tau, psi_tau = self._locate_newton(h, jrows, psi[idx], (t1 - t_rows[idx]) / 1000,
                                                   self.thresholds[jrows], norms[idx])
            else:  # large registers / CPU: the probes of noisy_cheb_fast need fewer terms
                tau, psi_tau = self._locate_rows(h, jrows, psi[idx], (t1 - t_rows[idx]) / 1000, self.thresholds[jrows])
            for new_active, (sub, states) in self._jump_batch(psi_tau, active, jrows).items():
                for k, state in zip(sub, states):
                    self._add(pending, new_active, [jrows[k]], state[None], t_rows[idx[k]] + 1000 * tau[k])
        return {a: (rows, torch.cat(states)) for a, (rows, states) in done.items()}

    def _locate_newton(self, h, jrows, psi0, window_us, thresholds, n_end):
        J = len(jrows)
        window = np.asarray(window_us, float)
        lo, hi = np.zeros(J), window.copy()
        n0 = psi0.abs().square().sum(1).double().cpu().numpy()
        with np.errstate(divide="ignore", invalid="ignore"):  # exponential interpolation between the two ends
            t = window * np.log(n0 / thresholds) / np.log(n0 / np.maximum(n_end, 1e-300))
        t = np.where(np.isfinite(t) & (t > 0) & (t < window), t, 0.5 * window)
        tau, out = np.zeros(J), [None] * J
        todo = np.arange(J)
        # each pass: one batched series from the segment start to every unconverged row's current guess t, then
        # Newton t <- t - (N - r) / N' (N' = -2 <G>), falling back to bisection when the step leaves the bracket
        for _ in range(self.NEWTON_MAX):
            states = self._propagate_rows(h, [jrows[a] for a in todo], psi0[todo], t[todo][:, None])[:, 0]
            N = states.abs().square().sum(1).double().cpu().numpy()
            dN = -2 * self._expect_g(states.to(torch.complex128)).cpu().numpy()
            r = thresholds[todo]
            above = N > r
            lo[todo] = np.where(above, t[todo], lo[todo])
            hi[todo] = np.where(above, hi[todo], t[todo])
            # tolerances at the working precision: in complex64 the norm is only good to ~1e-7, and float64 criteria
            # would never be met (every jump ran NEWTON_MAX iterations: complex64 3x slower than complex128 at n = 10)
            tol, btol = (1e-14, 4e-16) if self.dtype == torch.complex128 else (3e-6, 1e-6)
            converged = (np.abs(N - r) <= tol * np.maximum(r, 1e-300)) | (hi[todo] - lo[todo] <= btol * window[todo])
            with np.errstate(divide="ignore", invalid="ignore"):
                step = t[todo] - (N - r) / dN
            bad = ~np.isfinite(step) | (step <= lo[todo]) | (step >= hi[todo])
            step = np.where(bad, 0.5 * (lo[todo] + hi[todo]), step)
            for a, row in enumerate(todo):
                if converged[a]:
                    tau[row], out[row] = t[row], states[a].clone()  # copy: a view would keep this pass's block
            t[todo] = np.where(converged, t[todo], step)
            todo = todo[~converged]
            if not len(todo):
                break
        else:  # bracket below round-off without |N - r| <= 1e-14 r: accept the last iterate
            states = self._propagate_rows(h, [jrows[a] for a in todo], psi0[todo], t[todo][:, None])[:, 0]
            for a, row in enumerate(todo):
                tau[row], out[row] = t[row], states[a].clone()
        psi_tau = torch.stack(out)
        dev = np.abs(psi_tau.abs().square().sum(1).double().cpu().numpy() - thresholds).max()
        self.stats["jump_norm_dev"] = max(self.stats["jump_norm_dev"], float(dev))
        return tau, psi_tau

    def _jump_batch(self, psi_tau, active, jrows):
        """Apply one jump to every row: {new active mask: (indices into jrows, normalised states)}."""
        model = self.model
        psi = psi_tau.to(torch.complex128)
        J, dim = psi.shape
        n = dim.bit_length() - 1
        n_ops = len(model.jump_ldl)
        rho00, rho11, rho01 = self._reduced_rows(psi, off_diagonal=any(abs(L[0, 1]) > 0 for L in model.jump_ldl))
        weights = torch.stack([self._trace_with(L, rho00, rho11, rho01) for L in model.jump_ldl], 2)  # (J, n, K)
        weights = weights.clamp(min=0).reshape(J, n * n_ops).cpu().numpy()
        choice = np.empty(J, int)
        for a, row in enumerate(jrows):  # each row's stream, noisy_cheb's arithmetic
            w = weights[a]
            choice[a] = int(np.searchsorted(np.cumsum(w) / w.sum(), self.rngs[row].random(), side="right")
                            .clip(0, n * n_ops - 1))
        new = torch.empty_like(psi)
        result: dict = {}
        leaks = []
        for c in np.unique(choice):
            sub = np.nonzero(choice == c)[0]
            i, k = divmod(int(c), n_ops)
            if k < len(model.inside):  # L on qubit i: new_b = L_b0 v_0 + L_b1 v_1 on the two halves (no permute)
                L = model.inside[k][:2, :2]
                v = psi[sub].view(len(sub), 2 ** i, 2, -1)
                out = new[sub].view(len(sub), 2 ** i, 2, -1)
                for b in (0, 1):
                    out[:, :, b] = complex(L[b, 0]) * v[:, :, 0] + complex(L[b, 1]) * v[:, :, 1]
                new[sub] = out.reshape(len(sub), dim)
            else:
                leaks.extend((a, i, k - len(model.inside)) for a in sub)
        keep = np.setdiff1d(np.arange(J), [a for a, _, _ in leaks])
        if len(keep):
            states = new[keep]
            states = states / states.norm(dim=1, keepdim=True)
            result[active] = (list(keep), list(states.to(self.dtype)))
        for a, i, kl in leaks:  # leakage into x: the atom leaves the simulation (rare: one row at a time)
            row = jrows[a]
            L = torch.as_tensor(model.leak[kl], dtype=torch.complex128, device=psi.device)
            v = psi[a].view(2 ** i, 2, -1)
            state = (L[2, 0] * v[:, 0] + L[2, 1] * v[:, 1]).reshape(-1)
            act = self.active[row]
            atom = np.nonzero(act)[0][i]
            act[atom], self.leaked[row, atom] = False, True
            self.stats["leaks"] += 1
            rows_, states_ = result.get(tuple(act), ([], []))
            result[tuple(act)] = (rows_ + [a], states_ + [(state / state.norm()).to(self.dtype)])
        for a, row in enumerate(jrows):  # new thresholds, after the operator choice (noisy_cheb's order)
            self.thresholds[row] = self.rngs[row].random()
        self.stats["jumps"] += J
        return result


# ---------------------------------------------------------------- kernels

def _rowwise_step(diag, up, dn, hop, pairs, n):
    """step(cur, prev, out) on (R, 2^n) rows, row r with its own H~: diag (R, 2^n), up / dn (R, n), hop (R, P, 2)."""
    R, dim = diag.shape
    device = diag.device
    differ = ref._parity_masks(pairs, n, device)  # built on the first torch step only
    dn_c, up_c = dn[:, :, None, None], up[:, :, None, None]

    def torch_step(cur, prev, out):
        y = diag * cur
        for i in range(n):
            v, f = y.view(R, 2 ** i, 2, -1), cur.view(R, 2 ** i, 2, -1)
            v[:, :, 0].addcmul_(dn_c[:, i], f[:, :, 1])
            v[:, :, 1].addcmul_(up_c[:, i], f[:, :, 0])
        for p, (i, j) in enumerate(pairs.tolist()):
            flipped = cur.view(R, 2 ** i, 2, 2 ** (j - i - 1), 2, -1).flip((2, 4)).reshape(R, dim)
            y.addcmul_(torch.where(differ.get()[p], hop[:, p, 1:], hop[:, p, :1]), flipped)
        if prev is None:
            out.copy_(y)
        else:
            torch.sub(y.mul_(2), prev, out=out)

    if not (diag.is_cuda and dim >= ryd.TILE and ryd.USE_TRITON):
        return torch_step
    bits = torch.as_tensor(n - 1 - pairs, dtype=torch.int32, device=device).reshape(-1, 2).contiguous()
    hop_c = hop.contiguous()
    real = [torch.view_as_real(x.resolve_conj().contiguous()) for x in (diag, up, dn)]
    failed = False

    def step(cur, prev, out):
        nonlocal failed
        if not failed:
            try:
                tiles = dim // ryd.TILE
                _rowwise_kernel[(R * tiles,)](
                    torch.view_as_real(cur), torch.view_as_real(cur if prev is None else prev), *real, bits, hop_c,
                    torch.view_as_real(out), dim, len(pairs), N_BITS=n, RECURSION=prev is not None,
                    BLOCK_SIZE=ryd.TILE, TILES=tiles, num_warps=ryd.WARPS)
                return
            except Exception as error:
                import warnings
                warnings.warn(f"Triton kernel failed ({error!r}): torch step instead (slow)", stacklevel=2)
                failed = True
        torch_step(cur, prev, out)
    return step


if ryd.triton is not None:
    import triton.language as tl

    @ryd.triton.jit
    def _rowwise_kernel(cur_ptr, prev_ptr, diag_ptr, up_ptr, dn_ptr, bits_ptr, hop_ptr, out_ptr, dim, n_pairs,
                        N_BITS: tl.constexpr, RECURSION: tl.constexpr, BLOCK_SIZE: tl.constexpr,
                        TILES: tl.constexpr):
        """noisy_cheb_fast._batched_kernel with a per-row H~ (diag, up, dn, hop indexed by the row)."""
        pid = tl.program_id(0)
        r = pid // TILES
        offs = (pid % TILES) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        base = 2 * (r.to(tl.int64) * dim)
        cr = tl.load(cur_ptr + base + 2 * offs)
        ci = tl.load(cur_ptr + base + 2 * offs + 1)
        dr = tl.load(diag_ptr + base + 2 * offs)
        di = tl.load(diag_ptr + base + 2 * offs + 1)
        acc_r = dr * cr - di * ci
        acc_i = dr * ci + di * cr
        for i in tl.static_range(N_BITS):
            shift = N_BITS - 1 - i
            src = offs ^ (1 << shift)
            target_one = ((offs >> shift) & 1) == 1
            kr = tl.where(target_one, tl.load(up_ptr + 2 * (r * N_BITS + i)), tl.load(dn_ptr + 2 * (r * N_BITS + i)))
            ki = tl.where(target_one, tl.load(up_ptr + 2 * (r * N_BITS + i) + 1),
                          tl.load(dn_ptr + 2 * (r * N_BITS + i) + 1))
            sr = tl.load(cur_ptr + base + 2 * src)
            si = tl.load(cur_ptr + base + 2 * src + 1)
            acc_r += kr * sr - ki * si
            acc_i += kr * si + ki * sr
        for p in range(n_pairs):
            bi = tl.load(bits_ptr + 2 * p)
            bj = tl.load(bits_ptr + 2 * p + 1)
            a = tl.where(((offs >> bi) ^ (offs >> bj)) & 1 == 1, tl.load(hop_ptr + 2 * (r * n_pairs + p) + 1),
                         tl.load(hop_ptr + 2 * (r * n_pairs + p)))
            src = offs ^ ((1 << bi) | (1 << bj))
            acc_r += a * tl.load(cur_ptr + base + 2 * src)
            acc_i += a * tl.load(cur_ptr + base + 2 * src + 1)
        if RECURSION:
            acc_r = 2.0 * acc_r - tl.load(prev_ptr + base + 2 * offs)
            acc_i = 2.0 * acc_i - tl.load(prev_ptr + base + 2 * offs + 1)
        tl.store(out_ptr + base + 2 * offs, acc_r)
        tl.store(out_ptr + base + 2 * offs + 1, acc_i)

    @ryd.triton.jit
    def _series_kernel(psi_ptr, buf_ptr, coef_ptr, diag_ptr, up_ptr, dn_ptr, bits_ptr, hop_ptr, out_ptr, last_ptr,
                       n_terms, n_pairs, N_BITS: tl.constexpr, DIM: tl.constexpr, MP: tl.constexpr,
                       ROW_H: tl.constexpr, COEF_ROW: tl.constexpr):
        """One program per row r: out[r, m] = sum_k coef[k, m] T_k(H~) psi[r], k < n_terms, and last[r] = ||T_{K-1}||^2.
        T_k of the row lives in this program's registers (own entries) and in two scratch rows of buf (read by the
        flips of other threads), with a block barrier after each store; the scratch loads are volatile and the
        kernel is launched without software pipelining (num_stages=1). ROW_H: H~ indexed by the row; COEF_ROW:
        coef is (R, n_terms, MP), one table per row, else (n_terms, MP)."""
        r = tl.program_id(0).to(tl.int64)
        offs = tl.arange(0, DIM)
        moff = tl.arange(0, MP)
        vb = 2 * r * DIM
        hb = vb if ROW_H else 0
        nb = r * N_BITS if ROW_H else 0
        pb = r * n_pairs if ROW_H else 0
        cb = r * n_terms * MP if COEF_ROW else 0
        a_buf = buf_ptr + 4 * r * DIM
        dr = tl.load(diag_ptr + hb + 2 * offs)
        di = tl.load(diag_ptr + hb + 2 * offs + 1)
        cur_r = tl.load(psi_ptr + vb + 2 * offs)
        cur_i = tl.load(psi_ptr + vb + 2 * offs + 1)
        c_r = tl.load(coef_ptr + 2 * (cb + moff))
        c_i = tl.load(coef_ptr + 2 * (cb + moff) + 1)
        acc_r = c_r[:, None] * cur_r[None, :] - c_i[:, None] * cur_i[None, :]
        acc_i = c_r[:, None] * cur_i[None, :] + c_i[:, None] * cur_r[None, :]
        # T_{k-2}: unused at k = 1. Not initialised as `prev_r = cur_r`: Triton then merges the two loop-carried
        # values (same initial value) and prev never advances (T_3 came out as 2 H T_2 - T_0)
        prev_r = tl.zeros([DIM], dtype=cur_r.dtype)
        prev_i = tl.zeros([DIM], dtype=cur_i.dtype)
        tl.store(a_buf + 2 * offs, cur_r)
        tl.store(a_buf + 2 * offs + 1, cur_i)
        tl.debug_barrier()
        for k in range(1, n_terms):
            src_buf = a_buf + ((k - 1) % 2) * (2 * DIM)  # T_{k-1}; buffers alternate by parity (no carried pointers)
            dst_buf = a_buf + (k % 2) * (2 * DIM)
            hr = dr * cur_r - di * cur_i
            hi = dr * cur_i + di * cur_r
            for i in tl.static_range(N_BITS):
                shift = N_BITS - 1 - i
                src = offs ^ (1 << shift)
                one = ((offs >> shift) & 1) == 1
                kr = tl.where(one, tl.load(up_ptr + 2 * (nb + i)), tl.load(dn_ptr + 2 * (nb + i)))
                ki = tl.where(one, tl.load(up_ptr + 2 * (nb + i) + 1), tl.load(dn_ptr + 2 * (nb + i) + 1))
                sr = tl.load(src_buf + 2 * src, volatile=True)
                si = tl.load(src_buf + 2 * src + 1, volatile=True)
                hr += kr * sr - ki * si
                hi += kr * si + ki * sr
            for p in range(n_pairs):
                bi = tl.load(bits_ptr + 2 * p)
                bj = tl.load(bits_ptr + 2 * p + 1)
                a = tl.where(((offs >> bi) ^ (offs >> bj)) & 1 == 1, tl.load(hop_ptr + 2 * (pb + p) + 1),
                             tl.load(hop_ptr + 2 * (pb + p)))
                s = offs ^ ((1 << bi) | (1 << bj))
                hr += a * tl.load(src_buf + 2 * s, volatile=True)
                hi += a * tl.load(src_buf + 2 * s + 1, volatile=True)
            two = tl.where(k > 1, 2.0, 1.0)  # T_1 = H~ T_0, T_k = 2 H~ T_{k-1} - T_{k-2}
            one_prev = tl.where(k > 1, 1.0, 0.0)
            new_r = two * hr - one_prev * prev_r
            new_i = two * hi - one_prev * prev_i
            c_r = tl.load(coef_ptr + 2 * (cb + k * MP + moff))
            c_i = tl.load(coef_ptr + 2 * (cb + k * MP + moff) + 1)
            acc_r += c_r[:, None] * new_r[None, :] - c_i[:, None] * new_i[None, :]
            acc_i += c_r[:, None] * new_i[None, :] + c_i[:, None] * new_r[None, :]
            tl.store(dst_buf + 2 * offs, new_r)
            tl.store(dst_buf + 2 * offs + 1, new_i)
            tl.debug_barrier()
            prev_r = cur_r
            prev_i = cur_i
            cur_r = new_r
            cur_i = new_i
        o = 2 * ((r * MP + moff[:, None]) * DIM + offs[None, :])
        tl.store(out_ptr + o, acc_r)
        tl.store(out_ptr + o + 1, acc_i)
        tl.store(last_ptr + r, tl.sum(cur_r * cur_r + cur_i * cur_i, axis=0))


# ---------------------------------------------------------------- self-check

if __name__ == "__main__":
    import dataclasses
    import itertools
    import warnings

    from emu_base import HamiltonianType
    warnings.filterwarnings("ignore", message="long double is float64")
    rng0 = np.random.default_rng(0)
    relax, deph = np.sqrt(2.0) * np.array([[0, 1], [0, 0]], complex), np.sqrt(1.0 / 2) * np.diag([1.0, -1.0])
    n = 3
    J = np.triu(rng0.uniform(1, 5, (n, n)), 1)
    J = J + J.T
    fractions = np.linspace(0, 1, 9)
    base = ref._fake_sequence_data(n, rng0.uniform(1, 4, (4, n)), rng0.uniform(-2, 2, (4, n)), 0.3, J[None],
                                   [relax, deph], HamiltonianType.Rydberg, steps=4, dt=500)

    def compare(label, sds, **kw):
        for vec in (False, True):
            a, b = _compare(f"{label}, vectorized={vec}", sds, vectorized=vec, **kw)
        return a, b

    def _compare(label, sds, **kw):
        a = fast.run_sequence_data(sds, fractions, seed=5)
        b = run_sequence_data(sds, fractions, seed=5, **kw)
        err = float((a.occupations - b.occupations).abs().max())
        print(f"{label}: max |d<n>| vs noisy_cheb_fast trajectory by trajectory = {err:.1e}; jumps {b.stats['jumps']} "
              f"(fast {a.stats['jumps']}), batches {b.stats['batches']} (fast {a.stats['batches']}), row matvecs "
              f"{b.stats['row_matvecs'] + b.stats['lanczos_matvecs']} (fast {a.stats['row_matvecs']}), lanczos rejected "
              f"{b.stats['lanczos_rejected']}")
        assert err < 1e-8 and a.stats["jumps"] == b.stats["jumps"], (label, err)
        return a, b

    identical = list(itertools.repeat(base, 16))
    compare("1. shared H, Gershgorin", identical, bounds="gershgorin")
    compare("2. shared H, Lanczos", identical, bounds="lanczos")
    # shot-to-shot: every sample its own amplitude / detuning / couplings (same structure)
    samples = [dataclasses.replace(base, omega=base.omega * (1 + 0.1 * rng0.normal()),
                                   delta=base.delta + 0.3 * rng0.normal(),
                                   interaction_matrix=(lambda s: (lambda t: torch.as_tensor(J[None] * s)))(1 + 0.05 * rng0.normal()))
               for _ in range(16)]
    compare("3. shot-to-shot samples batched, Gershgorin", samples, bounds="gershgorin")
    compare("4. shot-to-shot samples batched, Lanczos", samples, bounds="lanczos")
    _TurboBatchV._small_ok = lambda self, psi, m=1: True  # Newton jump location on CPU too (the GPU uses it by default)
    _TurboBatchV._propagate_rows = lambda self, h, rows, psi, off: fast._Batch._propagate(
        self, h, self._step(h, rows), psi, np.asarray(off, float))
    compare("4b. Newton jump location forced on CPU", samples, bounds="lanczos")
    weird = 1.5 * np.array([[0.3, 0.5 + 0.2j], [0.1j, -0.2]])  # general eff_noise: L^dag L has off-diagonal terms
    general = [dataclasses.replace(sd, lindblad_ops=[torch.as_tensor(weird)]) for sd in samples]
    compare("4c. general eff_noise (off-diagonal L^dag L), Newton forced", general, bounds="lanczos")
    LANCZOS_MARGIN = -0.3  # interval deliberately too narrow: the guard must reject it and fall back
    _, b = compare("5. too-narrow Lanczos interval (guard)", samples, bounds="lanczos")
    assert b.stats["lanczos_rejected"] >= 1
    print("self-check OK")
