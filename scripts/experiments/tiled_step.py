"""tiled_step: EXPERIMENTAL IO-aware Chebyshev step (idea 2), a drop-in for noisy_cheb_fast._make_batched_step.

Why: the step kernels of the engines handle 256 consecutive amplitudes per program, so a flip of one of the 8 lowest
bits stays in the program but every other flip (and every pair flip) reads a distant part of the state: ~(n - 8 +
pairs) reads of the whole state per Chebyshev step, the bottleneck once the state no longer fits in the GPU's L2
(n >~ 20).
How: one pass handles a set S of bits: the 3 lowest (8 consecutive amplitudes = 128 bytes, coalesced loads) plus a
group of up to GROUP higher bits. Each program loads the 2^|S| amplitudes {base + any combination of the bits of S}
into registers and applies every flip of a bit in S, and every pair flip of two adjacent bits of S, in registers (a
flip of local bit k is a flip of axis 1 of the tile reshaped to (2^(|S|-1-k), 2, 2^k)). The groups of higher bits
overlap by one bit so that the pair flips of neighbouring atoms are inside one pass. Passes accumulate H~ x into a
scratch vector y (the first adds the diagonal, the last applies T_{k+1} = 2 H~ T_k - T_{k-1}): ~3 state reads/writes
per pass, 2-3 passes for n = 13-24, instead of ~n.
Limits of the prototype: H~ shared by the rows (noisy_cheb_fast's shared-H batches), pair flips only between atoms
whose bits are adjacent (chains: Heisenberg / XY nearest neighbours; Rydberg interactions are diagonal anyway).
Same dtype rules as the engines (complex128 / complex64 state, real pair amplitudes).
Status: correct (2e-16 / 1e-7 against noisy_cheb_fast._make_batched_step, n = 10-22) but SLOWER than the
current kernel at n = 22 (best 0.64 vs 0.52 ms per step): Triton's in-tile permutations (reshape + flip or
tl.gather) cost more than the DRAM traffic they save. A gain needs a CUDA kernel with an explicit shared-memory tile.
"""
from __future__ import annotations

import numpy as np
import torch
import triton
import triton.language as tl

LOW, GROUP = 3, 9  # tile = 2^(LOW + GROUP) = 4096 amplitudes per program


def plan(n: int, low: int = LOW, group: int = GROUP) -> list[list[int]]:
    """Bit sets of the passes (sorted): the low bits plus overlapping groups of the higher bits."""
    if n <= low + group:
        return [list(range(n))]
    passes, start = [], low
    while True:
        end = min(n, start + group)
        passes.append(list(range(low)) + list(range(start, end)))
        if end == n:
            return passes
        start = end - 1  # overlap: pairs of adjacent bits across groups stay inside one pass


@triton.jit
def _flip(v, lidx, K: tl.constexpr, S: tl.constexpr, GATHER: tl.constexpr):
    """v[l] -> v[l ^ (1 << K)] for a tile of 2^S values: tl.gather, or reshape + flip of an axis of size 2."""
    if GATHER:
        return tl.gather(v, lidx ^ (1 << K), 0)
    w = tl.reshape(v, [2 ** (S - 1 - K), 2, 2 ** K])
    w = tl.flip(w, 1)
    return tl.reshape(w, [2 ** S])


@triton.jit
def _pass_kernel(x_ptr, y_ptr, prev_ptr, out_ptr, diag_ptr, ss_ptr, so_ptr, amp_ptr, pamp_ptr, dim,
                 S: tl.constexpr, FLIPMASK: tl.constexpr, PAIRMASK: tl.constexpr, FIRST: tl.constexpr,
                 LAST: tl.constexpr, RECURSION: tl.constexpr, GATHER: tl.constexpr):
    """One pass over bit set S for row program_id(1), outer index program_id(0). x, y, prev, out: interleaved
    (re, im) rows of length dim; amp: (S, 4) [up re, up im, dn re, dn im] per local bit; pamp: (S, 2) [equal bits,
    different bits] for the pair (local k, k + 1)."""
    o = tl.program_id(0)
    r = tl.program_id(1).to(tl.int64)
    lidx = tl.arange(0, 2 ** S)
    hidx = tl.load(so_ptr + o) + tl.load(ss_ptr + lidx)  # index inside a row
    vidx = r * dim + hidx
    xr = tl.load(x_ptr + 2 * vidx)
    xi = tl.load(x_ptr + 2 * vidx + 1)
    if FIRST:
        dr = tl.load(diag_ptr + 2 * hidx)
        di = tl.load(diag_ptr + 2 * hidx + 1)
        acc_r = dr * xr - di * xi
        acc_i = dr * xi + di * xr
    else:
        acc_r = tl.load(y_ptr + 2 * vidx)
        acc_i = tl.load(y_ptr + 2 * vidx + 1)
    for k in tl.static_range(S):
        if (FLIPMASK >> k) & 1:
            fr = _flip(xr, lidx, k, S, GATHER)
            fi = _flip(xi, lidx, k, S, GATHER)
            one = ((lidx >> k) & 1) == 1  # target bit 1: up (|1><0|), else dn (|0><1|)
            kr = tl.where(one, tl.load(amp_ptr + 4 * k), tl.load(amp_ptr + 4 * k + 2))
            ki = tl.where(one, tl.load(amp_ptr + 4 * k + 1), tl.load(amp_ptr + 4 * k + 3))
            acc_r += kr * fr - ki * fi
            acc_i += kr * fi + ki * fr
    for k in tl.static_range(S - 1):
        if (PAIRMASK >> k) & 1:
            fr = _flip(_flip(xr, lidx, k, S, GATHER), lidx, k + 1, S, GATHER)
            fi = _flip(_flip(xi, lidx, k, S, GATHER), lidx, k + 1, S, GATHER)
            differ = (((lidx >> k) ^ (lidx >> (k + 1))) & 1) == 1
            a = tl.where(differ, tl.load(pamp_ptr + 2 * k + 1), tl.load(pamp_ptr + 2 * k))
            acc_r += a * fr
            acc_i += a * fi
    if LAST:
        if RECURSION:
            acc_r = 2.0 * acc_r - tl.load(prev_ptr + 2 * vidx)
            acc_i = 2.0 * acc_i - tl.load(prev_ptr + 2 * vidx + 1)
        tl.store(out_ptr + 2 * vidx, acc_r)
        tl.store(out_ptr + 2 * vidx + 1, acc_i)
    else:
        tl.store(y_ptr + 2 * vidx, acc_r)
        tl.store(y_ptr + 2 * vidx + 1, acc_i)


def make_tiled_step(diag, up, dn, hop, pairs, n, low: int = LOW, group: int = GROUP, num_warps: int = 8,
                    gather: bool = False):
    """step(cur, prev, out) on (R, 2^n) rows sharing H~ (same arguments as noisy_cheb_fast._make_batched_step)."""
    device, dim = diag.device, 2 ** n
    real = torch.float32 if diag.dtype == torch.complex64 else torch.float64
    passes = plan(n, low, group)
    up_c, dn_c = up.cpu().numpy(), dn.cpu().numpy()
    hop_c = hop.reshape(-1, 2).cpu().numpy()
    assigned_flip = set()
    pair_bits = [tuple(sorted((n - 1 - int(i), n - 1 - int(j)))) for i, j in np.asarray(pairs).reshape(-1, 2)]
    pair_done = [False] * len(pair_bits)
    configs = []
    for S in passes:
        s = len(S)
        local = {b: k for k, b in enumerate(S)}
        ss = torch.as_tensor([sum(((l >> k) & 1) << b for k, b in enumerate(S)) for l in range(2 ** s)],
                             dtype=torch.int64, device=device)
        outer = [b for b in range(n) if b not in local]
        so = torch.as_tensor([sum(((o >> k) & 1) << b for k, b in enumerate(outer)) for o in range(2 ** len(outer))],
                             dtype=torch.int64, device=device)
        amp = np.zeros((s, 4))
        flipmask = 0
        for b in S:
            if b not in assigned_flip:
                assigned_flip.add(b)
                atom = n - 1 - b
                k = local[b]
                flipmask |= 1 << k
                amp[k] = [up_c[atom].real, up_c[atom].imag, dn_c[atom].real, dn_c[atom].imag]
        pamp = np.zeros((s, 2))
        pairmask = 0
        for q, (b1, b2) in enumerate(pair_bits):
            if not pair_done[q] and b1 in local and b2 in local and local[b2] == local[b1] + 1:
                pair_done[q] = True
                pairmask |= 1 << local[b1]
                pamp[local[b1]] = hop_c[q]
        configs.append(dict(S=s, ss=ss, so=so, FLIPMASK=flipmask, PAIRMASK=pairmask,
                            amp=torch.as_tensor(amp, dtype=real, device=device),
                            pamp=torch.as_tensor(pamp, dtype=real, device=device)))
    if not all(pair_done):
        raise NotImplementedError("tiled_step: pair flips only between atoms whose bits are adjacent")
    diag_r = torch.view_as_real(diag.contiguous())
    scratch = {}

    def step(cur, prev, out):
        R = cur.shape[0]
        if R not in scratch:
            scratch.clear()
            scratch[R] = torch.empty(R, dim, dtype=cur.dtype, device=device)
        y = torch.view_as_real(scratch[R]) if len(configs) > 1 else None
        x_r, o_r = torch.view_as_real(cur), torch.view_as_real(out)
        p_r = torch.view_as_real(cur if prev is None else prev)
        for p, c in enumerate(configs):
            _pass_kernel[(c["so"].numel(), R)](x_r, y if y is not None else o_r, p_r, o_r, diag_r, c["ss"], c["so"],
                                               c["amp"], c["pamp"], dim, S=c["S"], FLIPMASK=c["FLIPMASK"],
                                               PAIRMASK=c["PAIRMASK"], FIRST=p == 0, LAST=p == len(configs) - 1,
                                               RECURSION=prev is not None, GATHER=gather, num_warps=num_warps)
    step.passes = passes
    return step
