"""multi_gpu: EXPERIMENTAL - noisy trajectories split over several GPUs, one process per GPU (idea 6).

Each process runs its share of the trajectories with noisy_cheb_turbo's batch classes and the same per-trajectory
random streams as a single-GPU run (SeedSequence(seed).spawn(N)[i] for trajectory i), so the result is identical to
noisy_cheb_turbo.run_sequence_data trajectory by trajectory. The validated engines are not modified: the batching loop
of noisy_cheb_turbo.run_sequence_data is repeated here with explicit seeds.
Trajectories are independent, so the speed-up is the number of GPUs, minus the start-up of the worker processes
(imports, ~10-20 s): worth it for runs of minutes, not seconds. SequenceData must be picklable (no lambdas: a
Heisenberg `replace_sequence_data` must be a module-level function).

Usage:  res = multi_gpu.run_sequence_data(sds, eval_fractions, seed=0, devices=("cuda:0", "cuda:1"))
"""
from __future__ import annotations


import time

import numpy as np
import torch

from emu_cheb.solvers import noisy_cheb as ref
from emu_cheb.solvers import noisy_cheb_fast as fast
from emu_cheb.solvers import noisy_cheb_turbo as turbo


def run_sequence_data(sequences, eval_fractions=(1.0,), seed: int = 0, devices=("cuda:0", "cuda:1"),
                      single: bool = False, bounds: str = "lanczos", small_kernel: bool = True,
                      initial_state: str | None = None) -> ref.NoisyResults:
    """As noisy_cheb_turbo.run_sequence_data, with the trajectories split over `devices` (contiguous chunks)."""
    import torch.multiprocessing as mp
    sds = list(sequences)
    seeds = np.random.SeedSequence(seed).spawn(len(sds))
    chunks = [c for c in np.array_split(np.arange(len(sds)), len(devices)) if len(c)]
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    procs = [ctx.Process(target=_worker, args=(dev, [sds[i] for i in c], [seeds[i] for i in c], c, eval_fractions,
                                                 dict(single=single, bounds=bounds, small_kernel=small_kernel,
                                                      initial_state=initial_state), queue))
             for dev, c in zip(devices, chunks)]
    for p in procs:
        p.start()
    parts = [queue.get() for _ in procs]
    for p in procs:
        p.join()
    errors = [part["error"] for part in parts if "error" in part]
    if errors:
        raise RuntimeError("worker failed:\n" + errors[0])
    parts = [{k: (torch.as_tensor(v) if isinstance(v, np.ndarray) and k != "times" else v) for k, v in part.items()}
             for part in parts]
    T, n_total = parts[0]["occ"].shape[1:]
    occ = torch.zeros(len(sds), T, n_total, dtype=torch.float64)
    leaked = torch.zeros(len(sds), T, n_total, dtype=torch.bool)
    P = sum(part["P"] for part in parts)
    corr = sum(part["corr"] for part in parts)
    stats = {"workers": [part["stats"] for part in parts]}
    for part in parts:
        occ[part["rows"]] = part["occ"]
        leaked[part["rows"]] = part["leaked"]
    return ref.NoisyResults(times_ns=parts[0]["times"], probabilities=P / len(sds), occupations=occ,
                            correlations=corr / len(sds), leaked=leaked, stats=stats)


def _worker(device, sds, seeds, rows, eval_fractions, kw, queue):
    import traceback
    try:
        torch.cuda.set_device(torch.device(device))  # Triton launches on the current device
        t0 = time.perf_counter()
        out = run_with_seeds(sds, seeds, eval_fractions, device, **kw)
        out["stats"]["compute_s"] = time.perf_counter() - t0
        out["rows"] = list(rows)
        # numpy, not tensors: torch would send shared-memory handles that die with this process
        queue.put({k: (v.numpy() if torch.is_tensor(v) else v) for k, v in out.items()})
    except BaseException:  # noqa: BLE001 - report to the parent instead of hanging it
        queue.put({"error": traceback.format_exc()})


def run_with_seeds(sds, seeds, eval_fractions, device, single=False, bounds="lanczos", small_kernel=True,
                   initial_state=None) -> dict:
    """noisy_cheb_turbo.run_sequence_data's loop with given per-trajectory seeds: sums of P and <n_i n_j> (CPU),
    per-trajectory occupations and leaks, times, stats."""
    device = torch.device(device)
    dtype = torch.complex64 if single else torch.complex128
    stats = dict(trajectories=len(sds), jumps=0, leaks=0, segments=0, matvecs=0, row_matvecs=0, jump_norm_dev=0.0,
                 max_growth=0.0, batches=0, lanczos_matvecs=0, lanczos_rejected=0, small_kernel_calls=0)
    torch.cuda.empty_cache()
    memory_bytes = 0.4 * torch.cuda.mem_get_info(device)[0]
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
        groups.setdefault(turbo._batch_key(m), []).append(i)
    out = dict(P=torch.zeros(T, 2 ** n_total, dtype=torch.float64, device=device),
               corr=torch.zeros(T, n_total, n_total, dtype=torch.float64, device=device),
               occ=torch.zeros(len(sds), T, n_total, dtype=torch.float64),
               leaked=torch.zeros(len(sds), T, n_total, dtype=torch.bool))
    for idxs in groups.values():
        dim = 2 ** int(models[idxs[0]].good.sum())
        per_row = fast.ROW_VECTORS + (0 if all(models[i] is models[idxs[0]] for i in idxs) else 3)
        r_max = max(1, int(memory_bytes / (per_row * dim * (8 if single else 16))))
        for c0 in range(0, len(idxs), r_max):
            chunk = idxs[c0:c0 + r_max]
            part = turbo._TurboBatchV([models[i] for i in chunk], [seeds[i] for i in chunk], times_ns, dtype, stats,
                                      initial_state, bounds, small_kernel).run()
            out["P"] += part["P"]
            out["corr"] += part["corr"]
            out["occ"][chunk] = part["occ"]
            out["leaked"][chunk] = part["leaked"]
            stats["batches"] += 1
    return dict(P=out["P"].cpu(), corr=out["corr"].cpu(), occ=out["occ"], leaked=out["leaked"], times=times_ns,
                stats=stats)
