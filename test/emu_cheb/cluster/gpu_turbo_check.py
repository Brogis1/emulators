"""Quick GPU check of noisy_cheb_turbo (small n): small-register series kernel and per-row (shot-to-shot) kernels vs
noisy_cheb_fast trajectory by trajectory (same streams), then ms/trajectory. Usage (repo root): python test/emu_cheb/cluster/gpu_turbo_check.py"""
import os
import time
import warnings

import numpy as np
import pulser
import torch
from pulser.devices import MockDevice

from emu_cheb.solvers import noisy_cheb_fast as fast
from emu_cheb.solvers import noisy_cheb_turbo as turbo

warnings.filterwarnings("ignore")
FR = np.linspace(0, 1, 11)
fresnel = pulser.devices.Device.from_abstract_repr(open(os.environ["FRESNEL_SPECS"]).read())
nm_fresnel = fresnel.default_noise_model
nm_strong = pulser.NoiseModel(relaxation_rate=0.5, dephasing_rate=0.5)
nm_shot = pulser.NoiseModel(amp_sigma=0.1, laser_waist=40.0, temperature=50.0, relaxation_rate=0.3, dephasing_rate=0.3)
print(torch.cuda.get_device_name(), flush=True)


def chain(n, duration):
    seq = pulser.Sequence(pulser.Register.rectangle(1, n, spacing=7.0, prefix="q"), MockDevice)
    seq.declare_channel("r", "rydberg_global")
    seq.add(pulser.Pulse.ConstantPulse(duration, 2 * np.pi, 3.0, 0.0), "r")
    return seq


def sds(seq, nm, n_traj):
    from emu_base import PulserData
    from emu_mps import MPSConfig
    return list(PulserData(sequence=seq, config=MPSConfig(dt=10, noise_model=nm, n_trajectories=n_traj,
                                                          optimize_qubit_ordering=False), dt=10).get_sequences())


def timed(fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, time.perf_counter() - t0


# 1. exactness on GPU: turbo (small kernel, shared H / per-row H, both bounds) vs fast
for n in (6, 9, 12):
    for label, nm in (("shared H (Lindblad)", nm_strong), ("per-row H (shot-to-shot + Lindblad)", nm_shot)):
        data = sds(chain(n, 1000), nm, 16)
        a = fast.run_sequence_data(data, FR, seed=3, device="cuda")
        for bounds, vec in (("gershgorin", False), ("lanczos", False), ("gershgorin", True), ("lanczos", True)):
            b = turbo.run_sequence_data(data, FR, seed=3, device="cuda", bounds=bounds, vectorized=vec)
            err = float((a.occupations - b.occupations).abs().max())
            print(f"n={n:2d} {label:36s} {bounds:10s} vec={vec!s:5s}: max |d<n>| vs fast {err:.1e}, jumps {b.stats['jumps']}/{a.stats['jumps']}, "
                  f"small-kernel calls {b.stats['small_kernel_calls']}, batches {b.stats['batches']}/{a.stats['batches']}, "
                  f"lanczos rejected {b.stats['lanczos_rejected']}", flush=True)
            assert err < 1e-9 and a.stats["jumps"] == b.stats["jumps"]

# 2. speed, small registers
for n in (8, 10, 12):
    for label, nm, n_traj in (("FRESNEL 2us", nm_fresnel, 512), ("shot-to-shot 2us", nm_shot, 256)):
        data = sds(chain(n, 2000), nm, n_traj)
        fast.run_sequence_data(data[:4], FR, device="cuda")  # warm-up / compile
        turbo.run_sequence_data(data[:4], FR, device="cuda", vectorized=False)
        turbo.run_sequence_data(data[:4], FR, device="cuda")
        _, t_fast = timed(lambda: fast.run_sequence_data(data, FR, device="cuda"))
        _, t_t1 = timed(lambda: turbo.run_sequence_data(data, FR, device="cuda", vectorized=False))
        res, t_t = timed(lambda: turbo.run_sequence_data(data, FR, device="cuda"))
        print(f"n={n:2d} {label:17s} ({n_traj} traj): fast {t_fast / n_traj * 1e3:6.2f} ms/traj | turbo (loop jumps) "
              f"{t_t1 / n_traj * 1e3:6.2f} | turbo (vectorized) {t_t / n_traj * 1e3:6.2f} ms/traj (x{t_fast / t_t:.1f} vs fast; "
              f"small-kernel calls {res.stats['small_kernel_calls']}, jumps {res.stats['jumps']})", flush=True)
