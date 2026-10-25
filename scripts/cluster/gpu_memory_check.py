"""GPU memory check of every engine at large registers: peak allocated memory against the free memory, with a
time-dependent drive (one piece per 10 ns step: exercises the Hamiltonian caches) and FRESNEL's noise, each run twice
in the same process (no growth between runs). Usage (scripts dir, A100): python cluster/gpu_memory_check.py"""
import os  # noqa: E402
import sys  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # scripts/: the engines
import os
import time
import warnings

import numpy as np
import pulser
import torch
from pulser.devices import MockDevice

import heisenberg_cheb_fast as heis
import noisy_cheb as ref
import noisy_cheb_fast as fast
import noisy_cheb_turbo as turbo

warnings.filterwarnings("ignore")
FR = np.linspace(0, 1, 6)
nm = pulser.devices.Device.from_abstract_repr(open(os.environ["FRESNEL_SPECS"]).read()).default_noise_model
total = torch.cuda.get_device_properties(0).total_memory
print(torch.cuda.get_device_name(), f"{total / 2 ** 30:.1f} GiB", flush=True)


def shaped(n, duration=500):
    seq = pulser.Sequence(pulser.Register.rectangle(1, n, spacing=7.0, prefix="q"), MockDevice)
    seq.declare_channel("r", "rydberg_global")
    seq.add(pulser.Pulse(pulser.BlackmanWaveform(duration, np.pi), pulser.RampWaveform(duration, -2, 2), 0.0), "r")
    return seq


def peak(label, fn):
    for rep in range(2):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        free0 = torch.cuda.mem_get_info()[0]
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        print(f"{label:46s} run {rep}: {time.perf_counter() - t0:7.1f} s, peak allocated "
              f"{torch.cuda.max_memory_allocated() / 2 ** 30:6.2f} GiB of {free0 / 2 ** 30:5.1f} GiB free", flush=True)


for n in (20, 22, 24):
    seq = shaped(n)
    peak(f"n={n} noisy_cheb (reference), 1 traj", lambda: ref.simulate(seq, nm, 1, FR, device="cuda"))
    peak(f"n={n} noisy_cheb_fast, 8 traj", lambda: fast.simulate(seq, nm, 8, FR, device="cuda"))
    peak(f"n={n} noisy_cheb_turbo, 8 traj", lambda: turbo.simulate(seq, nm, 8, FR, device="cuda"))

for n in (24, 26):
    c = np.diag(np.ones(n - 1), 1)
    J = np.stack([c + c.T] * 3)[None]
    peak(f"n={n} heisenberg evolve (quench), 11 times", lambda: heis.evolve(J, np.full((1, n), 2.0), np.full((1, n), 1.0),
                                                                            np.linspace(0, 2, 11), device="cuda"))
n = 22
c = np.diag(np.ones(n - 1), 1)
peak(f"n={n} heisenberg evolve_noisy, 4 traj", lambda: heis.evolve_noisy(np.stack([c + c.T] * 3), np.full(n, 2.0),
                                                                         np.full(n, 1.0), np.linspace(0, 1, 6),
                                                                         dephasing_rate=0.5, n_traj=4, device="cuda"))
print("memory check done", flush=True)
