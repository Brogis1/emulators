"""n=18: fast vs turbo interleaved (fast, turbo, fast, turbo) on the same samples, with time split by component."""
import os  # noqa: E402
import sys  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # scripts/: the engines
import os
import time
import warnings
from collections import defaultdict

import numpy as np
import pulser
import torch
from pulser.devices import MockDevice

import noisy_cheb as ref
import noisy_cheb_fast as fast
import noisy_cheb_turbo as turbo

warnings.filterwarnings("ignore")
FR = np.linspace(0, 1, 11)
nm = pulser.devices.Device.from_abstract_repr(open(os.environ["FRESNEL_SPECS"]).read()).default_noise_model
n = int(os.environ.get("NQ", "18"))
seq = pulser.Sequence(pulser.Register.rectangle(1, n, spacing=7.0, prefix="q"), MockDevice)
seq.declare_channel("r", "rydberg_global")
seq.add(pulser.Pulse.ConstantPulse(4000, 2 * np.pi, 3.0, 0.0), "r")
from emu_base import PulserData
from emu_mps import MPSConfig
data = list(PulserData(sequence=seq, config=MPSConfig(dt=10, noise_model=nm, n_trajectories=32,
                                                      optimize_qubit_ordering=False), dt=10).get_sequences())
T = defaultdict(float)


def timer(cls, name):
    f = getattr(cls, name)

    def wrapped(*a, **k):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = f(*a, **k)
        torch.cuda.synchronize()
        T[f"{cls.__name__}.{name}"] += time.perf_counter() - t0
        return out
    setattr(cls, name, wrapped)


for cls, name in ((fast._Batch, "_propagate"), (fast._Batch, "_locate"), (ref._Model, "_jump"),
                  (turbo._TurboBatchV, "_jump_batch"), (turbo._TurboBatch, "_locate_rows"),
                  (fast._Batch, "_record")):
    timer(cls, name)
fast.run_sequence_data(data[:2], FR, device="cuda")
turbo.run_sequence_data(data[:2], FR, device="cuda")
for rep in range(2):
    for label, fn in (("fast", lambda: fast.run_sequence_data(data, FR, device="cuda")),
                      ("turbo", lambda: turbo.run_sequence_data(data, FR, device="cuda", bounds="gershgorin"))):
        T.clear()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        res = fn()
        torch.cuda.synchronize()
        total = time.perf_counter() - t0
        parts = ", ".join(f"{k} {v:.2f}" for k, v in sorted(T.items(), key=lambda x: -x[1]))
        print(f"rep {rep} {label:5s}: {total:6.2f} s, segments {res.stats['segments']}, jumps {res.stats['jumps']}, "
              f"batches {res.stats['batches']} | {parts}", flush=True)
