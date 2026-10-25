"""Check of multi_gpu (idea 6): one GPU vs two on the same samples, trajectory by trajectory and wall time.
Usage (scripts dir, 2 GPUs): python experiments/multi_gpu_check.py N N_TRAJ DURATION_NS"""
import os  # noqa: E402
import sys  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # scripts/: the engines
import os
import sys
import time
import warnings

import numpy as np
import pulser
import torch
from pulser.devices import MockDevice


def main():
    warnings.filterwarnings("ignore")
    from emu_base import PulserData
    from emu_mps import MPSConfig

    import multi_gpu
    import noisy_cheb_turbo as turbo
    n, n_traj, duration = (int(x) for x in sys.argv[1:4])
    nm = pulser.devices.Device.from_abstract_repr(open(os.environ["FRESNEL_SPECS"]).read()).default_noise_model
    seq = pulser.Sequence(pulser.Register.rectangle(1, n, spacing=7.0, prefix="q"), MockDevice)
    seq.declare_channel("r", "rydberg_global")
    seq.add(pulser.Pulse.ConstantPulse(duration, 2 * np.pi, 3.0, 0.0), "r")
    sds = list(PulserData(sequence=seq, config=MPSConfig(dt=10, noise_model=nm, n_trajectories=n_traj,
                                                         optimize_qubit_ordering=False), dt=10).get_sequences())
    fr = np.linspace(0, 1, 6)
    turbo.run_sequence_data(sds[:2], fr, device="cuda:0")  # warm-up / compile
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    one = turbo.run_sequence_data(sds, fr, seed=0, device="cuda:0")
    torch.cuda.synchronize()
    t_one = time.perf_counter() - t0
    t0 = time.perf_counter()
    two = multi_gpu.run_sequence_data(sds, fr, seed=0, devices=("cuda:0", "cuda:1"))
    t_two = time.perf_counter() - t0
    compute = [w["compute_s"] for w in two.stats["workers"]]
    err = float((one.occupations - two.occupations).abs().max())
    print(f"n={n}, {n_traj} traj, {duration} ns: 1 GPU {t_one:6.1f} s | 2 GPUs wall {t_two:6.1f} s (worker compute "
          f"{', '.join(f'{c:.1f}' for c in compute)} s; start-up ~{t_two - max(compute):.1f} s) -> x{t_one / t_two:.2f} wall, "
          f"x{t_one / max(compute):.2f} compute | max |d<n>| 1 vs 2 GPUs {err:.1e}", flush=True)
    assert err < 1e-12


if __name__ == "__main__":
    main()
