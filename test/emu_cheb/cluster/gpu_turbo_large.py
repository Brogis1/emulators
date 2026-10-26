"""noisy_cheb_turbo at larger registers (A100): exactness vs noisy_cheb_fast at n=16 (same streams), then ms/trajectory
and matvecs per trajectory for fast, turbo (Lanczos) and turbo (Gershgorin), FRESNEL noise, 4 us, 11 output times."""
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
nm_fresnel = pulser.devices.Device.from_abstract_repr(open(os.environ["FRESNEL_SPECS"]).read()).default_noise_model
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


for label, nm in (("FRESNEL", nm_fresnel), ("shot-to-shot", nm_shot)):
    data = sds(chain(16, 1000), nm, 8)
    a = fast.run_sequence_data(data, FR, seed=3, device="cuda")
    for bounds in ("gershgorin", "lanczos"):
        b = turbo.run_sequence_data(data, FR, seed=3, device="cuda", bounds=bounds)
        err = float((a.occupations - b.occupations).abs().max())
        print(f"n=16 {label:12s} {bounds:10s}: max |d<n>| vs fast {err:.1e}, jumps {b.stats['jumps']}/{a.stats['jumps']}, "
              f"lanczos rejected {b.stats['lanczos_rejected']}", flush=True)
        assert err < 1e-9 and a.stats["jumps"] == b.stats["jumps"]

for n, n_traj in ((14, 128), (16, 64), (18, 32), (20, 16)):
    data = sds(chain(n, 4000), nm_fresnel, n_traj)
    fast.run_sequence_data(data[:2], FR, device="cuda")  # warm-up / compile
    turbo.run_sequence_data(data[:2], FR, device="cuda")
    ra, t_fast = timed(lambda: fast.run_sequence_data(data, FR, device="cuda"))
    rl, t_l = timed(lambda: turbo.run_sequence_data(data, FR, device="cuda", bounds="lanczos"))
    rg, t_g = timed(lambda: turbo.run_sequence_data(data, FR, device="cuda", bounds="gershgorin"))
    print(f"n={n:2d} FRESNEL 4us ({n_traj} traj, batches fast {ra.stats['batches']} turbo {rl.stats['batches']}): fast {t_fast / n_traj * 1e3:8.1f} ms/traj ({ra.stats['row_matvecs'] / n_traj:.0f} "
          f"matvecs/traj) | turbo Lanczos {t_l / n_traj * 1e3:8.1f} ({(rl.stats['row_matvecs'] + rl.stats['lanczos_matvecs']) / n_traj:.0f}, "
          f"rejected {rl.stats['lanczos_rejected']}) | turbo Gershgorin {t_g / n_traj * 1e3:8.1f} ({rg.stats['row_matvecs'] / n_traj:.0f})"
          f" -> best turbo x{t_fast / min(t_l, t_g):.2f} vs fast", flush=True)
