"""noisy_vs_emu_mps: our noisy engines against noisy emu-mps, channel by channel, pulse features, FRESNEL's model.

Every case is a small pulser sequence (3 atoms) run by emu-mps and by one of our engines (--engine reference |
fast | turbo) on the very same PulserData samples (one SequenceData per trajectory, as MPSBackend.run):
- deterministic cases (noiseless, pulse features, and shot-to-shot channels: amplitude, detuning incl. high
  frequency, Doppler, register, SPAM state preparation) are compared trajectory by trajectory: max |<n_i> difference|;
  the floor is emu-mps' own TDVP / truncation error (1e-15 for quenches, up to 1e-6 for XY couplings in 1/r^3);
- Lindblad channels (dephasing, depolarizing, relaxation, eff_noise, leakage) are random: trajectory means of <n_i>(t)
  are compared with z = |mean_mps - mean_ours| / (s sqrt(1 / N_mps + 1 / N_ours)), s the per-trajectory spread from a
  separate N_SPREAD-trajectory turbo sample (a sample's own spread misses rare early jumps and inflated z);
- readout errors (SPAM p_false_pos / p_false_neg): emu-mps' sampled bitstrings against readout(P): chi^2 when the
  trajectories are deterministic, else per-bitstring z from the trajectory-to-trajectory spread of the frequencies
  (shots of one trajectory share its P, so they are not independent).
"noise effect" = max |<n_i>_noisy - <n_i>_noiseless|, also in standard errors: a test only means something if that
number is large. emu-mps results of cases without shot-to-shot noise are cached (EMU_MPS_CACHE), so other engines
can be compared without rerunning emu-mps.

Usage: python test/emu_cheb/validation/noisy_vs_emu_mps.py [CASE ...] [--traj N] [--cheb-factor K] [--engine E] [--single]
(default: every case; our engine runs K x N trajectories for random channels). FRESNEL's spec is read from
FRESNEL_SPECS (abstract representation fetched from Pasqal Cloud, default ~/.cache/pasqal/fresnel_specs.json).
"""
from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import time
import warnings
from collections import Counter

import numpy as np
import pulser
import torch
from emu_base import PulserData
from emu_mps import BitStrings, MPSBackend, MPSConfig, Occupation
from pulser.devices import MockDevice
from scipy.stats import chi2

from emu_cheb.solvers import noisy_cheb
from emu_cheb.solvers import noisy_cheb_fast
from emu_cheb.solvers import noisy_cheb_turbo

ENGINE, SINGLE = "reference", False
CACHE = os.environ.get("EMU_MPS_CACHE", "/tmp/heis_gpu/emu_cache")
FRESNEL_SPECS = os.environ.get("FRESNEL_SPECS", os.path.expanduser("~/.cache/pasqal/fresnel_specs.json"))
DT, FRACTIONS, SHOTS = 10, np.linspace(0, 1, 11), 2000
N_SPREAD = 20000  # trajectories of the separate sample that sets the standard errors (random channels)
LEAK = np.zeros((3, 3))
LEAK[2, 0] = 1.0  # pulser basis (r, g, x): |x><r|

CHANNELS = {  # name: (noise model factory, device)
    "noiseless": (lambda: None, "mock"),
    "dephasing": (lambda: pulser.NoiseModel(dephasing_rate=1.0), "mock"),
    "depolarizing": (lambda: pulser.NoiseModel(depolarizing_rate=1.0), "mock"),
    "relaxation": (lambda: pulser.NoiseModel(relaxation_rate=1.0), "mock"),
    "eff_noise": (lambda: pulser.NoiseModel(eff_noise_rates=(0.8,), eff_noise_opers=(np.array([[0.3, 1], [0.2j, -0.3]]),)),
                  "mock"),
    "leakage": (lambda: pulser.NoiseModel(with_leakage=True, eff_noise_rates=(1.0,), eff_noise_opers=(LEAK,)), "mock"),
    "spam_prep": (lambda: pulser.NoiseModel(state_prep_error=0.3), "mock"),
    "readout": (lambda: pulser.NoiseModel(p_false_pos=0.1, p_false_neg=0.05), "mock"),
    "amplitude": (lambda: pulser.NoiseModel(amp_sigma=0.1, laser_waist=40.0), "mock"),
    "detuning": (lambda: pulser.NoiseModel(detuning_sigma=0.5), "mock"),
    "detuning_hf": (lambda: pulser.NoiseModel(detuning_hf_psd=(2.0, 1.0), detuning_hf_omegas=(5.0, 20.0)), "mock"),
    "doppler": (lambda: pulser.NoiseModel(temperature=50.0), "mock"),
    "register": (lambda: pulser.NoiseModel(trap_waist=1.0, trap_depth=150.0, temperature=50.0), "mock"),
    "fresnel_dephasing": (lambda: _fresnel_part("dephasing"), "fresnel"),
    "fresnel_relaxation": (lambda: _fresnel_part("relaxation"), "fresnel"),
    "fresnel_readout": (lambda: _fresnel_part("SPAM"), "fresnel"),
    "fresnel_full": (lambda: _fresnel().default_noise_model, "fresnel"),
    # pulse features (noiseless: deterministic, compared trajectory by trajectory), NMR-like local control
    "shaped": (lambda: None, "shaped"),
    "local_dmm": (lambda: None, "local_dmm"),
    "slm_rydberg": (lambda: None, "slm_rydberg"),
    "xy_slm": (lambda: None, "xy_slm"),
    "heisenberg_local": (lambda: None, "heisenberg_local"),
    # the same with Lindblad noise (statistical)
    "local_dmm_noisy": (lambda: pulser.NoiseModel(relaxation_rate=0.5, dephasing_rate=0.5), "local_dmm"),
    "xy_slm_noisy": (lambda: pulser.NoiseModel(dephasing_rate=0.5, depolarizing_rate=0.5), "xy_slm"),
    "heisenberg_local_noisy": (lambda: pulser.NoiseModel(relaxation_rate=0.5, dephasing_rate=0.5), "heisenberg_local"),
}
ANISOTROPY = np.array([1.0, 0.7, 1.3])  # heisenberg_local: [Jxx, Jyy, Jzz] = ANISOTROPY * C6 / r^6 / 20


def _heisenberg(sd):
    """SequenceData -> Heisenberg mode (3 x n x n stack, as examples/emu_mps/heisenberg.ipynb), drive kept."""
    from emu_base import HamiltonianType
    J = torch.as_tensor(ANISOTROPY)[:, None, None] * sd.interaction_matrix(0.0)[0].real.double() / 20
    return dataclasses.replace(sd, interaction_matrix=lambda t: J, hamiltonian_type=HamiltonianType.XY)


REPLACE = {"heisenberg_local": _heisenberg}


def _fresnel():
    return pulser.devices.Device.from_abstract_repr(open(FRESNEL_SPECS).read())


def _fresnel_part(kind):
    nm = _fresnel().default_noise_model
    return {"dephasing": pulser.NoiseModel(dephasing_rate=nm.dephasing_rate),
            "relaxation": pulser.NoiseModel(relaxation_rate=nm.relaxation_rate),
            "SPAM": pulser.NoiseModel(p_false_pos=nm.p_false_pos, p_false_neg=nm.p_false_neg)}[kind]


def make_sequence(device_name):
    """3-atom sequences. mock: chain, constant pulse (quench); fresnel: its layout rules and channel limits, 4 us;
    shaped: Blackman amplitude, detuning ramp, phases; local_dmm: global shaped pulse + local pulses with their own
    phases on q0 then q2 + a DMM detuning map ramp (NMR-like local control); slm_rydberg / xy_slm: SLM mask on q1 for
    the first pulse (XY: mw_global, shaped, phased); heisenberg_local: local_dmm in Heisenberg mode (_heisenberg)."""
    from pulser.register import RegisterLayout
    if device_name in ("local_dmm", "heisenberg_local", "slm_rydberg"):
        reg = RegisterLayout([[7.0 * k, 0] for k in range(8)]).define_register(0, 1, 2, qubit_ids=["q0", "q1", "q2"])
        seq = pulser.Sequence(reg, MockDevice)
        seq.declare_channel("g", "rydberg_global")
        if device_name == "slm_rydberg":
            seq.config_slm_mask(["q1"], "dmm_0")
            seq.add(pulser.Pulse.ConstantPulse(500, 3.0, 1.0, 0.0), "g")
            seq.add(pulser.Pulse.ConstantPulse(500, 3.0, 1.0, 0.0), "g")
            return seq
        seq.declare_channel("loc", "rydberg_local", initial_target="q0")
        seq.config_detuning_map(reg.define_detuning_map({"q0": 1.0, "q1": 0.4, "q2": 0.0}), "dmm_0")
        seq.add(pulser.Pulse(pulser.BlackmanWaveform(600, np.pi), pulser.RampWaveform(600, -2, 2), 0.7), "g")
        seq.add(pulser.Pulse.ConstantPulse(300, 3.0, 1.0, 1.2), "loc")
        seq.target("q2", "loc")
        seq.add(pulser.Pulse.ConstantPulse(300, 2.0, -1.0, 0.3), "loc")
        seq.add_dmm_detuning(pulser.RampWaveform(400, -5, 0), "dmm_0")
        return seq
    if device_name == "xy_slm":
        seq = pulser.Sequence(pulser.Register.rectangle(1, 3, spacing=10.0, prefix="q"), MockDevice)
        seq.declare_channel("mw", "mw_global")
        seq.config_slm_mask(["q1"])
        seq.add(pulser.Pulse(pulser.BlackmanWaveform(500, np.pi), pulser.RampWaveform(500, -1, 1), 0.4), "mw")
        seq.add(pulser.Pulse.ConstantPulse(500, 2.0, 0.5, 1.0), "mw")
        return seq
    if device_name == "shaped":
        reg = pulser.Register.rectangle(1, 3, spacing=8.0, prefix="q")
        seq = pulser.Sequence(reg, MockDevice)
        seq.declare_channel("r", "rydberg_global")
        seq.add(pulser.Pulse(pulser.BlackmanWaveform(700, 1.5 * np.pi), pulser.RampWaveform(700, -3, 3), 0.7), "r")
        seq.add(pulser.Pulse(pulser.InterpolatedWaveform(500, [1, 4, 2, 3]), pulser.ConstantWaveform(500, 1.5), 2.0), "r")
        return seq
    if device_name == "mock":
        reg = pulser.Register.rectangle(1, 3, spacing=8.0, prefix="q")
        seq = pulser.Sequence(reg, MockDevice)
        seq.declare_channel("r", "rydberg_global")
        seq.add(pulser.Pulse.ConstantPulse(1000, 2 * np.pi, 2.0, 0.0), "r")
        return seq
    device = _fresnel()
    reg = pulser.Register.from_coordinates([(0, 0), (6.0, 0), (12.0, 0)], prefix="q").with_automatic_layout(device)
    seq = pulser.Sequence(reg, device)
    seq.declare_channel("r", "rydberg_global")
    seq.add(pulser.Pulse.ConstantPulse(4000, 2 * np.pi, 3.0, 0.0), "r")
    return seq


def emu_mps_trajectories(sds, config):
    """Per trajectory: occupations (N, T, n) and, with readout errors, final bitstring counts (list of Counter)."""
    occ, bits = [], []
    for sd in sds:
        res = MPSBackend._run_from_sequence_data(sd, config)
        occ.append(torch.stack([torch.as_tensor(o) for o in res.occupation]).real)
        if config.noise_model is not None and (config.noise_model.p_false_pos or config.noise_model.p_false_neg):
            bits.append(Counter(res.bitstrings[-1]))
    return torch.stack(occ).double(), bits


def run(name, n_traj, cheb_factor, noiseless_cache={}):
    factory, device_name = CHANNELS[name]
    noise_model = factory()
    seq = make_sequence(device_name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        config = MPSConfig(dt=DT, noise_model=noise_model, n_trajectories=n_traj, optimize_qubit_ordering=False,
                           precision=1e-10, log_level=logging.WARNING,
                           observables=[Occupation(evaluation_times=list(FRACTIONS)),
                                        BitStrings(evaluation_times=[1.0], num_shots=SHOTS)])
    sds = list(PulserData(sequence=seq, config=config, dt=DT).get_sequences())
    if device_name in REPLACE:
        sds = [REPLACE[device_name](sd) for sd in sds]
    # emu-mps cannot run a sample whose state preparation errors leave fewer than 2 atoms: skipped for both codes
    skipped = len(sds)
    sds = [sd for sd in sds if sd.state_prep_error == 0 or sum(not b for b in sd.bad_atoms) >= 2]
    skipped -= len(sds)
    random_jumps = bool(sds[0].lindblad_ops)

    # emu-mps results are cached for channels without shot-to-shot noise (all samples identical): another engine
    # can then be compared without rerunning emu-mps; shot-to-shot channels need the same fresh samples for both
    cache = os.path.join(CACHE, f"{name}_{len(sds)}.pt")
    if not sds[0].has_stochastic_noise and os.path.exists(cache):
        occ_mps, bits, t_mps = torch.load(cache, weights_only=False)
    else:
        t0 = time.perf_counter()
        occ_mps, bits = emu_mps_trajectories(sds, config)
        t_mps = time.perf_counter() - t0
        if not sds[0].has_stochastic_noise:
            os.makedirs(CACHE, exist_ok=True)
            torch.save((occ_mps, bits, t_mps), cache)
    t0 = time.perf_counter()
    if ENGINE == "turbo":
        ours = noisy_cheb_turbo.run_sequence_data(sds * (cheb_factor if random_jumps else 1), FRACTIONS, seed=1,
                                                  single=SINGLE)
    elif ENGINE == "fast":
        ours = noisy_cheb_fast.run_sequence_data(sds * (cheb_factor if random_jumps else 1), FRACTIONS, seed=1,
                                                 single=SINGLE)
    else:
        ours = noisy_cheb.run_sequence_data(sds * (cheb_factor if random_jumps else 1), FRACTIONS, seed=1)
    t_cheb = time.perf_counter() - t0
    occ_cheb = ours.occupations.double()

    line = (f"{name:20s} N_mps {len(sds):4d} ({t_mps / len(sds) * 1e3:7.1f} ms/traj)  N_cheb {len(occ_cheb):5d} "
            f"({t_cheb / len(occ_cheb) * 1e3:6.1f} ms/traj)  ")
    if device_name not in noiseless_cache:
        clean = noisy_cheb.simulate(seq, None, 1, FRACTIONS, dt=DT,
                                    replace_sequence_data=REPLACE.get(device_name)).occupations[0].double()
        noiseless_cache[device_name] = clean
    effect = float((occ_mps.mean(0) - noiseless_cache[device_name]).abs().max())
    if not random_jumps:  # deterministic per sample: trajectory by trajectory
        diff = float((occ_mps - occ_cheb).abs().max())
        line += f"per-trajectory max |d<n>| {diff:.1e}  noise effect {effect:.3f}"
        verdict = diff < 1e-3
    else:
        # standard errors from the per-trajectory spread of a large separate sample (N_SPREAD turbo trajectories):
        # both codes sample the same quantum-jump trajectories, and a sample's own spread misses rare early jumps
        # (e.g. < 1 relaxation among 200 trajectories at 100 ns with FRESNEL's 0.01 / us), which inflated z
        big = noisy_cheb_turbo.run_sequence_data(sds * max(1, N_SPREAD // len(sds)), FRACTIONS, seed=99)
        spread = big.occupations.double().std(0)
        m1, m2 = occ_mps.mean(0), occ_cheb.mean(0)
        se = (spread * (1 / len(occ_mps) + 1 / len(occ_cheb)) ** 0.5).clamp(min=1e-6)
        z = float(((m1 - m2).abs() / se).max())
        line += (f"max |d<n>| {float((m1 - m2).abs().max()):.4f}  max z {z:.1f} ({m1.numel()} values)  "
                 f"noise effect {effect:.3f} = {effect / float(se.max()):.0f} std. err.")
        verdict = z < 4.5
    if bits:
        nm = config.noise_model
        p = noisy_cheb.readout(ours.probabilities[-1], nm.p_false_pos, nm.p_false_neg).numpy()
        n = len(seq.register.qubit_ids)
        counts = np.array([[c.get(format(x, f"0{n}b"), 0) for x in range(2 ** n)] for c in bits], float)  # (N, 2^n)
        if not random_jumps:  # deterministic trajectories: pooled shots are multinomial around mean readout(P)
            pooled = counts.sum(0)
            total = pooled.sum()
            stat = float((((pooled - total * p) ** 2) / np.maximum(total * p, 1e-12)).sum())
            p_value = float(chi2.sf(stat, 2 ** n - 1))
            line += f"  bitstrings chi2 p-value {p_value:.2f} ({int(total)} shots)"
            verdict = verdict and p_value > 1e-3
        else:  # shots of one trajectory share its P: errors from the trajectory-to-trajectory spread of frequencies
            freq = counts / counts.sum(1, keepdims=True)
            se = freq.std(0, ddof=1) / np.sqrt(len(freq)) * np.sqrt(1 + len(freq) / len(occ_cheb))
            zb = float((np.abs(freq.mean(0) - p) / np.maximum(se, 1e-4)).max())
            line += f"  bitstrings max z {zb:.1f} ({2 ** n} values, {len(freq)} trajectories x {SHOTS} shots)"
            verdict = verdict and zb < 4.5
    if skipped:
        line += f"  ({skipped} samples with < 2 good atoms skipped)"
    print(line + ("  OK" if verdict else "  MISMATCH"), flush=True)
    return verdict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("channels", nargs="*", default=list(CHANNELS))
    parser.add_argument("--traj", type=int, default=200)
    parser.add_argument("--cheb-factor", type=int, default=10)
    parser.add_argument("--engine", choices=("reference", "fast", "turbo"), default="reference")
    parser.add_argument("--single", action="store_true", help="fast engine in complex64")
    args = parser.parse_args()
    ENGINE, SINGLE = args.engine, args.single
    warnings.filterwarnings("ignore")
    torch.set_num_threads(1)
    ok = [run(c, args.traj if c != "noiseless" else 1, args.cheb_factor) for c in args.channels]
    print("ALL OK" if all(ok) else "SOME MISMATCH")
