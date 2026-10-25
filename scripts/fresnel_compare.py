"""Three-way FRESNEL check: emu-mps (array results), noisy_cheb_fast, exact dense Lindblad; occupations and, with
readout errors, bitstring frequencies. Usage (scripts dir): python fresnel_compare.py RESULTS_DIR [n_cheb]"""
import glob
import sys
import warnings
from collections import Counter

import numpy as np
import torch

warnings.filterwarnings("ignore")
results_dir = sys.argv[1]
n_cheb = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
sys.argv = sys.argv[:1]
import noisy_cheb as ref  # noqa: E402
import noisy_cheb_fast as fast  # noqa: E402
import noisy_vs_emu_mps as h  # noqa: E402
import rydberg_cheb_fast as ryd  # noqa: E402
from emu_base import PulserData  # noqa: E402
from emu_mps import MPSConfig  # noqa: E402


def z(m1, se1, m2, se2=0.0):
    return float((np.abs(m1 - m2) / np.maximum(np.sqrt(se1 ** 2 + se2 ** 2), 1e-4)).max())


all_ok = True
for case in ("fresnel_relaxation", "fresnel_full"):
    files = sorted(glob.glob(f"{results_dir}/{case}_*.pt"))
    if not files:
        continue
    parts = [torch.load(f, weights_only=False) for f in files]
    occ_mps = torch.cat([p[0] for p in parts]).double().numpy()
    bits_mps = [c for p in parts for c in p[1]]
    seq = h.make_sequence("fresnel")
    nm = h.CHANNELS[case][0]()
    sds = list(PulserData(sequence=seq, config=MPSConfig(dt=h.DT, noise_model=nm, n_trajectories=n_cheb,
                                                         optimize_qubit_ordering=False), dt=h.DT).get_sequences())
    ours = fast.run_sequence_data(sds, h.FRACTIONS, seed=11)
    occ_ours = ours.occupations.double().numpy()
    model = ref._Model(sds[0], torch.device("cpu"))
    out = model.output_times(h.FRACTIONS)
    rho_diag = torch.as_tensor(ref._dense_lindblad(model, out))
    n = model.n_total
    occ_exact = (rho_diag @ ryd.moments(torch.eye(2 ** n, dtype=torch.float64), n)[0]).numpy()
    clean = ref.simulate(seq, None, 1, h.FRACTIONS, dt=h.DT).occupations[0].double().numpy()
    m1, s1 = occ_mps.mean(0), occ_mps.std(0, ddof=1) / np.sqrt(len(occ_mps))
    m2, s2 = occ_ours.mean(0), occ_ours.std(0, ddof=1) / np.sqrt(len(occ_ours))
    effect = float(np.abs(occ_exact - clean).max())
    print(f"{case}: emu-mps {len(occ_mps)} traj, noisy_cheb_fast {len(occ_ours)} traj, noise effect (exact - noiseless) "
          f"{effect:.4f} = {effect / s1.max():.1f} emu-mps std. err.")
    zs = dict(mps_exact=z(m1, s1, occ_exact), ours_exact=z(m2, s2, occ_exact), mps_ours=z(m1, s1, m2, s2))
    print(f"    <n_i>(t), max z over {m1.size} values: emu-mps vs exact {zs['mps_exact']:.1f}, ours vs exact "
          f"{zs['ours_exact']:.1f}, emu-mps vs ours {zs['mps_ours']:.1f}; max |emu-mps - ours| {np.abs(m1 - m2).max():.4f}")
    ok = max(zs.values()) < 4.5
    if bits_mps:
        p_exact = ref.readout(rho_diag[-1], nm.p_false_pos, nm.p_false_neg).numpy()
        p_ours = ref.readout(ours.probabilities[-1], nm.p_false_pos, nm.p_false_neg).numpy()
        counts = np.array([[c.get(format(x, f"0{n}b"), 0) for x in range(2 ** n)] for c in bits_mps], float)
        freq = counts / counts.sum(1, keepdims=True)
        f_mean, f_se = freq.mean(0), freq.std(0, ddof=1) / np.sqrt(len(freq))
        zb = dict(mps_exact=z(f_mean, f_se, p_exact), ours_exact=float(np.abs(p_ours - p_exact).max()))
        print(f"    bitstrings (readout errors): emu-mps vs exact max z {zb['mps_exact']:.1f} over {2 ** n} values; "
              f"max |ours - exact| {zb['ours_exact']:.4f}; readout shifts P by {np.abs(p_exact - rho_diag[-1].numpy()).max():.3f}")
        ok = ok and zb["mps_exact"] < 4.5
    print("    OK" if ok else "    MISMATCH")
    all_ok = all_ok and ok
print("ALL OK" if all_ok else "SOME MISMATCH")
