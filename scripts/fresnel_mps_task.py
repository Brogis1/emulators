"""One array task: 100 emu-mps trajectories of a FRESNEL noise case (noisy_vs_emu_mps setup); saves occupations and
final bitstring counts. Usage (scripts dir): python fresnel_mps_task.py CASE TASK_ID OUT_DIR"""
import logging, os, sys, warnings
import torch
warnings.filterwarnings("ignore")
case, task, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
sys.argv = sys.argv[:1]
import noisy_vs_emu_mps as h
from emu_base import PulserData
from emu_mps import BitStrings, MPSConfig, Occupation

path = os.path.join(out, f"{case}_{task:03d}.pt")
if os.path.exists(path):  # idempotent on requeue
    sys.exit(0)
torch.set_num_threads(1)
seq = h.make_sequence("fresnel")
nm = h.CHANNELS[case][0]()
config = MPSConfig(dt=h.DT, noise_model=nm, n_trajectories=100, optimize_qubit_ordering=False, precision=1e-10,
                   log_level=logging.WARNING, observables=[Occupation(evaluation_times=list(h.FRACTIONS)),
                                                           BitStrings(evaluation_times=[1.0], num_shots=h.SHOTS)])
sds = list(PulserData(sequence=seq, config=config, dt=h.DT).get_sequences())
occ, bits = h.emu_mps_trajectories(sds, config)
torch.save((occ, bits), path + ".tmp")
os.replace(path + ".tmp", path)
print(case, task, "done", occ.shape)
