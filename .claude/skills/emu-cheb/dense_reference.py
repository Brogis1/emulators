"""Exact dense references for emu_cheb / emu_mps runs, n <= ~10 (state vector) or ~6 (Lindblad).

Driven by the same PulserData sample `sd` as the emulators (omega/delta/phi constant on each dt step), so the
comparison is exact for the discretised drive. Conventions: atom 0 = most significant bit, index 1 = excited,
n = |1><1|. Pass your OWN interaction (heisenberg_int / xy_int): sd.interaction_matrix is pulser's matrix, not a
custom J or Heisenberg stack you passed to the backend.

    from dense_reference import xy_int, heisenberg_int, state_dynamics, lindblad_dynamics
    (sd,) = PulserData(sequence=seq, config=ChebConfig(dt=dt, noise_model=noise), dt=dt).get_sequences()
    occ = state_dynamics(sd, heisenberg_int(stack), "01010")      # (steps + 1, n): <n_i> on the dt grid
    occ = lindblad_dynamics(sd, heisenberg_int(stack), "01010")   # uses sd.lindblad_ops (noise_model above)
"""
from functools import reduce

import torch

dtype = torch.complex128
X = torch.tensor([[0, 1], [1, 0]], dtype=dtype)
Y = torch.tensor([[0, -1j], [1j, 0]], dtype=dtype)
Z = torch.tensor([[1, 0], [0, -1]], dtype=dtype)
N1 = torch.tensor([[0, 0], [0, 1]], dtype=dtype)  # |1><1|
P0 = torch.tensor([[1, 0], [0, 0]], dtype=dtype)  # |0><0| (|u><u| in pulser's XY basis)


def op(single, i, n):
    return reduce(torch.kron, [single if k == i else torch.eye(2, dtype=dtype) for k in range(n)])


def pair_sum(M, A, B, n):
    M = torch.as_tensor(M, dtype=torch.float64)
    return sum(M[i, j] * op(A, i, n) @ op(B, j, n) for i in range(n) for j in range(i + 1, n))


def heisenberg_int(stack):
    """sum_{i<j} Jxx XX + Jyy YY + Jzz ZZ for a (3, n, n) stack."""
    n = len(stack[0])
    return sum(pair_sum(Jk, P, P, n) for Jk, P in zip(stack, (X, Y, Z)))


def xy_int(J, C6=None):
    """pulser's XY stack [J, C6]: sum_{i<j} J/2 (XX + YY) + C6 |00><00|."""
    n = len(J)
    H = pair_sum(J, X, X, n) / 2 + pair_sum(J, Y, Y, n) / 2
    return H if C6 is None else H + pair_sum(C6, P0, P0, n)


def _hamiltonians(sd, H_int):
    n = H_int.shape[0].bit_length() - 1
    w, d, p = sd.omega.real.cpu(), sd.delta.real.cpu(), sd.phi.real.cpu()
    for k in range(len(w)):
        yield H_int + sum(w[k, j] / 2 * (torch.cos(p[k, j]) * op(X, j, n) + torch.sin(p[k, j]) * op(Y, j, n))
                          - d[k, j] * op(N1, j, n) for j in range(n))


def state_dynamics(sd, H_int, bits):
    """<n_i> at every dt-grid time, from the basis state `bits` ("1" = excited, atom 0 first)."""
    n = len(bits)
    Ns = [op(N1, j, n) for j in range(n)]
    psi = torch.zeros(2**n, dtype=dtype)
    psi[int(bits, 2)] = 1
    occ = [torch.stack([torch.vdot(psi, Nj @ psi).real for Nj in Ns])]
    for H, t0, t1 in zip(_hamiltonians(sd, H_int), sd.target_times, sd.target_times[1:]):
        psi = torch.linalg.matrix_exp(-1j * H * (t1 - t0) / 1000) @ psi  # ns -> us
        occ.append(torch.stack([torch.vdot(psi, Nj @ psi).real for Nj in Ns]))
    return torch.stack(occ)


def lindblad_dynamics(sd, H_int, bits, n_sub=10):
    """<n_i> at every dt-grid time from the exact Lindblad equation (RK4, n_sub substeps per dt), sd.lindblad_ops."""
    n = len(bits)
    Ns = [op(N1, j, n) for j in range(n)]
    Ls = [op(L.cpu(), j, n) for L in sd.lindblad_ops for j in range(n)]
    LdL = sum((L.mH @ L for L in Ls), torch.zeros(2**n, 2**n, dtype=dtype))

    def rhs(rho, H_eff):
        return -1j * (H_eff @ rho - rho @ H_eff.mH) + sum((L @ rho @ L.mH for L in Ls), torch.zeros_like(rho))

    rho = torch.zeros(2**n, 2**n, dtype=dtype)
    rho[int(bits, 2), int(bits, 2)] = 1
    occ = [torch.stack([torch.trace(Nj @ rho).real for Nj in Ns])]
    for H, t0, t1 in zip(_hamiltonians(sd, H_int), sd.target_times, sd.target_times[1:]):
        H_eff, h = H - 0.5j * LdL, (t1 - t0) / 1000 / n_sub
        for _ in range(n_sub):
            k1 = rhs(rho, H_eff)
            k2 = rhs(rho + h / 2 * k1, H_eff)
            k3 = rhs(rho + h / 2 * k2, H_eff)
            k4 = rhs(rho + h * k3, H_eff)
            rho = rho + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        occ.append(torch.stack([torch.trace(Nj @ rho).real for Nj in Ns]))
    return torch.stack(occ)


if __name__ == "__main__":  # self-check against emu_cheb: XY via pulser's stack and XXZ via heisenberg=, + noise
    import numpy as np
    import pulser
    from emu_base import PulserData
    from emu_cheb import ChebBackend, ChebConfig, Occupation, StateVector

    n, T, dt, bits = 4, 600, 10, "0101"
    J = np.array([[0, 1.0, 0.5, 0], [1.0, 0, 0.8, 0], [0.5, 0.8, 0, 1.2], [0, 0, 1.2, 0]])
    seq = pulser.Sequence(pulser.Register.rectangle(1, n, spacing=10.0, prefix="q"), pulser.devices.MockDevice)
    seq.declare_channel("mw", "mw_global")
    seq.add(pulser.Pulse(pulser.BlackmanWaveform(T, np.pi), pulser.RampWaveform(T, -3.0, 3.0), 0.0), "mw")
    psi0 = torch.zeros(2**n, dtype=dtype)
    psi0[int(bits, 2)] = 1
    times = [k / 20 for k in range(21)]
    rows = [round(t * T / dt) for t in times]

    def cheb(noise=None, n_traj=1, **extra):
        config = ChebConfig(observables=[Occupation(evaluation_times=times)], initial_state=StateVector(psi0, gpu=False),
                            noise_model=noise, n_trajectories=n_traj, dt=dt, **extra)
        backend = ChebBackend(seq, config=config)
        return torch.stack(backend.run().occupation), backend

    noise = pulser.NoiseModel(dephasing_rate=0.2, depolarizing_rate=0.1)
    (sd,) = PulserData(sequence=seq, config=ChebConfig(dt=dt, noise_model=noise), dt=dt).get_sequences()
    stack = np.stack([J / 2, J / 2, 0.3 * J])
    err_xy = (cheb(interaction_matrix=np.stack([J, 0 * J]))[0] - state_dynamics(sd, xy_int(J), bits)[rows]).abs().max()
    err_xxz = (cheb(heisenberg=stack)[0] - state_dynamics(sd, heisenberg_int(stack), bits)[rows]).abs().max()
    occ, backend = cheb(noise, 2000, heisenberg=stack)
    se = (backend.raw.occupations.std(0) ** 2 + 1 / 2000) ** 0.5 / 2000**0.5  # floored: rare jumps at early times
    z = ((occ - lindblad_dynamics(sd, heisenberg_int(stack), bits)[rows]) / se).abs().max()
    print(f"XY {err_xy:.1e}, XXZ {err_xxz:.1e}, noisy XXZ max |z| {z:.2f}")
    assert err_xy < 1e-10 and err_xxz < 1e-10 and z < 4.5
