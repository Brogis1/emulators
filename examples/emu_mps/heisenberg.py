# Heisenberg model in emu-mps, bypassing pulser via SequenceData.
#
# H(t) = ∑ⱼ Ωⱼ/2 [cos(ϕⱼ)Xⱼ + sin(ϕⱼ)Yⱼ] - ∑ⱼ Δⱼnⱼ
#        + ∑ᵢ﹤ⱼ Jxx[i,j]XᵢXⱼ + Jyy[i,j]YᵢYⱼ + Jzz[i,j]ZᵢZⱼ
#
# X, Y, Z are Pauli matrices, n = (1 - Z)/2, Ω and Δ in rad/µs, times in ns.
# omega[k], delta[k], phi[k] are held constant on [target_times[k], target_times[k+1]].
# The result is checked against exact (dense) piecewise-constant evolution.
from functools import reduce

import torch

from emu_base import HamiltonianType, SequenceData
from emu_mps import MPSBackend, MPSConfig, StateResult

dtype = torch.complex128
torch.manual_seed(0)

n = 5
dt = 10  # ns
nsteps = 30
target_times = [dt * k for k in range(nsteps + 1)]

# Couplings in rad/µs: isotropic J(XX+YY+ZZ) is Jxx = Jyy = Jzz.
# Only the upper triangle (i < j) is used; keep them symmetric anyway.
nn_chain = torch.diag(torch.ones(n - 1, dtype=torch.float64), 1)
nn_chain = nn_chain + nn_chain.T
J = torch.stack([1.0 * nn_chain, 0.7 * nn_chain, 1.3 * nn_chain])  # [Jxx, Jyy, Jzz]

# discretized drive, shape (nsteps, n)
t_mid = torch.tensor(target_times[:-1], dtype=torch.float64) + dt / 2
omega = (2.0 * torch.sin(torch.pi * t_mid / target_times[-1]))[:, None].repeat(1, n)
delta = torch.linspace(-1.0, 1.0, nsteps, dtype=torch.float64)[:, None].repeat(1, n)
phi = torch.zeros(nsteps, n, dtype=torch.float64)

seq_data = SequenceData(
    omega=omega.to(dtype),
    delta=delta.to(dtype),
    phi=phi.to(dtype),
    interaction_matrix=lambda t: J,  # 3 x n x n => Heisenberg
    qubit_ids=tuple(f"q{i}" for i in range(n)),
    bad_atoms=(False,) * n,
    lindblad_ops=[],  # noiseless
    state_prep_error=0.0,
    target_times=target_times,
    eigenstates=["0", "1"],
    hamiltonian_type=HamiltonianType.XY,
    has_stochastic_noise=False,
)

config = MPSConfig(
    observables=[StateResult(evaluation_times=[1.0])],
    optimize_qubit_ordering=False,
    precision=1e-10,
)
results = MPSBackend._run_from_sequence_data(seq_data, config)
mps = results.state[-1]
psi_mps = reduce(lambda a, b: torch.tensordot(a, b, dims=1), mps.factors).reshape(-1)


# exact reference
def op(single: torch.Tensor, i: int) -> torch.Tensor:
    return reduce(
        torch.kron, [single if k == i else torch.eye(2, dtype=dtype) for k in range(n)]
    )


X = torch.tensor([[0, 1], [1, 0]], dtype=dtype)
Y = torch.tensor([[0, -1j], [1j, 0]], dtype=dtype)
Z = torch.tensor([[1, 0], [0, -1]], dtype=dtype)
N = torch.tensor([[0, 0], [0, 1]], dtype=dtype)

H_int = sum(
    J[a, i, j] * op(P, i) @ op(P, j)
    for a, P in enumerate((X, Y, Z))
    for i in range(n)
    for j in range(i + 1, n)
)
psi = torch.zeros(2**n, dtype=dtype)
psi[0] = 1.0
for k in range(nsteps):
    H = H_int + sum(
        omega[k, j]
        / 2
        * (torch.cos(phi[k, j]) * op(X, j) + torch.sin(phi[k, j]) * op(Y, j))
        - delta[k, j] * op(N, j)
        for j in range(n)
    )
    psi = torch.linalg.matrix_exp(-1j * H * dt / 1000) @ psi

fidelity = abs(torch.vdot(psi, psi_mps.cpu())) ** 2
print(f"fidelity(emu-mps, exact) = {fidelity:.12f}")
assert fidelity > 1 - 1e-6
