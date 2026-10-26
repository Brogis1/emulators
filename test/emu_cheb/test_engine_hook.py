"""The opt-in on_state hook of the engines in emu_cheb.solvers (the contract emu_cheb relies on). CPU, seconds."""
import numpy as np
import pytest
import torch

from emu_cheb.solvers import heisenberg_cheb_fast, noisy_cheb, noisy_cheb_fast, noisy_cheb_turbo, rydberg_cheb_fast

pytestmark = pytest.mark.filterwarnings("ignore")
ENGINES = {"reference": noisy_cheb, "fast": noisy_cheb_fast, "turbo": noisy_cheb_turbo}
FRACTIONS = (0.0, 0.25, 0.5, 1.0)
N = 3


def _sequences(n_traj):
    """n_traj samples of one SequenceData: piecewise-constant drive, dephasing and relaxation (non-Hermitian part)."""
    from emu_base import HamiltonianType
    rng = np.random.default_rng(1)
    omega, delta, phi = rng.uniform(2, 6, (8, N)), rng.uniform(-3, 3, (8, N)), rng.uniform(0, 1, (8, N))
    J = np.array([[0, 4.0, 0.5], [4.0, 0, 4.0], [0.5, 4.0, 0]])
    dephasing = np.sqrt(0.5) * np.diag([1.0, -1.0])
    relaxation = np.sqrt(0.3) * np.array([[0, 1.0], [0, 0]])  # |g><r| (emu basis: index 0 = ground)
    sd = noisy_cheb._fake_sequence_data(N, omega, delta, phi, J, [dephasing, relaxation], HamiltonianType.Rydberg)
    return [sd] * n_traj


@pytest.mark.parametrize("name", ENGINES)
def test_hook_sees_every_state_of_every_trajectory(name):
    n_traj, seen = 6, {}

    def hook(j, traj_ids, psi, active):
        assert psi.shape == (len(traj_ids), 2 ** int(active.sum()))
        for row, k in enumerate(traj_ids):
            p = psi[row].abs().square().double()
            full = torch.zeros(2 ** N, dtype=torch.float64)
            full[noisy_cheb._embedding(active, torch.device("cpu"))] = p / p.sum()
            seen[(j, k)] = full

    res = ENGINES[name].run_sequence_data(_sequences(n_traj), FRACTIONS, seed=3, on_state=hook)
    assert set(seen) == {(j, k) for j in range(len(FRACTIONS)) for k in range(n_traj)}
    for (j, k), full in seen.items():  # the state handed over is trajectory k's (its occupations match)
        occ = rydberg_cheb_fast.moments(full[None], N)[0][0]
        assert torch.allclose(occ, res.occupations[k, j], atol=1e-12)
    for j in range(len(FRACTIONS)):
        mean = torch.stack([seen[(j, k)] for k in range(n_traj)]).mean(0)
        assert torch.allclose(mean, res.probabilities[j], atol=1e-12)
    assert res.stats["jumps"] > 0  # the trajectories did jump: the hook saw post-jump states too


@pytest.mark.parametrize("name", ENGINES)
def test_noop_hook_changes_nothing(name):
    sds = _sequences(4)
    a = ENGINES[name].run_sequence_data(sds, FRACTIONS, seed=5)
    b = ENGINES[name].run_sequence_data(sds, FRACTIONS, seed=5, on_state=lambda *args: None)
    assert torch.equal(a.probabilities, b.probabilities)
    assert torch.equal(a.occupations, b.occupations)


@pytest.mark.parametrize("solver", ["rydberg", "heisenberg"])
def test_quench_hook_gives_psi_without_returning_states(solver):
    """rydberg_cheb_fast / heisenberg_cheb_fast: on_state sees psi(t_j) of every realization while output stays
    "probabilities" (|psi|^2 equals the returned probabilities); a no-op hook changes nothing."""
    rng = np.random.default_rng(2)
    R, n, times = 2, 4, np.array([0.0, 0.3, 0.7, 1.1])
    J = rng.uniform(0, 5, (R, n, n))
    J = (J + np.swapaxes(J, 1, 2)) / 2 * (1 - np.eye(n))
    if solver == "heisenberg":
        J = np.stack([J, 0.5 * J, -0.3 * J], 1)  # (R, 3, n, n)
    module = rydberg_cheb_fast if solver == "rydberg" else heisenberg_cheb_fast
    omega, delta = rng.uniform(1, 4, (R, n)), rng.uniform(-2, 2, (R, n))
    seen = {}
    P = module.evolve(J, omega, delta, times, device="cpu", on_state=lambda j, psi: seen.__setitem__(j, psi.clone()))
    assert P.dtype == torch.float64 and sorted(seen) == list(range(len(times)))
    for j, psi in seen.items():
        assert psi.shape == (R, 2**n) and torch.allclose(psi.abs().square(), P[j], atol=1e-14)
    states = module.evolve(J, omega, delta, times, device="cpu", output="state")
    assert all(torch.allclose(seen[j], states[j], atol=1e-14) for j in seen)  # the same amplitudes, phase included
    assert torch.equal(P, module.evolve(J, omega, delta, times, device="cpu", on_state=lambda *args: None))
