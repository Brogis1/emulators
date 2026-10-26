"""emu_cheb's wiring against emu-sv and dense references. The engines themselves are validated (emu_cheb/solvers/README.md):
these tests stay on N <= 4 atoms (CPU, seconds)."""
import logging
from functools import reduce

import numpy as np
import pulser
import pytest
import torch
from pulser.backend import AggregationMethod, Observable
from pulser.devices import MockDevice

from emu_base import PulserData
from emu_cheb import (
    BitStrings,
    ChebBackend,
    ChebConfig,
    CorrelationMatrix,
    DenseOperator,
    Energy,
    EnergyVariance,
    Expectation,
    Fidelity,
    Occupation,
    StateResult,
    StateVector,
    connected,
)
from emu_cheb.solvers import noisy_cheb
from emu_sv import SVBackend, SVConfig, Solver

pytestmark = pytest.mark.filterwarnings("ignore")
QUIET = dict(gpu=False, log_level=logging.WARN)
TIMES = [0.0, 0.25, 0.5, 0.75, 1.0]
N = 4
NEEL = StateVector.from_state_amplitudes(eigenstates=("r", "g"), amplitudes={"rgrg": 1.0})

X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
Y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128)
Z = torch.diag(torch.tensor([1.0, -1.0], dtype=torch.complex128))
NUM = torch.diag(torch.tensor([0.0, 1.0], dtype=torch.complex128))


def op(single, i, n=N):
    """single on atom i (atom 0 = most significant bit, index 0 = ground), identity elsewhere."""
    return reduce(torch.kron, [single if k == i else torch.eye(2, dtype=torch.complex128) for k in range(n)])


def rydberg_sequence(n=N, shaped=True):
    reg = pulser.Register.rectangle(1, n, spacing=7.0, prefix="q")
    seq = pulser.Sequence(reg, MockDevice)
    seq.declare_channel("ch", "rydberg_global")
    if shaped:
        seq.add(pulser.Pulse(pulser.BlackmanWaveform(500, np.pi), pulser.RampWaveform(500, -5.0, 5.0), 0.0), "ch")
    else:
        seq.add(pulser.Pulse.ConstantPulse(500, 3.0, 1.0, 0.0), "ch")
    return seq


def xy_sequence(n=N, shaped=True):
    """Microwave channel, as examples/emu_mps/heisenberg.ipynb."""
    reg = pulser.Register.rectangle(1, n, spacing=10.0, prefix="q")
    seq = pulser.Sequence(reg, MockDevice)
    seq.declare_channel("mw", "mw_global")
    if shaped:
        seq.add(pulser.Pulse(pulser.BlackmanWaveform(500, np.pi), pulser.RampWaveform(500, -3.0, 3.0), 0.0), "mw")
    else:
        seq.add(pulser.Pulse.ConstantPulse(500, 2.0, 0.5, 0.0), "mw")
    return seq


def stacked(results, tag):
    return torch.stack([torch.as_tensor(v) for v in getattr(results, tag)])


# ---------------------------------------------------------------- diagonal observables, engines


def test_noiseless_rydberg_matches_emu_sv():
    seq = rydberg_sequence()

    def observables():
        return [Occupation(evaluation_times=TIMES), CorrelationMatrix(evaluation_times=TIMES)]

    backend = ChebBackend(seq, config=ChebConfig(observables=observables(), initial_state=NEEL, **QUIET))
    cheb = backend.run()
    sv_config = SVConfig(observables=observables(), initial_state=NEEL, krylov_tolerance=1e-12, **QUIET)
    sv = SVBackend(seq, config=sv_config).run()
    assert backend.raw.stats["optimization"] == "turbo"  # shaped pulse: not a quench
    for tag in ("occupation", "correlation_matrix"):
        assert cheb.get_result_times(tag) == sv.get_result_times(tag)
        assert torch.allclose(stacked(cheb, tag), stacked(sv, tag), atol=1e-8), tag
    occ, corr = cheb.occupation[-1], cheb.correlation_matrix[-1]
    assert torch.allclose(connected(occ, corr), corr - occ[:, None] * occ[None, :])


def dense_heisenberg_occupations(sd, J3):
    """<n_i> at every grid time: H = drive + sum_{i<j} Jxx XX + Jyy YY + Jzz ZZ (Pauli matrices, independent of the
    engines), one dense exponential per drive step."""
    pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    H_int = sum(J3[a, i, j] * op(P, i) @ op(P, j) for a, P in enumerate((X, Y, Z)) for i, j in pairs)
    psi = torch.zeros(2**N, dtype=torch.complex128)
    psi[0] = 1.0
    occ = [torch.stack([torch.vdot(psi, op(NUM, q) @ psi).real for q in range(N)])]
    t = sd.target_times
    for k in range(len(t) - 1):
        w, d, p = sd.omega[k].real, sd.delta[k].real, sd.phi[k].real
        H = H_int + sum(
            w[q] / 2 * (torch.cos(p[q]) * op(X, q) + torch.sin(p[q]) * op(Y, q)) - d[q] * op(NUM, q) for q in range(N)
        )
        psi = torch.linalg.matrix_exp(-1j * H * (t[k + 1] - t[k]) / 1000) @ psi
        occ.append(torch.stack([torch.vdot(psi, op(NUM, q) @ psi).real for q in range(N)]))
    return torch.stack(occ)


@pytest.mark.parametrize("kind", ["stack", "anisotropy"])
def test_heisenberg_matches_dense(kind):
    seq = xy_sequence()
    plain = ChebConfig(observables=[Occupation(evaluation_times=TIMES)], **QUIET)
    (sd,) = PulserData(sequence=seq, config=plain, dt=10).get_sequences()
    if kind == "stack":  # any graph: random symmetric couplings, zero diagonal
        J3 = torch.as_tensor(np.random.default_rng(0).uniform(-2, 2, (3, N, N)))
        J3 = (J3 + J3.transpose(1, 2)) / 2
        J3 = J3 - torch.diag_embed(torch.diagonal(J3, dim1=1, dim2=2))
        couplings = J3.numpy()
    else:  # multiples of the sequence's own C3 / r^3 matrix
        couplings = (1.0, 0.5, -0.3)
        J3 = torch.tensor(couplings, dtype=torch.float64)[:, None, None] * sd.interaction_matrix(0.0)[0].real.double()
    config = ChebConfig(observables=[Occupation(evaluation_times=TIMES)], heisenberg=couplings, **QUIET)
    res = ChebBackend(seq, config=config).run()
    grid = np.asarray(sd.target_times) / sd.target_times[-1]
    rows = [int(np.argmin(np.abs(grid - t))) for t in TIMES]
    assert torch.allclose(stacked(res, "occupation"), dense_heisenberg_occupations(sd, J3)[rows], atol=1e-8)


def test_noisy_matches_emu_sv_lindblad():
    seq = rydberg_sequence(n=3)
    noise = pulser.NoiseModel(dephasing_rate=0.5, depolarizing_rate=0.2, p_false_pos=0.1, p_false_neg=0.05)
    times, n_traj, shots = [0.0, 0.5, 1.0], 2000, 10
    observables = [Occupation(evaluation_times=times), BitStrings(evaluation_times=[1.0], num_shots=shots)]
    backend = ChebBackend(seq, config=ChebConfig(observables=observables, noise_model=noise, n_trajectories=n_traj, **QUIET))
    cheb = backend.run()
    sv_observables = [Occupation(evaluation_times=times), StateResult(evaluation_times=[1.0])]
    sv_config = SVConfig(observables=sv_observables, noise_model=noise, solver=Solver.LINDBLAD, **QUIET)
    sv = SVBackend(seq, config=sv_config).run()
    bound = 5 * 0.5 / n_traj**0.5  # per-trajectory occupations and probabilities lie in [0, 1]: std <= 0.5
    assert (stacked(cheb, "occupation") - stacked(sv, "occupation")).abs().max() < bound
    assert backend.raw.occupations.shape == (n_traj, len(times), 3)  # error bars stay available
    counts = cheb.final_bitstrings
    assert sum(counts.values()) == n_traj * shots  # as emu-sv's bag union of per-trajectory samples
    exact = noisy_cheb.readout(sv.state[-1].data.diagonal().real.double(), 0.1, 0.05)
    freq = torch.tensor([counts.get(format(x, "03b"), 0) for x in range(8)], dtype=torch.float64) / (n_traj * shots)
    assert (freq - exact).abs().max() < bound + 5 * 0.5 / (n_traj * shots) ** 0.5


def test_engines_agree():
    """fast and turbo give each trajectory its own random stream (SeedSequence(seed).spawn): same seed, same
    trajectories. The reference draws all trajectories from one stream: it agrees statistically."""
    seq = rydberg_sequence(n=3)
    noise = pulser.NoiseModel(dephasing_rate=0.5, relaxation_rate=0.3)
    occ = {}
    for opt, n_traj in (("reference", 200), ("fast", 200), ("turbo", 200)):
        config = ChebConfig(
            observables=[Occupation(evaluation_times=TIMES)], noise_model=noise, n_trajectories=n_traj, seed=7,
            optimization=opt, **QUIET,
        )
        backend = ChebBackend(seq, config=config)
        occ[opt] = stacked(backend.run(), "occupation")
        assert backend.raw.stats["optimization"] == opt
    assert torch.allclose(occ["fast"], occ["turbo"], atol=1e-9)
    assert (occ["reference"] - occ["fast"]).abs().max() < 5 * 0.5 * (2 / 200) ** 0.5  # two independent means


@pytest.mark.parametrize("model", ["rydberg", "heisenberg"])
def test_auto_picks_quench_and_matches_turbo(model):
    seq = rydberg_sequence(shaped=False) if model == "rydberg" else xy_sequence(shaped=False)
    extra = {} if model == "rydberg" else {"heisenberg": (1.0, 0.5, -0.3)}

    def observables():
        return [Occupation(evaluation_times=TIMES), CorrelationMatrix(evaluation_times=TIMES)]

    auto = ChebBackend(seq, config=ChebConfig(observables=observables(), **extra, **QUIET))
    res_auto = auto.run()
    assert auto.raw.stats["optimization"] == "quench"
    turbo_config = ChebConfig(observables=observables(), optimization="turbo", **extra, **QUIET)
    res_turbo = ChebBackend(seq, config=turbo_config).run()
    for tag in ("occupation", "correlation_matrix"):
        assert torch.allclose(stacked(res_auto, tag), stacked(res_turbo, tag), atol=1e-8), tag


def test_quench_refuses_shaped_pulse():
    config = ChebConfig(observables=[Occupation(evaluation_times=[1.0])], optimization="quench", **QUIET)
    with pytest.raises(ValueError, match="time-dependent"):
        ChebBackend(rydberg_sequence(), config=config).run()


# ---------------------------------------------------------------- off-diagonal observables


def off_diagonal_observables(times):
    x0 = DenseOperator.from_operator_repr(
        eigenstates=("r", "g"), n_qudits=N, operations=[(1.0, [({"rg": 1.0, "gr": 1.0}, {0})])]
    )
    return [
        Energy(evaluation_times=times),
        EnergyVariance(evaluation_times=times),
        Expectation(x0, evaluation_times=times),
        Fidelity(NEEL, evaluation_times=times),
        StateResult(evaluation_times=[1.0]),
    ]


@pytest.mark.parametrize("shaped", [True, False])  # turbo path, quench path
def test_off_diagonal_matches_emu_sv(shaped):
    seq = rydberg_sequence(shaped=shaped)
    initial = NEEL if shaped else None  # the quench starts from the ground state
    backend = ChebBackend(seq, config=ChebConfig(observables=off_diagonal_observables(TIMES), initial_state=initial, **QUIET))
    cheb = backend.run()
    assert backend.raw.stats["optimization"] == ("turbo" if shaped else "quench")
    sv_config = SVConfig(
        observables=off_diagonal_observables(TIMES), initial_state=initial, krylov_tolerance=1e-12, **QUIET
    )
    sv = SVBackend(seq, config=sv_config).run()
    for tag in ("energy", "energy_variance", "expectation", "fidelity"):
        assert torch.allclose(stacked(cheb, tag), stacked(sv, tag), atol=1e-8), tag
    assert abs(float(torch.vdot(cheb.state[-1].data, sv.state[-1].data).abs()) - 1) < 1e-8


@pytest.mark.parametrize("noisy", [True, False])  # turbo path (several trajectories), quench path
def test_off_diagonal_only_at_their_own_times(noisy):
    """Occupation at 5 times, Energy at the end only, Expectation at 2 times: the off-diagonal observables are
    evaluated (and stored) only at their own times, once per trajectory."""
    x0 = DenseOperator.from_operator_repr(
        eigenstates=("r", "g"), n_qudits=N, operations=[(1.0, [({"rg": 1.0, "gr": 1.0}, {0})])]
    )

    def observables():
        return [
            Occupation(evaluation_times=TIMES),
            Energy(evaluation_times=[1.0]),
            Expectation(x0, evaluation_times=[0.5, 1.0]),
        ]

    n_traj = 10 if noisy else 1
    noise = pulser.NoiseModel(dephasing_rate=0.5) if noisy else None
    seq = rydberg_sequence(shaped=False)
    backend = ChebBackend(seq, config=ChebConfig(observables=observables(), noise_model=noise, n_trajectories=n_traj, **QUIET))
    cheb = backend.run()
    assert backend.raw.stats["optimization"] == ("turbo" if noisy else "quench")
    assert backend.raw.stats["off_diagonal_evaluations"] == n_traj * (1 + 2)
    assert cheb.get_result_times("energy") == [1.0]
    assert cheb.get_result_times("expectation") == [0.5, 1.0]
    assert cheb.get_result_times("occupation") == TIMES
    if not noisy:
        sv = SVBackend(seq, config=SVConfig(observables=observables(), krylov_tolerance=1e-12, **QUIET)).run()
        for tag in ("energy", "expectation"):
            assert torch.allclose(stacked(cheb, tag), stacked(sv, tag), atol=1e-8), tag


def test_expectation_under_noise_matches_lindblad():
    n, times, n_traj = 3, [0.5, 1.0], 1000
    seq = rydberg_sequence(n=n)
    noise = pulser.NoiseModel(dephasing_rate=0.5, relaxation_rate=0.3)
    x0 = DenseOperator.from_operator_repr(
        eigenstates=("r", "g"), n_qudits=n, operations=[(1.0, [({"rg": 1.0, "gr": 1.0}, {0})])]
    )
    config = ChebConfig(observables=[Expectation(x0, evaluation_times=times)], noise_model=noise, n_trajectories=n_traj, **QUIET)
    cheb = ChebBackend(seq, config=config).run()
    sv_config = SVConfig(observables=[StateResult(evaluation_times=times)], noise_model=noise, solver=Solver.LINDBLAD, **QUIET)
    rhos = SVBackend(seq, config=sv_config).run().state
    exact = torch.stack([torch.trace(rho.data.cpu() @ op(X, 0, n)).real for rho in rhos])
    assert (stacked(cheb, "expectation").real - exact).abs().max() < 5 / n_traj**0.5  # |<X_0>| <= 1 per trajectory


# ---------------------------------------------------------------- errors


class Purity(Observable):
    def __init__(self, **kwargs):
        super().__init__(default_aggregation_method=AggregationMethod.MEAN, **kwargs)

    @property
    def _base_tag(self):
        return "purity"

    def apply(self, **kwargs):
        return 1.0


def test_config_errors():
    with pytest.raises(ValueError):
        ChebConfig(optimization="warp")
    with pytest.raises(ValueError):
        ChebConfig(bounds="exact")
    with pytest.raises(ValueError):
        ChebConfig(optimization="reference", single=True)
    with pytest.raises(ValueError):
        ChebConfig(heisenberg=np.ones((2, 4, 4)))
    with pytest.raises(NotImplementedError):
        ChebConfig(observables=[Purity(evaluation_times=[1.0])])
    with pytest.raises(ValueError, match="single trajectory"):
        ChebConfig(observables=[StateResult(evaluation_times=[1.0])], n_trajectories=10)


def test_non_basis_initial_state_raises():
    plus = StateVector.from_state_amplitudes(eigenstates=("r", "g"), amplitudes={"gggg": 0.6, "rggg": 0.8})
    config = ChebConfig(observables=[Occupation(evaluation_times=[1.0])], initial_state=plus, **QUIET)
    with pytest.raises(NotImplementedError, match="basis state"):
        ChebBackend(rydberg_sequence(), config=config).run()
