"""ChebConfig: the configuration of emu_cheb's ChebBackend, in the style of emu-sv's SVConfig (see README.md)."""
from __future__ import annotations

import logging
import pathlib
from typing import Any, ClassVar

import numpy as np
from pulser.backend import (
    BitStrings,
    CorrelationMatrix,
    EmulationConfig,
    Energy,
    EnergySecondMoment,
    EnergyVariance,
    Expectation,
    Fidelity,
    Occupation,
    StateResult,
)

from emu_base import init_logging
from emu_sv import DenseOperator, StateVector

DIAGONAL = (Occupation, CorrelationMatrix, BitStrings)  # from the probabilities the engines compute anyway
OFF_DIAGONAL = (Energy, EnergyVariance, EnergySecondMoment, Expectation, Fidelity, StateResult)  # need psi itself
OPTIMIZATIONS = ("auto", "quench", "turbo", "fast", "reference")


class ChebConfig(EmulationConfig):
    """Configuration of ChebBackend. The options of pulser's EmulationConfig (observables, noise_model,
    n_trajectories, initial_state, interaction_matrix, with_modulation, prefer_device_noise_model,
    default_evaluation_times, default_num_shots) work as for emu-sv; note that pulser's default n_trajectories is 1:
    set it for noisy runs, as with emu-mps.

    Args:
        dt: drive sampling step (ns), as SVConfig.dt: the drive is piecewise constant on this grid.
        optimization: which engine of emu_cheb.solvers runs (README.md, "Engines"):
            "auto" (default): "quench" when the run allows it, "turbo" otherwise;
            "quench": rydberg_cheb_fast / heisenberg_cheb_fast (noiseless, constant drive, phase 0, from the ground
                state; ValueError otherwise);
            "turbo": noisy_cheb_turbo; "fast": noisy_cheb_fast; "reference": noisy_cheb (any sequence and noise;
                the three give the same trajectories up to round-off).
        gpu: None (default): CUDA if available; True / False force it.
        seed: seed of the trajectories' random numbers (noise samples come from pulser's RNG, as in emu-sv).
        heisenberg: Heisenberg couplings replacing the sequence's interaction (README.md, "Hamiltonians"): a
            (3, n, n) stack [Jxx, Jyy, Jzz] in rad/us (symmetric, zero diagonal, any graph), or (Jx, Jy, Jz)
            multiplying the sequence's own interaction matrix.
        single: complex64 instead of complex128 ("fast" / "turbo"; "auto" ignores it when it picks "quench").
        bounds: spectral bounds of "turbo" and "quench": "lanczos" (default, guarded) or "gershgorin" (rigorous).
        interaction_cutoff: interaction coefficients below it are set to 0 (as SVConfig).
        log_level, log_file: as SVConfig; the engine's statistics are logged at INFO.
    """

    _enforce_expected_kwargs: ClassVar[bool] = True
    _state_type = StateVector
    _operator_type = DenseOperator

    def __init__(
        self,
        *,
        dt: int = 10,
        optimization: str = "auto",
        gpu: bool | None = None,
        seed: int = 0,
        heisenberg: Any = None,
        single: bool = False,
        bounds: str = "lanczos",
        interaction_cutoff: float = 0.0,
        log_level: int = logging.INFO,
        log_file: pathlib.Path | None = None,
        **kwargs: Any,
    ):
        if optimization not in OPTIMIZATIONS:
            raise ValueError(f"optimization must be one of {OPTIMIZATIONS}, got {optimization!r}")
        if bounds not in ("lanczos", "gershgorin"):
            raise ValueError(f'bounds must be "lanczos" or "gershgorin", got {bounds!r}')
        if single and optimization in ("reference", "quench"):
            raise ValueError(f'single=True needs optimization "auto", "fast" or "turbo", not {optimization!r}')
        if heisenberg is not None:
            heisenberg = np.asarray(heisenberg, dtype=float)
            square_stack = heisenberg.ndim == 3 and heisenberg.shape[0] == 3 and heisenberg.shape[1] == heisenberg.shape[2]
            if not (heisenberg.shape == (3,) or square_stack):
                raise ValueError(f"heisenberg must have shape (3,) or (3, n, n), got {heisenberg.shape}")
        super().__init__(
            dt=dt,
            optimization=optimization,
            gpu=gpu,
            seed=seed,
            heisenberg=heisenberg,
            single=single,
            bounds=bounds,
            interaction_cutoff=interaction_cutoff,
            log_level=log_level,
            log_file=log_file,
            **kwargs,
        )
        for obs in self.observables:  # fail now, not after a long run
            if not isinstance(obs, DIAGONAL + OFF_DIAGONAL):
                raise NotImplementedError(f"emu_cheb does not support {type(obs).__name__} (README.md, Observables)")
            if isinstance(obs, StateResult) and (self.n_trajectories or 1) > 1:
                raise ValueError("StateResult needs a single trajectory; for noisy runs use Expectation or Fidelity")
        logger = init_logging(log_level, log_file)
        if self.noise_model.runs not in (1, None) or self.noise_model.samples_per_run not in (1, None):
            logger.warning("The runs and samples_per_run values of the NoiseModel are ignored!")

    def _expected_kwargs(self) -> set[str]:
        return super()._expected_kwargs() | {
            "dt",
            "optimization",
            "gpu",
            "seed",
            "heisenberg",
            "single",
            "bounds",
            "interaction_cutoff",
            "log_level",
            "log_file",
        }
