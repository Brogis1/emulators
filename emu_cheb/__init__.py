"""emu_cheb: an emu-sv-style backend (ChebBackend, ChebConfig) for the Chebyshev engines in emu_cheb.solvers (README.md)."""
from pulser.backend import (
    BitStrings,
    CorrelationMatrix,
    Energy,
    EnergySecondMoment,
    EnergyVariance,
    Expectation,
    Fidelity,
    Occupation,
    Results,
    StateResult,
)

from emu_sv import DenseOperator, StateVector

from .backend import ChebBackend
from .config import ChebConfig
from .observables import connected

__all__ = [
    "__version__",
    "BitStrings",
    "ChebBackend",
    "ChebConfig",
    "CorrelationMatrix",
    "DenseOperator",
    "Energy",
    "EnergySecondMoment",
    "EnergyVariance",
    "Expectation",
    "Fidelity",
    "Occupation",
    "Results",
    "StateResult",
    "StateVector",
    "connected",
]

__version__ = "0.1.0"
