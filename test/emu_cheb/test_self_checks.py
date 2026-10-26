"""The engines' own self-checks (python -m emu_cheb.solvers.<module>: dense references, trajectory-by-trajectory
identities, Lindblad limits; CPU, a few minutes in total). Opt-in: EMU_CHEB_SELF_CHECKS=1 pytest test/emu_cheb."""
import os
import subprocess
import sys

import pytest

MODULES = ["rydberg_cheb_fast", "heisenberg_cheb_fast", "noisy_cheb", "noisy_cheb_fast", "noisy_cheb_turbo"]


@pytest.mark.skipif(os.environ.get("EMU_CHEB_SELF_CHECKS") != "1", reason="slow: set EMU_CHEB_SELF_CHECKS=1")
@pytest.mark.parametrize("module", MODULES)
def test_self_check(module):
    run = subprocess.run([sys.executable, "-W", "ignore", "-m", f"emu_cheb.solvers.{module}"], capture_output=True,
                         text=True, timeout=1800)
    assert run.returncode == 0 and run.stdout.strip().endswith("self-check OK"), run.stdout[-2000:] + run.stderr[-2000:]
