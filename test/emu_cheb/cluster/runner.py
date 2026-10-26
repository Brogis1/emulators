"""Persistent GPU test runner, for fast iteration on the cluster.

Keeps torch / pulser / emu-mps imported, Triton kernels compiled and test data cached (in the namespace G shared by
all tasks), so that a test takes seconds instead of a batch job. Task files (*.py) dropped into <dir>/inbox are run in
G, in name order; their output goes to <dir>/outbox/<name>.out, then <name>.done marks completion. Modules of emu_cheb
(solvers first) and of examples/emu_cheb/experiments are reloaded, dependencies first, when their file changed since
the previous task.

Usage, from the repo root inside tmux on a GPU allocation (emu_cheb installed: pip install -e ci/emu_cheb; release the
allocation when done):
    srun -p internal -G2 -c 16 --mem 96G --time 02:00:00 python test/emu_cheb/cluster/runner.py /work/<user>/runner
"""
import contextlib
import glob
import importlib
import io
import os
import sys
import time
import traceback

root = sys.argv[1]
repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
experiments = os.path.join(repo, "examples", "emu_cheb", "experiments")
sys.path.insert(0, experiments)  # tiled_step, multi_gpu
os.chdir(repo)
for sub in ("inbox", "outbox"):
    os.makedirs(os.path.join(root, sub), exist_ok=True)
WATCHED = [  # (module, file), dependency order
    *[(f"emu_cheb.solvers.{m}", os.path.join(repo, "emu_cheb", "solvers", m + ".py"))
      for m in ("rydberg_cheb_fast", "noisy_cheb", "noisy_cheb_fast", "noisy_cheb_turbo", "heisenberg_cheb_fast")],
    *[(f"emu_cheb.{m}", os.path.join(repo, "emu_cheb", m + ".py")) for m in ("config", "observables", "backend")],
    *[(m, os.path.join(experiments, m + ".py")) for m in ("tiled_step", "multi_gpu")],
]
mtimes: dict = {}
G: dict = {"__name__": "task"}
G["G"] = G  # tasks reach the persistent namespace as G


def reload_changed():
    stale = False
    for name, path in WATCHED:
        if not os.path.exists(path):
            continue
        m = os.path.getmtime(path)
        if mtimes.get(name) != m:
            stale = True  # reload this and everything after it (they may hold references to it)
        mtimes[name] = m
        if stale and name in sys.modules:
            importlib.reload(sys.modules[name])


print("runner ready", flush=True)
while True:
    for task in sorted(glob.glob(os.path.join(root, "inbox", "*.py"))):
        name = os.path.splitext(os.path.basename(task))[0]
        src = open(task).read()
        os.remove(task)
        buf = io.StringIO()
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            try:
                reload_changed()
                exec(compile(src, task, "exec"), G)
            except BaseException:  # noqa: BLE001 - report everything, keep the runner alive
                traceback.print_exc()
        buf.write(f"[runner] {name}: {time.perf_counter() - t0:.1f} s\n")
        with open(os.path.join(root, "outbox", name + ".out"), "w") as f:
            f.write(buf.getvalue())
        open(os.path.join(root, "outbox", name + ".done"), "w").close()
    time.sleep(0.3)
