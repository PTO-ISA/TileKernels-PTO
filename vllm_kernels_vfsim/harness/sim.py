"""Launch a case under ``msprof op simulator`` when no NPU is present.

The profiled program must make the TileLang kernel the first device operator.
Build tensors on the host and move them with ``.npu()``. Do not call
``torch.zeros(..., device='npu')`` before the kernel: that fill is an operator,
and msprof profiles only the first one.

``cycles`` are read from ``core0.veccore0_instr_exe.csv`` by harness.cycles.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .cycles import sum_vf_cycles

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOC = "Ascend950PR_9599"

_KEEP = (
    "[status]",
    "[sim]",
    "PASS",
    "FAIL",
    "AssertionError",
    "Traceback",
    "Error",
    "error:",
    "VMI-",
)


def npu_ready() -> bool:
    try:
        import torch
        import torch_npu  # noqa: F401

        return bool(torch.npu.is_available())
    except Exception:
        return False


def should_reexec() -> bool:
    if os.environ.get("VFSIM_CPU_ONLY") == "1":
        return False
    if os.environ.get("TLP_IN_SIMULATOR") == "1":
        return False
    return not npu_ready()


def _child_env() -> dict:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{ROOT}{os.pathsep}{existing}" if existing else str(ROOT)
    env["TLP_IN_SIMULATOR"] = "1"
    ascend = env.get("ASCEND_HOME_PATH", "")
    if ascend:
        sim_lib = str(Path(ascend) / "tools" / "simulator" / DEFAULT_SOC / "lib")
        env["LD_LIBRARY_PATH"] = sim_lib + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    return env


def _relay(out: str) -> None:
    for line in out.splitlines():
        if any(key in line for key in _KEEP):
            print(line)


def _passed(out: str) -> bool:
    if "\nFAIL" in out or out.strip().endswith("FAIL"):
        return False
    return "\nPASS" in out or out.strip().endswith("PASS") or "result=PASS" in out


def _out_dir() -> Path:
    root = Path(os.environ.get("VFSIM_SIM_ROOT", Path.home() / ".cache" / "vfsim"))
    root.mkdir(parents=True, exist_ok=True)
    try:
        root.chmod(0o700)
    except OSError:
        pass
    path = Path(tempfile.mkdtemp(prefix="run_", dir=root))
    try:
        path.chmod(0o700)
    except OSError:
        pass
    return path


def run_under_msprof(argv: list[str]) -> dict:
    """Run ``python <argv>`` under msprof. Returns status, stdout, out_dir, cycles."""
    msprof = shutil.which("msprof")
    ascend = os.environ.get("ASCEND_HOME_PATH")
    if not msprof or not ascend:
        raise RuntimeError("msprof or ASCEND_HOME_PATH is missing")
    sim_lib = Path(ascend) / "tools" / "simulator" / DEFAULT_SOC / "lib"
    if not sim_lib.is_dir():
        raise RuntimeError(f"no simulator libs at {sim_lib}")
    out_dir = _out_dir()
    env = _child_env()
    cmd = [
        msprof,
        "op",
        "simulator",
        f"--soc-version={DEFAULT_SOC}",
        f"--output={out_dir}",
        sys.executable,
        *argv,
    ]
    print(f"[sim] out={out_dir}")
    start = time.time()
    proc = subprocess.run(
        cmd,
        env=env,
        cwd=str(Path.home()),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    elapsed = time.time() - start
    _relay(proc.stdout)
    passed = _passed(proc.stdout)
    if proc.returncode != 0 and passed:
        print(f"[sim] ignoring teardown status {proc.returncode} after PASS")
    cycles = sum_vf_cycles(out_dir) if passed else {"cycles": None, "csv": None}
    print(
        f"[sim] wall {elapsed:.1f}s soc={DEFAULT_SOC} exit={proc.returncode} "
        f"pass={passed} vf_cycles={cycles.get('cycles')}"
    )
    return {
        "passed": passed,
        "returncode": 0 if passed else 1,
        "stdout": proc.stdout,
        "out_dir": str(out_dir),
        "cycles": cycles,
        "seconds": elapsed,
    }


def run_script(script: Path, impl: str) -> dict:
    return run_under_msprof([str(script.resolve()), impl])


def maybe_reexec() -> None:
    """Re-launch this process under the simulator. Exits the parent."""
    if not should_reexec():
        return
    script = str(Path(sys.argv[0]).resolve())
    result = run_under_msprof([script, *sys.argv[1:]])
    raise SystemExit(result["returncode"])
