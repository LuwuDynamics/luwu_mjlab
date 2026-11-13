"""Run the upstream CPU trainers and diagnostics from the luwu_mjlab checkout."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
RIDER = "Rider2-Balance-Flat"
TARS = "Tars-Getup"
TASKS = {
  RIDER: (
    "rider", "third_party/xgorider2_rl/training",
    {"train": "train2.py", "play": "viewer2.py", "evaluate": "evaluate2.py",
     "debug": "tools/sim_frf_check.py"},
  ),
  TARS: (
    "tars", "third_party/rig_tars_rl/tars_rl",
    {"train": "rl/ppo_getup.py", "play": str(ROOT / "scripts/tars_replay.py"),
     "evaluate": "rl/eval_getup.py", "debug": "rl/getup_audit.py",
     "firmware": "rl/eval_getup_c.py"},
  ),
}


def run(task: str, action: str, args: list[str]) -> int:
  name, relative_path, scripts = TASKS[task]
  workdir = ROOT / relative_path
  if not (workdir / "requirements.txt").is_file():
    raise SystemExit("Initialize task sources with: git submodule update --init")
  if action not in scripts:
    raise SystemExit(f"{task} does not provide {action}.")

  env = os.environ.copy()
  interpreter_var = f"LUWU_{name.upper()}_PYTHON"
  interpreter = env.get(interpreter_var)
  if not interpreter:
    candidates = (
      ROOT.parent / "venvs" / name / "bin/python",
      workdir / ".venv/bin/python",
    )
    interpreter = next((str(path) for path in candidates if path.is_file()), None)
  if not interpreter or not shutil.which(interpreter):
    raise SystemExit(f"Install {workdir}/requirements.txt in a dedicated environment "
                     f"and set {interpreter_var} to its Python executable.")

  # Let the upstream interpreter resolve its own packages and MuJoCo version.
  env.pop("PYTHONPATH", None)
  env.pop("PYTHONHOME", None)
  env.setdefault("OMP_NUM_THREADS", "1")
  env.setdefault("OPENBLAS_NUM_THREADS", "1")
  toolchain = ROOT.parent / "toolchain/usr"
  if name == "rider" and (toolchain / "bin/clang++").is_file():
    env["PATH"] = f"{toolchain / 'bin'}{os.pathsep}{env.get('PATH', '')}"
    libraries = toolchain / "lib/x86_64-linux-gnu"
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
      [str(libraries)] + ([env["LD_LIBRARY_PATH"]] if env.get("LD_LIBRARY_PATH") else [])
    )

  script = workdir / scripts[action]
  return subprocess.run([interpreter, str(script), *args], cwd=workdir, env=env).returncode


def dispatch(action: str) -> None:
  if len(sys.argv) > 1 and sys.argv[1] in TASKS:
    raise SystemExit(run(sys.argv[1], action, sys.argv[2:]))


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("action", choices=("evaluate", "debug", "firmware"))
  parser.add_argument("task", choices=tuple(TASKS))
  parser.add_argument("args", nargs=argparse.REMAINDER)
  options = parser.parse_args()
  raise SystemExit(run(options.task, options.action, options.args))
