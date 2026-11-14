"""Render a Tars policy rollout using the upstream evaluation environment."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "src/tasks/tars/tars_rl"
sys.path.insert(0, str(ROOT / "rl"))
from eval_getup import DT, GetupEnv, STRESS, cal_kw, load_policy  # noqa: E402
from render_utils import render_strip  # noqa: E402


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--policy", type=Path,
                      default=ROOT / "checkpoints/getup_ppo_c3/final_it500.pt")
  parser.add_argument("--out", type=Path, required=True, help="Output PNG")
  parser.add_argument("--plant", choices=("cal", "orig"), default="cal")
  parser.add_argument("--kind", choices=("back", "front", "random"), default="back")
  parser.add_argument("--stress", choices=tuple(STRESS), default="none")
  parser.add_argument("--seed", type=int, default=6000)
  parser.add_argument("--seconds", type=float, default=12)
  args = parser.parse_args()
  if args.seconds < DT:
    parser.error(f"--seconds must be at least {DT}")
  if args.out.suffix.lower() != ".png":
    parser.error("--out must end in .png")

  policy, checkpoint = load_policy(args.policy)
  cfg = checkpoint["args"]
  plant = cal_kw(args.stress)
  initial_pose = None if args.kind == "random" else {"kind": args.kind, "hip": (0.0, 0.0)}
  observation_options = {
    key: value for key, value in json.loads(cfg.get("env_kw") or "{}").items()
    if key in ("obs_q_mode", "imu_clip")
  }
  env = GetupEnv(
    seed=args.seed, plant_cal=plant, plant_mix=float(args.plant == "cal"),
    max_steps=round(args.seconds / DT), com_shift=plant.get("com_shift"),
    getup_init=initial_pose, action_repeat=cfg.get("repeat", 5),
    time_obs=bool(cfg.get("time_obs", 1)), rate_hip=cfg.get("rate_hip", 8.0),
    rate_lift=cfg.get("rate_lift", 0.25), continue_after_success=False,
    **observation_options,
  )
  obs, _ = env.reset()
  frames = [env.d.qpos.copy()]
  times = [env.t]
  while True:
    action = policy(obs)
    obs, reward, done, info = env.step(action)
    if not all(np.isfinite(x).all() for x in (obs, action, reward, env.d.qpos)):
      raise RuntimeError(f"Non-finite state at {env.t:.3f} s")
    frames.append(env.d.qpos.copy())
    times.append(env.t)
    if done or info["truncated"]:
      break

  samples = np.linspace(0, len(frames) - 1, min(8, len(frames)), dtype=int)
  args.out.parent.mkdir(parents=True, exist_ok=True)
  render_strip(env.m, [frames[i] for i in samples], args.out,
               labels=[f"{times[i]:.2f} s" for i in samples])
  print(json.dumps({"success": env.succeeded, "time_to_stand": env.t_success,
                    "duration": env.t, "policy_steps": len(frames) - 1,
                    "frames": str(args.out.resolve())}))


if __name__ == "__main__":
  main()
