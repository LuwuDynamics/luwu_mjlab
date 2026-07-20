from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import mini2sw_flat_walk_env_cfg
from .rl_cfg import mini2sw_walk_ppo_runner_cfg

register_mjlab_task(
  task_id="Mini2SW-Walk-Flat",
  env_cfg=mini2sw_flat_walk_env_cfg(),
  play_env_cfg=mini2sw_flat_walk_env_cfg(play=True),
  rl_cfg=mini2sw_walk_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
