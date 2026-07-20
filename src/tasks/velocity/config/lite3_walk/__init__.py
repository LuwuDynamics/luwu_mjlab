from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import lite3_flat_walk_env_cfg
from .rl_cfg import lite3_walk_ppo_runner_cfg

register_mjlab_task(
  task_id="Lite3-Walk-Flat",
  env_cfg=lite3_flat_walk_env_cfg(),
  play_env_cfg=lite3_flat_walk_env_cfg(play=True),
  rl_cfg=lite3_walk_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
