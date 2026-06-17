from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import xgomini_flat_env_cfg
from .rl_cfg import xgomini_ppo_runner_cfg

register_mjlab_task(
  task_id="XGOMini-Flat",
  env_cfg=xgomini_flat_env_cfg(),
  play_env_cfg=xgomini_flat_env_cfg(play=True),
  rl_cfg=xgomini_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
