"""Mini2SW flat-terrain recovery (turn-over) environment configuration."""

import math

from src.assets.robots import get_mini2sw_turn_robot_cfg
from src.tasks.velocity.mdp.actions import HeldJointPositionActionCfg
from src.tasks.velocity.latency import apply_joint_feedback_delay, with_delayed_articulation
from src.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from src.tasks.velocity.mdp import UniformVelocityCommandCfg

import src.tasks.velocity.mdp as src_mdp

_NUM_FEET = 4
_CRITIC_ZERO_FOOT_OBS = {
  "foot_height": _NUM_FEET,
  "foot_air_time": _NUM_FEET,
  "foot_contact": _NUM_FEET,
  "foot_contact_forces": _NUM_FEET * 3,
}


def mini2sw_flat_turn_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Mini2SW flat terrain recovery configuration."""
  cfg = make_velocity_env_cfg()
  cfg.episode_length_s = 8.0
  cfg.sim.mujoco.timestep = 0.002
  cfg.decimation = 10
  cfg.sim.njmax = 2000
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 128
  cfg.sim.nconmax = 64

  cfg.scene.entities = {
    "robot": with_delayed_articulation(get_mini2sw_turn_robot_cfg(), play=play),
  }

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  wheel_geom_names = (
    "fl_wheel_collision",
    "fr_wheel_collision",
    "bl_wheel_collision",
    "br_wheel_collision",
  )

  allowed_ground_geom_names = wheel_geom_names 

  nonfoot_ground_cfg = ContactSensorCfg(
    name="nonfoot_ground_touch",
    primary=ContactMatch(
      mode="geom",
      entity="robot",
      pattern=r".*",
      exclude=allowed_ground_geom_names,
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  ) + (nonfoot_ground_cfg,)

  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]
  cfg.observations["actor"].terms["phase"].params["period"] = 0.4
  apply_joint_feedback_delay(cfg, play=play)
  for name, size in _CRITIC_ZERO_FOOT_OBS.items():
    cfg.observations["critic"].terms[name].func = src_mdp.constant_zeros
    cfg.observations["critic"].terms[name].params = {"size": size}

  cfg.actions["joint_pos"] = HeldJointPositionActionCfg(
    entity_name="robot",
    actuator_names=(".*",),
    scale=0.25,
    use_default_offset=True,
    hold_duration_s=1.0,
  )

  cfg.viewer.body_name = "base"
  cfg.viewer.distance = 1.5
  cfg.viewer.elevation = -10.0

  cfg.events.pop("push_robot", None)
  cfg.events["reset_base"].params["pose_range"] = {
    "x": (-0.2, 0.2),
    "y": (-0.2, 0.2),
    "z": (0.05, 0.05),
    "roll": (-math.pi, math.pi),
    "pitch": (-math.pi, math.pi),
    "yaw": (-math.pi, math.pi),
  }

  cfg.events["reset_base"].params["velocity_range"] = {}
  cfg.events["reset_robot_joints"].params["position_range"] = (-0.3, 0.3)
  cfg.events["reset_robot_joints"].params["velocity_range"] = (-0.05, 0.05)
  cfg.events["reset_robot_joints"].params["asset_cfg"] = SceneEntityCfg(
    "robot", joint_names=(".*",)
  )
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = wheel_geom_names
  cfg.events["foot_friction"].params["ranges"] = (0.2, 3.0)
  cfg.events["base_com"].params["asset_cfg"].body_names = ("base",)

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.heading_command = False
  twist_cmd.rel_heading_envs = 0.0
  twist_cmd.rel_standing_envs = 1.0
  twist_cmd.resampling_time_range = (10.0, 10.0)
  twist_cmd.ranges.lin_vel_x = (0.0, 0.0)
  twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
  twist_cmd.ranges.ang_vel_z = (0.0, 0.0)
  twist_cmd.ranges.heading = None

  for key in (
    "track_linear_velocity",
    "track_angular_velocity",
    "foot_gait",
    "foot_clearance",
    "foot_slip",
    "soft_landing",
    "stand_still",
  ):
    cfg.rewards.pop(key, None)

  cfg.rewards["pose"].func = src_mdp.gravity_gated_variable_posture
  cfg.rewards["pose"].weight = 10.0
  cfg.rewards["pose"].params["gravity_z_threshold"] = -0.75
  cfg.rewards["pose"].params["asset_cfg"] = SceneEntityCfg(
    "robot",
    joint_names=(
      r"^(fl|fr|bl|br)_(hip|thigh|calf)_joint$",
      r"^arm_(yaw|thigh|calf)_joint$",
    ),
  )
  cfg.rewards["pose"].params["std_standing"] = {
    r"^(fl|fr|bl|br)_hip_joint$": 0.08,
    r"^(fl|fr|bl|br)_thigh_joint$": 0.15,
    r"^(fl|fr|bl|br)_calf_joint$": 0.20,
    r"^arm_yaw_joint$": 0.20,
    r"^arm_thigh_joint$": 0.25,
    r"^arm_calf_joint$": 1.0,
  }
  cfg.rewards["pose"].params["std_walking"] = {
    r"^(fl|fr|bl|br)_hip_joint$": 0.12,
    r"^(fl|fr|bl|br)_thigh_joint$": 0.25,
    r"^(fl|fr|bl|br)_calf_joint$": 0.30,
    r"^arm_yaw_joint$": 0.28,
    r"^arm_thigh_joint$": 0.32,
    r"^arm_calf_joint$": 1.0,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r"^(fl|fr|bl|br)_hip_joint$": 0.12,
    r"^(fl|fr|bl|br)_thigh_joint$": 0.30,
    r"^(fl|fr|bl|br)_calf_joint$": 0.35,
    r"^arm_yaw_joint$": 0.32,
    r"^arm_thigh_joint$": 0.36,
    r"^arm_calf_joint$": 1.0,
  }

  cfg.rewards["joint_vel_l2"] = RewardTermCfg(
    func=mdp.joint_vel_l2,
    weight=-5e-3,
  )

  cfg.rewards["body_orientation_l2"].weight = -0.1
  cfg.rewards["body_orientation_l2"].params["asset_cfg"].body_names = ("base",)
  cfg.rewards["body_ang_vel"].func = src_mdp.gravity_gated_body_ang_vel_penalty
  cfg.rewards["body_ang_vel"].weight = -0.2
  cfg.rewards["body_ang_vel"].params["gravity_z_threshold"] = 1.0
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("base",)
  cfg.rewards["angular_momentum"].func = src_mdp.gravity_gated_angular_momentum_penalty
  cfg.rewards["angular_momentum"].weight = -0.2
  cfg.rewards["angular_momentum"].params["gravity_z_threshold"] = -0.4

  cfg.rewards["recovery_progress"] = RewardTermCfg(
    func=src_mdp.recovery_progress,
    weight=10.0,
    params={"asset_cfg": SceneEntityCfg("robot")},
  )
  cfg.rewards["nonfoot_contact"] = RewardTermCfg(
    func=mdp.illegal_contact,
    weight=-0.5,
    params={"sensor_name": nonfoot_ground_cfg.name, "force_threshold": 0.5},
  )
  cfg.rewards["action_rate_l2"] = RewardTermCfg(
    func=mdp.action_rate_l2,
    weight=-0.25,
  )

  cfg.terminations.pop("fell_over", None)
  cfg.terminations.pop("illegal_contact", None)
  cfg.curriculum.clear()

  if play:
    cfg.episode_length_s = 5.0
    cfg.observations["actor"].enable_corruption = False

    cfg.scene.num_envs = 1
  return cfg
