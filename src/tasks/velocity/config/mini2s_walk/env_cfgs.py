"""Mini2S walking velocity environment configurations."""

from src.assets.robots import get_mini2s_walk_robot_cfg
from src.tasks.velocity.latency import apply_joint_feedback_delay, with_delayed_articulation
from src.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from src.tasks.velocity.mdp import UniformVelocityCommandCfg


def _mini2s_walk_base_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Mini2S base walking velocity configuration with calf foot contact."""
  cfg = make_velocity_env_cfg()

  cfg.sim.mujoco.timestep = 0.004
  cfg.decimation = 5
  # cfg.sim.mujoco.ccd_iterations = 100
  # cfg.sim.contact_sensor_maxmatch = 128

  cfg.scene.entities = {
    "robot": with_delayed_articulation(get_mini2s_walk_robot_cfg(), play=play),
  }

  foot_names = ("fl", "fr", "bl", "br")
  site_names = foot_names
  calf_geom_names = tuple(f"{name}_calf_collision" for name in foot_names)

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="geom", pattern=calf_geom_names, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )

  nonfoot_ground_cfg = ContactSensorCfg(
    name="nonfoot_ground_touch",
    primary=ContactMatch(
      mode="geom",
      entity="robot",
      pattern=r".*",
      exclude=calf_geom_names,
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )

  cfg.scene.sensors = (cfg.scene.sensors or ()) + (feet_ground_cfg, nonfoot_ground_cfg)

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = 0.25

  cfg.viewer.body_name = "base"
  cfg.viewer.distance = 1.5
  cfg.viewer.elevation = -10.0

  cfg.observations["critic"].terms["foot_height"].params["asset_cfg"].site_names = site_names
  cfg.observations["actor"].terms["phase"].params["period"] = 0.4
  apply_joint_feedback_delay(cfg, play=play)

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = calf_geom_names
  cfg.events["foot_friction"].params["ranges"] = (0.2, 3.0)
  cfg.events["base_com"].params["asset_cfg"].body_names = ("base",)

  cfg.rewards["pose"].params["std_standing"] = {
    r".*(fl|fr|bl|br)_hip_joint.*": 0.05,
    r".*(fl|fr|bl|br)_thigh_joint.*": 0.1,
    r".*(fl|fr|bl|br)_calf_joint.*": 0.15,
    r"^arm_yaw_joint$": 0.1,
    r"^arm_thigh_joint$": 0.1,
    r"^arm_calf_joint$": 0.1,
  }
  cfg.rewards["pose"].params["std_walking"] = {
    r".*(fl|fr|bl|br)_hip_joint.*": 0.15,
    r".*(fl|fr|bl|br)_thigh_joint.*": 0.35,
    r".*(fl|fr|bl|br)_calf_joint.*": 0.35,
    r"^arm_yaw_joint$": 0.1,
    r"^arm_thigh_joint$": 0.1,
    r"^arm_calf_joint$": 0.1,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*(fl|fr|bl|br)_hip_joint.*": 0.15,
    r".*(fl|fr|bl|br)_thigh_joint.*": 0.5,
    r".*(fl|fr|bl|br)_calf_joint.*": 0.5,
    r"^arm_yaw_joint$": 0.1,
    r"^arm_thigh_joint$": 0.1,
    r"^arm_calf_joint$": 0.1,
  }
  cfg.rewards["pose"].params["walking_threshold"] = 0.05
  cfg.rewards["pose"].params["running_threshold"] = 0.3

  cfg.rewards["track_linear_velocity"].weight = 4.0
  cfg.rewards["track_linear_velocity"].params["std"] = 0.15
  cfg.rewards["track_angular_velocity"].weight = 4.0
  cfg.rewards["track_angular_velocity"].params["std"] = 0.15

  cfg.rewards["body_ang_vel"].weight = -0.08
  cfg.rewards["angular_momentum"].weight = -0.03
  cfg.rewards["foot_gait"].params["period"] = 0.4
  cfg.rewards["foot_gait"].weight = 1.0
  cfg.rewards["foot_slip"].weight = -0.05

  cfg.rewards["foot_gait"].params["offset"] = [0.0, 0.5, 0.0, 0.5]
  cfg.rewards["body_orientation_l2"].params["asset_cfg"].body_names = ("base",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("base",)
  cfg.rewards["foot_clearance"].params["asset_cfg"].site_names = site_names
  cfg.rewards["foot_clearance"].params["target_height"] = 0.02
  cfg.rewards["foot_clearance"].weight = -0.5
  cfg.rewards["foot_slip"].params["asset_cfg"].site_names = site_names

  cfg.rewards["nonfoot_contact"] = RewardTermCfg(
    func=mdp.illegal_contact,
    weight=-3,
    params={"sensor_name": nonfoot_ground_cfg.name, "force_threshold": 0.5},
  )

  cfg.rewards["action_rate_l2"] = RewardTermCfg(
    func=mdp.action_rate_l2,
    weight=-0.2,
  )

  cfg.terminations["illegal_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": nonfoot_ground_cfg.name, "force_threshold": 10.0},
  )

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
      cfg.scene.terrain.terrain_generator.curriculum = False
      cfg.scene.terrain.terrain_generator.num_cols = 5
      cfg.scene.terrain.terrain_generator.num_rows = 5
      cfg.scene.terrain.terrain_generator.border_width = 10.0
    if play:
      cfg.scene.num_envs = 1
  return cfg


def mini2s_flat_walk_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Mini2S flat terrain walking velocity configuration."""
  cfg = _mini2s_walk_base_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.heading_command = False
  twist_cmd.rel_heading_envs = 0.1
  twist_cmd.rel_standing_envs = 0.0
  twist_cmd.ranges.lin_vel_x = (-0.5, 0.6)
  twist_cmd.ranges.lin_vel_y = (-0.3, 0.3)
  twist_cmd.ranges.ang_vel_z = (-0.8, 0.8)
  twist_cmd.ranges.heading = None

  cfg.curriculum.pop("terrain_levels", None)
  cfg.curriculum.pop("command_vel", None)

  if play:
    twist_cmd.ranges.lin_vel_x = (-0.35, 0.45)
    twist_cmd.ranges.lin_vel_y = (-0.25, 0.25)
    twist_cmd.ranges.ang_vel_z = (-0.0, 0.0)
    cfg.scene.num_envs = 1
  return cfg
