"""Shared observation and actuator latency settings for velocity tasks."""

from __future__ import annotations

from dataclasses import replace

from mjlab.actuator import (
  DelayedActuatorCfg,
  XmlMotorActuatorCfg,
  XmlPositionActuatorCfg,
)
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg

JOINT_OBS_DELAY_MIN_LAG = 1
JOINT_OBS_DELAY_MAX_LAG = 2
ACTUATOR_DELAY_MIN_LAG = 1
ACTUATOR_DELAY_MAX_LAG = 2

_JOINT_OBS_TERMS = ("joint_pos", "joint_vel")


def apply_joint_feedback_delay(cfg: ManagerBasedRlEnvCfg, play: bool = False) -> None:
  """Apply 1-3 step delay to joint position/velocity observations."""
  min_lag = 0 if play else JOINT_OBS_DELAY_MIN_LAG
  max_lag = 0 if play else JOINT_OBS_DELAY_MAX_LAG
  for group_name in ("actor", "critic"):
    group = cfg.observations[group_name]
    for term_name in _JOINT_OBS_TERMS:
      if term_name not in group.terms:
        continue
      term = group.terms[term_name]
      term.delay_min_lag = min_lag
      term.delay_max_lag = max_lag


def make_delayed_xml_actuator(play: bool = False) -> DelayedActuatorCfg | XmlPositionActuatorCfg:
  """Create XML position actuator with optional 1-3 physics-step command delay."""
  base_cfg = XmlPositionActuatorCfg(target_names_expr=(".*",))
  if play:
    return base_cfg
  return DelayedActuatorCfg(
    base_cfg=base_cfg,
    delay_target="position",
    delay_min_lag=ACTUATOR_DELAY_MIN_LAG,
    delay_max_lag=ACTUATOR_DELAY_MAX_LAG,
  )


def with_delayed_articulation(
  robot_cfg: EntityCfg, play: bool = False
) -> EntityCfg:
  """Return robot cfg whose actuators use 1-3 step command delay when training."""
  articulation = robot_cfg.articulation
  delayed_articulation = EntityArticulationInfoCfg(
    actuators=(make_delayed_xml_actuator(play),),
    soft_joint_pos_limit_factor=articulation.soft_joint_pos_limit_factor,
  )
  return replace(robot_cfg, articulation=delayed_articulation)


def with_delayed_articulation_multi(
  robot_cfg: EntityCfg, play: bool = False
) -> EntityCfg:
  """Wrap each XML actuator group with a 1-2 physics-step command delay.

  Preserves the robot's existing actuator groups (unlike
  :func:`with_delayed_articulation`, which replaces everything with a single
  position actuator). Position actuators delay their position target; motor
  actuators delay their effort target. Used for robots that mix position servos
  with torque motors, e.g. ``mini2sw_v2`` (legs=position, wheels=motor).
  """
  articulation = robot_cfg.articulation
  new_actuators = []
  for actuator in articulation.actuators:
    if play:
      new_actuators.append(actuator)
      continue
    delay_target = (
      "effort" if isinstance(actuator, XmlMotorActuatorCfg) else "position"
    )
    new_actuators.append(
      DelayedActuatorCfg(
        base_cfg=actuator,
        delay_target=delay_target,
        delay_min_lag=ACTUATOR_DELAY_MIN_LAG,
        delay_max_lag=ACTUATOR_DELAY_MAX_LAG,
      )
    )
  delayed_articulation = EntityArticulationInfoCfg(
    actuators=tuple(new_actuators),
    soft_joint_pos_limit_factor=articulation.soft_joint_pos_limit_factor,
  )
  return replace(robot_cfg, articulation=delayed_articulation)
