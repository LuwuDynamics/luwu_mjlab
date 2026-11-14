"""Shared observation and actuator latency settings for velocity tasks."""

from __future__ import annotations

from dataclasses import replace

from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg

JOINT_OBS_DELAY_MIN_LAG = 1
JOINT_OBS_DELAY_MAX_LAG = 2
ACTUATOR_DELAY_MIN_LAG = 1
ACTUATOR_DELAY_MAX_LAG = 2

_JOINT_OBS_TERMS = ("joint_pos", "joint_vel")


def apply_joint_feedback_delay(cfg: ManagerBasedRlEnvCfg, play: bool = False) -> None:
  """Apply 1-2 step delay to joint position/velocity observations."""
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


def make_delayed_xml_actuator(play: bool = False) -> XmlActuatorCfg:
  """Create an XML actuator with optional 1-2 physics-step command delay."""
  return XmlActuatorCfg(
    target_names_expr=(".*",),
    delay_min_lag=0 if play else ACTUATOR_DELAY_MIN_LAG,
    delay_max_lag=0 if play else ACTUATOR_DELAY_MAX_LAG,
  )


def with_delayed_articulation(
  robot_cfg: EntityCfg, play: bool = False
) -> EntityCfg:
  """Apply command delay while preserving the robot's actuator configuration."""
  return with_delayed_articulation_multi(robot_cfg, play=play)


def with_delayed_articulation_multi(
  robot_cfg: EntityCfg, play: bool = False
) -> EntityCfg:
  """Apply native delay to each actuator group, including mixed XML types."""
  articulation = robot_cfg.articulation
  assert articulation is not None
  delayed_articulation = replace(
    articulation,
    actuators=tuple(
      replace(
        actuator,
        delay_min_lag=0 if play else ACTUATOR_DELAY_MIN_LAG,
        delay_max_lag=0 if play else ACTUATOR_DELAY_MAX_LAG,
      )
      for actuator in articulation.actuators
    ),
  )
  return replace(robot_cfg, articulation=delayed_articulation)
