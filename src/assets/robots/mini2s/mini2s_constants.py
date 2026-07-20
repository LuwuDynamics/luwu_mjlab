"""Unitree Mini2S constants."""

from pathlib import Path

import mujoco

from src import SRC_PATH
from mjlab.actuator import XmlPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

MINI2S_XML: Path = (
  SRC_PATH / "assets" / "robots" / "mini2s" / "xmls" / "mini2s.xml"
)
assert MINI2S_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  """Load mesh assets for Mini2S from the mesh directory referenced by meshdir."""
  assets: dict[str, bytes] = {}
  meshdir_clean = meshdir.rstrip("/")
  mesh_path = (MINI2S_XML.parent / meshdir_clean).resolve()
  update_assets(assets, mesh_path, meshdir_clean)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(MINI2S_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


##
# Actuator config.
##

MINI2S_XML_ACTUATOR = XmlPositionActuatorCfg(
  target_names_expr=(".*",),
)

##
# Keyframes.
##

INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.20),
  joint_pos={
    "^(fl|fr|bl|br)_thigh_joint$": 1,
    "^(fl|fr|bl|br)_calf_joint$": 0.0,
    "^(fl|fr|bl|br)_hip_joint$": 0.0,
    "arm_yaw_joint": 0.0,
    "arm_thigh_joint": -1.57,
    "arm_calf_joint": 1.35,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

_CALF_FOOT_REGEX = r"^(fl|fr|bl|br)_calf_collision$"

# Only calf feet collide with ground; all other robot geoms are disabled.
MINI2S_WALK_COLLISION = CollisionCfg(
  geom_names_expr=(_CALF_FOOT_REGEX,),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
  solimp=(0.9, 0.95, 0.023),
  disable_other_geoms=True,
)

# All robot collision geoms touch ground; no self-collision (contype=1, conaffinity=0).
MINI2S_TURN_COLLISION = CollisionCfg(
  geom_names_expr=(r".*_collision",),
  condim={_CALF_FOOT_REGEX: 3, r".*_collision": 1},
  priority={_CALF_FOOT_REGEX: 1},
  friction={_CALF_FOOT_REGEX: (0.6,)},
  solimp={_CALF_FOOT_REGEX: (0.9, 0.95, 0.023)},
  contype=1,
  conaffinity=0,
  disable_other_geoms=False,
)

##
# Final config.
##

MINI2S_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(MINI2S_XML_ACTUATOR,),
  soft_joint_pos_limit_factor=0.9,
)


def get_MINI2S_robot_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(),
    spec_fn=get_spec,
    articulation=MINI2S_ARTICULATION,
  )


def get_mini2s_robot_cfg() -> EntityCfg:
  return get_MINI2S_robot_cfg()


def get_mini2s_walk_robot_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(MINI2S_WALK_COLLISION,),
    spec_fn=get_spec,
    articulation=MINI2S_ARTICULATION,
  )


def get_mini2s_turn_robot_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(MINI2S_TURN_COLLISION,),
    spec_fn=get_spec,
    articulation=MINI2S_ARTICULATION,
  )
