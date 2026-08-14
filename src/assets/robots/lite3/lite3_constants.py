"""XGO Lite3 robot constants."""

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

LITE3_XML: Path = (
  SRC_PATH / "assets" / "robots" / "lite3" / "xmls" / "lite3.xml"
)
assert LITE3_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  """Load mesh assets for XGO Lite3 from the mesh directory referenced by meshdir."""
  assets: dict[str, bytes] = {}
  meshdir_clean = meshdir.rstrip("/")
  mesh_path = (LITE3_XML.parent / meshdir_clean).resolve()
  update_assets(assets, mesh_path, meshdir_clean)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(LITE3_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


##
# Actuator config.
##

LITE3_XML_ACTUATOR = XmlPositionActuatorCfg(
  target_names_expr=(".*",),
)

##
# Keyframes.
##

INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.10),
  joint_pos={
    "^(FL|FR|RR|RL)_thigh_joint$": 0.95,
    "^(FL|FR|RR|RL)_shank_joint$": 0.0,
    "^(FL|FR|RR|RL)_hip_joint$": 0.0,
    "mainarm_joint": -1.55,
    "forearm_joint": 1.3,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

_SHANK_FOOT_REGEX = r"^(FL|FR|RR|RL)_shank_collision$"

# Only shank feet collide with ground; all other robot geoms are disabled.
LITE3_WALK_COLLISION = CollisionCfg(
  geom_names_expr=(_SHANK_FOOT_REGEX,),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
  solimp=(0.9, 0.95, 0.023),
  disable_other_geoms=True,
)

# All robot collision geoms touch ground; no self-collision (contype=1, conaffinity=0).
LITE3_TURN_COLLISION = CollisionCfg(
  geom_names_expr=(r".*_collision",),
  condim={_SHANK_FOOT_REGEX: 3, r".*_collision": 1},
  priority={_SHANK_FOOT_REGEX: 1},
  friction={_SHANK_FOOT_REGEX: (0.6,)},
  solimp={_SHANK_FOOT_REGEX: (0.9, 0.95, 0.023)},
  contype=1,
  conaffinity=0,
  disable_other_geoms=False,
)

##
# Final config.
##

LITE3_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(LITE3_XML_ACTUATOR,),
  soft_joint_pos_limit_factor=0.9,
)


def get_LITE3_robot_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(),
    spec_fn=get_spec,
    articulation=LITE3_ARTICULATION,
  )


def get_lite3_robot_cfg() -> EntityCfg:
  return get_LITE3_robot_cfg()


def get_lite3_walk_robot_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(LITE3_WALK_COLLISION,),
    spec_fn=get_spec,
    articulation=LITE3_ARTICULATION,
  )


def get_lite3_turn_robot_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(LITE3_TURN_COLLISION,),
    spec_fn=get_spec,
    articulation=LITE3_ARTICULATION,
  )
