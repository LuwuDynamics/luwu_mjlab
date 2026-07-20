"""Task-specific action terms for velocity tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class HeldJointPositionActionCfg(JointPositionActionCfg):
  """Joint position action that holds default pose for an initial duration."""

  hold_duration_s: float = 2.0

  def build(self, env: ManagerBasedRlEnv) -> HeldJointPositionAction:
    return HeldJointPositionAction(self, env)


class HeldJointPositionAction(JointPositionAction):
  """Hold the reset joint pose for ``hold_duration_s``, then apply policy actions."""

  cfg: HeldJointPositionActionCfg

  def process_actions(self, actions: torch.Tensor) -> None:
    super().process_actions(actions)
    elapsed = self._env.episode_length_buf * self._env.step_dt
    hold_mask = elapsed < self.cfg.hold_duration_s
    if not hold_mask.any():
      return
    self._raw_actions[hold_mask] = 0.0
    if isinstance(self._scale, float):
      self._processed_actions[hold_mask] = (
        self._raw_actions[hold_mask] * self._scale + self._offset[hold_mask]
      )
    else:
      self._processed_actions[hold_mask] = (
        self._raw_actions[hold_mask] * self._scale[hold_mask]
        + self._offset[hold_mask]
      )

  def apply_actions(self) -> None:
    elapsed = self._env.episode_length_buf * self._env.step_dt
    hold_mask = elapsed < self.cfg.hold_duration_s
    if hold_mask.any():
      target = self._entity.data.joint_pos[:, self._target_ids].clone()
      if (~hold_mask).any():
        encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]
        target[~hold_mask] = self._processed_actions[~hold_mask] - encoder_bias[~hold_mask]
    else:
      encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]
      target = self._processed_actions - encoder_bias
    self._entity.set_joint_position_target(target, joint_ids=self._target_ids)
