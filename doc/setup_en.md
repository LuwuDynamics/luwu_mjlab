# luwu_mjlab Installation Guide

Installation guide for the [Luwu Dynamics](https://www.xgorobot.com/) RL training environment.

## Requirements

- Ubuntu 22.04 (recommended)
- NVIDIA GPU with driver 550+
- Python 3.11

## 1. Create Conda environment

```bash
conda create -n luwu_mjlab python=3.11
conda activate luwu_mjlab
```

## 2. Install luwu_mjlab

```bash
git clone https://github.com/LuwuDynamics-RIG/luwu_mjlab.git
cd luwu_mjlab
pip install -e .
```

This installs [**mjlab**](https://github.com/mujocolab/mjlab) and MuJoCo Warp as dependencies.

## 3. Verify

```bash
# List Luwu tasks only
python scripts/list_envs.py --keyword XGOMini

# Zero-action sanity check
python scripts/play.py XGOMini-Flat --agent zero
```

## 4. Start training

```bash
python scripts/train.py XGOMini-Flat --env.scene.num-envs=4096
```

Logs are saved under `logs/rsl_rl/xgomini_velocity/` (TensorBoard).

View TensorBoard:

```bash
tensorboard --logdir logs/rsl_rl/xgomini_velocity
```
