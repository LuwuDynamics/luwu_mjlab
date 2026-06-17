# luwu_mjlab

**luwu_mjlab** is the reinforcement learning training environment for [**Luwu Dynamics**](https://www.xgorobot.com/) (陆吾智能) products. It is used to train and validate locomotion policies in MuJoCo simulation before deployment.

This project is built on the open-source [**mjlab**](https://github.com/mujocolab/mjlab) framework, which combines an Isaac Lab–style manager API with MuJoCo Warp GPU simulation. luwu_mjlab adds Luwu robot models and task configurations on top of mjlab.

## Preview

| Simulation | Real robot |
|------------|------------|
| ![xgomini walk sim](doc/gif/xgomini_walk_sim.gif) | ![xgomini walk real](doc/gif/xgomini_walk_real.gif) |
| ![xgomini getup sim](doc/gif/xgomini_getup_sim.gif) | ![xgomini getup real](doc/gif/xgomini_getup_real.gif) |

## Install

```bash
conda create -n luwu_mjlab python=3.11
conda activate luwu_mjlab
cd luwu_mjlab
pip install -e .
```

See [doc/setup_en.md](doc/setup_en.md) for details.

## Training (XGOMini example)

```bash
python scripts/train.py XGOMini-Flat --env.scene.num-envs=4096
```

## Play / evaluation

```bash
# MDP sanity check (no checkpoint needed)
python scripts/play.py XGOMini-Flat --agent zero

# Load latest local checkpoint
python scripts/play.py XGOMini-Flat

# Load a specific checkpoint
python scripts/play.py XGOMini-Flat \
  --checkpoint-file logs/rsl_rl/xgomini_velocity/2026-xx-xx_xx-xx-xx/model_xx.pt
```

## Links

- [Luwu Dynamics](https://www.xgorobot.com/)
- [mjlab repository](https://github.com/mujocolab/mjlab)
- [mjlab documentation](https://mujocolab.github.io/mjlab/)
