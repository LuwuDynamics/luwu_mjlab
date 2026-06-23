# luwu_mjlab 安装指南

[陆吾智能（Luwu Dynamics）](https://www.xgorobot.com/) 强化学习训练环境安装说明。

## 系统要求

- Ubuntu 22.04（推荐）
- NVIDIA GPU + 驱动 550+
- Python 3.11

## 1. 创建 Conda 环境

```bash
conda create -n luwu_mjlab python=3.11
conda activate luwu_mjlab
```

## 2. 安装 luwu_mjlab

```bash
git clone https://github.com/LuwuDynamics/luwu_mjlab.git
cd luwu_mjlab
pip install -e .
```

安装完成后会自动拉取依赖 [**mjlab**](https://github.com/mujocolab/mjlab) 与 MuJoCo Warp。

## 3. 验证

```bash
# 列出陆吾任务
python scripts/list_envs.py --keyword XGOMini

# 零动作 sanity check
python scripts/play.py XGOMini-Flat --agent zero
```

## 4. 开始训练

```bash
python scripts/train.py XGOMini-Flat --env.scene.num-envs=4096
```

日志目录：`logs/rsl_rl/xgomini_velocity/`（TensorBoard）。

查看 TensorBoard：

```bash
tensorboard --logdir logs/rsl_rl/xgomini_velocity
```
