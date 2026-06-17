# luwu_mjlab

**luwu_mjlab** 是 [**陆吾智能（Luwu Dynamics）**](https://www.xgorobot.com/) 产品线的强化学习训练环境，用于在 MuJoCo 仿真中训练、验证四足等机器人的运动策略。

本项目基于开源框架 [**mjlab**](https://github.com/mujocolab/mjlab) 构建。mjlab 将 Isaac Lab 风格的高层 API 与 MuJoCo Warp GPU 仿真结合；luwu_mjlab 在其上扩展陆吾机器人模型与任务配置。

## 效果预览

| 仿真 | 实机 |
|------|------|
| ![xgomini walk sim](doc/gif/xgomini_walk_sim.gif) | ![xgomini walk real](doc/gif/xgomini_walk_real.gif) |
| ![xgomini getup sim](doc/gif/xgomini_getup_sim.gif) | ![xgomini getup real](doc/gif/xgomini_getup_real.gif) |

## 安装

```bash
conda create -n luwu_mjlab python=3.11
conda activate luwu_mjlab
cd luwu_mjlab
pip install -e .
```

详见 [doc/setup_zh.md](doc/setup_zh.md)。

## 训练（XGOMini 示例）

```bash
python scripts/train.py XGOMini-Flat --env.scene.num-envs=4096
```

## 验证

```bash
# MDP sanity check（无需 checkpoint）
python scripts/play.py XGOMini-Flat --agent zero

# 加载最新本地 checkpoint
python scripts/play.py XGOMini-Flat

# 指定 checkpoint
python scripts/play.py XGOMini-Flat \
  --checkpoint-file logs/rsl_rl/xgomini_velocity/2026-xx-xx_xx-xx-xx/model_xx.pt
```

## 相关链接

- [陆吾智能官网](https://www.xgorobot.com/)
- [mjlab 官方仓库](https://github.com/mujocolab/mjlab)
- [mjlab 文档](https://mujocolab.github.io/mjlab/)
