# Rider2 与 Tars

| 任务 | 上游 | 执行方式 |
| --- | --- | --- |
| `Rider2-Balance-Flat` | [xgorider2_rl](https://github.com/LuwuDynamics/xgorider2_rl) | MuJoCo 3.16、SB3 PPO、C++ 控制器 |
| `Tars-Getup` | [rig_tars_rl](https://github.com/LuwuDynamics/rig_tars_rl) | MuJoCo 3.17、原版 PyTorch PPO、C 控制器仿真 |

两个任务位于 `src/tasks/`，使用 Git 子模块固定各自的提交。两个任务通过 luwu_mjlab CLI 调用原生 CPU 流程，使用上游模型、奖励、观测、训练算法与权重格式。项目原有四足任务使用 mjlab 1.6.0 和 MuJoCo Warp。

## 环境

现有训练服务器上的独立环境已装在项目同级的 `venvs/mjlab`、`venvs/rider` 和 `venvs/tars`。在项目根目录运行下面的命令。

```bash
PY="$PWD/../venvs/mjlab/bin/python"
"$PY" scripts/list_envs.py
```

新服务器先执行 `git submodule update --init`。为两个上游仓库各用 Python 3.11 创建独立环境并安装各自的 `requirements.txt`。可把环境建在上游的训练目录下，命名为 `.venv`，也可用 `LUWU_RIDER_PYTHON` 或 `LUWU_TARS_PYTHON` 指向相应解释器。Rider 需要 `clang++`，Tars 固件闭环评估需要 `cc`。主环境使用项目根目录的 `pip install -e .` 安装。

## Rider2

以下训练命令使用上游发布配方。评估可指定 SB3 checkpoint，原生窗口可读取上游发布或训练后导出的 C 权重。窗口需要 X11/Wayland。

```bash
RIDER="$PWD/src/tasks/rider2/training"
RUN="$PWD/logs/rider2"
"$PY" scripts/train.py Rider2-Balance-Flat --output "$RUN" \
  --envs 5 --steps 4000000 --eval-every 500000 --eval-seeds 8 \
  --eval-workers 5 --corner-fraction 0.5 --seed 7171
"$PY" scripts/native_task.py evaluate Rider2-Balance-Flat \
  --policy ppo "$RUN/last.zip" --scenarios stand stand_push drive
"$PY" scripts/play.py Rider2-Balance-Flat --model "$RIDER/artifacts/balance-v6/policy_weights.h"
"$PY" scripts/native_task.py debug Rider2-Balance-Flat
```

## Tars

以下命令从上游第二阶段 checkpoint 开始训练起身策略，然后评估并生成仰躺起身的回放图。评估也可使用上游第三阶段的 `checkpoints/getup_ppo_c3/final_it500.pt`。

```bash
TARS="$PWD/src/tasks/tars/tars_rl"
RUN="$PWD/logs/tars-getup"
"$PY" scripts/train.py Tars-Getup --init "$TARS/checkpoints/getup_ppo_c2/final_it600.pt" \
  --out "$RUN" --iters 500 --seed 0 --lr 0.00015 --lr_max 0.0003 \
  --reset_std 1 --log_std0 -2.0 --print_every 10 \
  --env_kw '{"reward_mode":"v2","continue_after_success":false,"act_pen":0.3,"obs_q_mode":"cmd","imu_clip":true,"vel_cap":7.5,"vel_pen":0.03,"p_weak":0.25}'
"$PY" scripts/native_task.py evaluate Tars-Getup --policy "$RUN/last.pt" \
  --no-keyframes --plants cal --kinds back,front --n 4 --procs 2
MUJOCO_GL=egl "$PY" scripts/play.py Tars-Getup --policy "$RUN/last.pt" --out "$RUN/replay.png"
"$PY" scripts/native_task.py debug Tars-Getup --policy "$RUN/last.pt" --n 4 --procs 2
"$PY" scripts/native_task.py firmware Tars-Getup --modes alt --plants cal --kinds back,front --n 4 --procs 2
```

入口把参数原样传给上游脚本，并在上游训练目录运行。给 checkpoint、日志和输出文件传绝对路径即可从项目根目录定位。通过对应入口的 `--help` 可查看上游参数。
