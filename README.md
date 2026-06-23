# luwu_mjlab

**luwu_mjlab** is the reinforcement learning training environment for [Luwu Dynamics (陆吾智能)](https://www.xgorobot.com/), built to train and evaluate locomotion policies for quadruped robots (XGOMini) in MuJoCo simulation.

The project is based on **[mjlab](https://github.com/mujocolab/mjlab)** v1.2.0, which combines Isaac Lab-style APIs with GPU-accelerated MuJoCo Warp. luwu_mjlab extends mjlab with Luwu robot models and task configurations.

## Demo

| Simulation | Real Robot |
|------------|------------|
| ![xgomini walk sim](doc/gif/xgomini_walk_sim.gif) | ![xgomini walk real](doc/gif/xgomini_walk_real.gif) |
| ![xgomini getup sim](doc/gif/xgomini_getup_sim.gif) | ![xgomini getup real](doc/gif/xgomini_getup_real.gif) |

## Installation

### Linux (Ubuntu 22.04)

Requirements: NVIDIA GPU + driver 550+ + Python 3.11.

```bash
conda create -n luwu_mjlab python=3.11
conda activate luwu_mjlab
git clone https://github.com/LuwuDynamics/luwu_mjlab.git
cd luwu_mjlab
pip install -e .
```

Verify:

```bash
python scripts/list_envs.py --keyword XGOMini
python scripts/play.py XGOMini-Flat --agent zero
```

Train:

```bash
python scripts/train.py XGOMini-Flat --env.scene.num-envs=4096
tensorboard --logdir logs/rsl_rl/xgomini_velocity
```

### Windows

Requirements: Windows 10/11 + NVIDIA GPU (≥RTX 20xx) + ≥16 GB RAM.

**1. Install NVIDIA driver**

Requires ≥ R570. Download from https://www.nvidia.com/en-us/drivers/ , choose Express Installation, reboot.

```powershell
nvidia-smi   # Verify: should show driver version and CUDA UMD ≥13.0
```

**2. Install Miniconda**

Download from https://docs.anaconda.com/miniconda/install/#windows-installers , check "Just Me" + "Add to PATH". Open a new terminal and verify with `conda --version`.

**3. Create environment + CUDA + PyTorch + project**

```cmd
conda create -n luwu_mjlab python=3.11
conda activate luwu_mjlab
conda install -c nvidia cuda-toolkit=13.0.2 cudnn
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
cd <project_root>
pip install -e .
```

Verify CUDA: `nvcc --version` (should output release 13.0).
Verify PyTorch: `python -c "import torch; print(torch.cuda.is_available())"` (should output True).

**4. Smoke test**

```cmd
python scripts/play.py XGOMini-Flat --agent zero --num-envs 4
```

Use `--viewer viser` for web-based viewer when no display is available.

> **warp-lang must be <1.14**, as mjlab 1.2.0 depends on `wp.context.runtime` which was removed in newer versions.

Common issues:
- CUDA not available → check driver + `nvidia-smi`
- warp initialization failed → `pip show warp-lang` to confirm version 1.12.1
- DLL load failed → ensure `%CONDA_PREFIX%\Library\bin` is early in PATH

Clean reinstall:

```cmd
conda deactivate && conda remove -n luwu_mjlab --all
conda create -n luwu_mjlab python=3.11 && conda activate luwu_mjlab
conda install -c nvidia cuda-toolkit=13.0.2 cudnn
pip install torch==2.12 torchvision --index-url https://download.pytorch.org/whl/cu130
cd <project_root> && pip install -e .
```

### Google Colab

Colab provides free T4 GPU — no local setup needed for training and visualization.

**1. New notebook**

Open [Google Colab](https://colab.research.google.com/), set runtime to **T4 GPU**.

Or use the pre-configured notebook: [luwu_mjlab_colab.ipynb](https://colab.research.google.com/drive/1JrrJyGF9LYSPqhgQ5k89SmClbh30MRkM?usp=sharing)

**2. Install**

```python
!nvidia-smi
!git clone https://github.com/LuwuDynamics/luwu_mjlab.git
%cd luwu_mjlab
!ls
!pip install -e .
```

After installation completes, restart the runtime (Runtime → Restart and run all, or Ctrl+M .).

**3. Background training + TensorBoard**

```python
%cd luwu_mjlab
!python scripts/train.py XGOMini-Flat --env.scene.num-envs=4096 > output.log 2>&1 &

%load_ext tensorboard
%tensorboard --logdir ./logs/rsl_rl/xgomini_velocity
```

## Training

```bash
python scripts/train.py <task_id> [--overrides...]

Example:
python scripts/train.py XGOMini-Flat --env.scene.num-envs 4096
```

Two-stage tyro CLI: first argument selects a task, remaining arguments override `TrainConfig` fields (a frozen dataclass holding `env` + `agent`).

Execution flow: `main() → launch_training() → run_train()`. Single-GPU runs directly, multi-GPU uses `torchrunx`.

Common options:

| Option | Description |
|---|---|
| `--env.scene.num-envs 4096` | Number of parallel environments |
| `--agent.max-iterations 10000` | Training iterations |
| `--agent.resume` | Resume from latest checkpoint (regex match) |
| `--motion-file <path>` | Required for tracking tasks |
| `--video` | Record training videos |
| `--gpu-ids all` / `[0,1]` | GPU selection |

## Evaluation

```bash

# Load latest checkpoint
python scripts/play.py XGOMini-Flat

The simulation can be viewed in a browser at http://localhost:8080
╭────── viser (listening *:8080) ───────╮
│             ╷                         │
│   HTTP      │ http://localhost:8080   │
│   Websocket │ ws://localhost:8080     │
│             ╵                         │
╰───────────────────────────────────────╯

# Load a specific checkpoint
python scripts/play.py XGOMini-Flat \
  --checkpoint-file logs/rsl_rl/xgomini_velocity/2026-xx-xx_xx-xx-xx/model_xx.pt

# MDP sanity check (zero actions, no checkpoint needed)
python scripts/play.py XGOMini-Flat --agent zero
```

## Project Structure

```
scripts/
  train.py              # Training entry point (RSL-RL PPO)
  play.py               # Inference / evaluation entry point
  list_envs.py          # List registered tasks
  visualize_terrain.py  # Terrain preview

src/
  assets/robots/xgomini/  # MJCF XML, STL/DAE meshes, robot config constants
  assets/motions/          # Reference motion files for tracking tasks (.npz)
  tasks/
    __init__.py            # Auto-registers tasks via import_packages()
    velocity/              # Velocity-tracking locomotion task
      velocity_env_cfg.py           # make_velocity_env_cfg() factory
      config/xgomini/
        env_cfgs.py                 # rough / flat environment configs
        rl_cfg.py                   # PPO hyperparameters
      mdp/                          # observations, rewards, terminations, curriculums, commands
      rl/runner.py                  # VelocityOnPolicyRunner (ONNX export)
    tracking/              # Motion-imitation tracking task (mirrors velocity/ structure)
```

## Key Dependencies

Python **3.11**, versions pinned in `setup.py`:

| Package | Version | Description |
|---|---|---|
| `mjlab` | 1.2.0 | RL environment framework |
| `mujoco` | 3.6.0 | Physics engine |
| `mujoco-warp` | 3.6.0 | GPU-accelerated MuJoCo |
| `warp-lang` | 1.12.1 | GPU computing DSL (must be <1.14) |
| `scipy` | ≥1.17.0 | Scientific computing |

## Logging & Checkpoints

- Path: `logs/rsl_rl/<experiment_name>/<timestamp>/`; checkpoints: `model_<iteration>.pt`
- Config YAMLs written to `params/` (skipped when resuming)
- Default runner: `MjlabOnPolicyRunner`; `VelocityOnPolicyRunner` adds ONNX export + wandb

## Links

- [Luwu Dynamics website](https://www.xgorobot.com/)
- [mjlab repository](https://github.com/mujocolab/mjlab)
- [mjlab documentation](https://mujocolab.github.io/mjlab/)
