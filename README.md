# Handy Embodied Intelligence

一个面向新手的 MuJoCo + Gymnasium + Stable-Baselines3 教学项目：从零搭建一个“自动扶正平板机器人”，支持键盘示教、行为克隆初始化、PPO 强化学习训练、TensorBoard 可视化和权重回放。

完整教程见 [docs/tutorial_zh.md](docs/tutorial_zh.md)。

## 项目目标

机器人由一个平板和两节串联舵机杆组成。策略需要在倒地后控制两个舵机，让平板扶正，并在站稳后让杆尽量伸直、减少抖动。

策略输入只使用真实硬件上容易获得的数据：

```text
[
  sin(theta_x), cos(theta_x),
  sin(theta_y), cos(theta_y),
  omega_x, omega_y,
  joint_x, joint_y
]
```

也就是 IMU 姿态/角速度和两个舵机角度。杆尖位置、接触状态、接触力等仿真特权信息只用于 reward/debug，不喂给策略网络。

## 目录

```text
config.py                     # 机械、控制、reset、reward 集中配置
env/self_righting_env.py      # Gymnasium 环境
train.py                      # PPO 训练入口
evaluate.py                   # 加载权重并打开 MuJoCo viewer
scripts/view_model.py         # 键盘遥控环境
scripts/record_demo.py        # 键盘示教并保存 obs/action 数据
scripts/inspect_demo.py       # 查看示教数据质量
scripts/train_bc.py           # 行为克隆预训练 PPO policy
scripts/latest_checkpoint.py  # 找最新 checkpoint
scripts/physics_smoke_tests.py
requirements.txt
docs/tutorial_zh.md
```

训练输出默认写到 `runs/`，示教数据默认写到 `data/demos/`。这些文件不会提交到 Git。

## 安装

```bash
conda create -n RL python=3.11 -y
conda activate RL
pip install -r requirements.txt
```

如果要安装 GPU 版 PyTorch，按你的显卡驱动选择对应 wheel。例如 CUDA 12.6：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

## 快速运行

验证 MuJoCo 模型：

```bash
python scripts/physics_smoke_tests.py
```

打开键盘遥控：

```bash
python scripts/view_model.py --tilt-deg 85
```

启动从零 PPO：

```bash
python train.py ^
  --total-timesteps 3000000 ^
  --save-path runs/ppo_self_righting_scratch ^
  --checkpoint-dir runs/checkpoints_scratch ^
  --checkpoint-freq 10000 ^
  --device cuda
```

默认 actor/critic 都是 `512 512 512`。想实验小网络可以加：

```bash
python train.py --hidden-sizes 64 64
```

打开 TensorBoard：

```bash
tensorboard --logdir runs/logs/tensorboard --host 127.0.0.1 --port 6006
```

加载最新 checkpoint：

```bash
python scripts/latest_checkpoint.py runs/checkpoints_scratch
python evaluate.py --model-path runs/checkpoints_scratch/ppo_self_righting_100000_steps.zip --episodes 30 --tilt-deg 85
```

## 示教 + BC + PPO

录制示教：

```bash
python scripts/record_demo.py --output data/demos/manual_demo.npz --min-tilt-deg 55 --tilt-deg 85
```

示教按键：

- `A/D`：减小/增大 `joint_x`
- `W/S`：增大/减小 `joint_y`
- `C`：舵机目标回中
- `K`：保存当前 episode 并 reset
- `X` 或 `R`：丢弃当前 episode 并 reset
- `Space`：暂停/继续

检查示教数据：

```bash
python scripts/inspect_demo.py --demo-path data/demos/manual_demo.npz
```

行为克隆预训练：

```bash
python scripts/train_bc.py ^
  --demo-path data/demos/manual_demo.npz ^
  --save-path runs/ppo_self_righting_bc_init.zip ^
  --epochs 1500 ^
  --batch-size 512 ^
  --device cuda
```

从 BC 权重继续 PPO：

```bash
python train.py ^
  --load-model-path runs/ppo_self_righting_bc_init.zip ^
  --total-timesteps 3000000 ^
  --save-path runs/ppo_self_righting_bc_ppo ^
  --checkpoint-dir runs/checkpoints_bc_ppo ^
  --checkpoint-freq 10000 ^
  --device cuda
```

## 关键设计

- 物理频率：500 Hz
- 舵机 PD：200 Hz
- 策略频率：约 30 Hz
- 默认 episode：8 s
- 动作：两个舵机目标角，归一化到 `[-1, 1]`
- `joint_x` 限位：`±78 deg`
- `joint_y` 限位：`±115 deg`
- 默认网络：actor/critic 各 3 层 512 hidden，总参数约 1.06M

更多 reward、success、调参和教学解释见 [docs/tutorial_zh.md](docs/tutorial_zh.md)。
