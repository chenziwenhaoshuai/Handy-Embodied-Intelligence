# Handy Embodied Intelligence 教学教程

这份教程面向第一次接触 MuJoCo、Gymnasium 和强化学习的新手。目标不是只给一份能跑的代码，而是把“仿真环境、手动示教、行为克隆、PPO 训练、调参观察”串成一个完整的教学流程。

## 1. 这个项目在做什么

我们做的是一个自动扶正平板机器人：

- 平板倒在地上；
- 平板上有两节串联舵机杆；
- 策略控制两个舵机；
- 目标是让平板站起来，并在站稳后让杆伸直、减少抖动。

它适合用来学习 embodied intelligence 的基本闭环：

```text
传感器 observation -> 策略 policy -> 舵机 action -> 物理世界 -> 新 observation
```

## 2. 代码结构

```text
config.py
```

集中保存机械尺寸、质量、舵机限位、控制频率、reward 权重、episode 时长等参数。新手调参时优先看这里。

```text
env/self_righting_env.py
```

Gymnasium 环境。这里定义了：

- MuJoCo 模型加载；
- reset 随机倒地；
- step 动作执行；
- observation；
- reward；
- success 判断。

```text
train.py
```

PPO 训练入口。可以从零训练，也可以用 `--load-model-path` 从已有 checkpoint 继续训练。

```text
evaluate.py
```

加载 PPO 权重并打开 MuJoCo viewer，可视化模型效果。

```text
scripts/record_demo.py
```

键盘示教工具。你手动控制机器人，按 `K` 保存当前 episode，得到 `(observation, action)` 数据。

```text
scripts/train_bc.py
```

行为克隆。把你的示教动作拟合成一个 PPO policy 初始化权重。

## 3. 安装环境

建议使用 conda：

```bash
conda create -n RL python=3.11 -y
conda activate RL
pip install -r requirements.txt
```

如果你想用 GPU 版 PyTorch，需要安装和显卡驱动匹配的 wheel。例如：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

验证：

```bash
python scripts/physics_smoke_tests.py
```

看到多行 `PASS` 就说明 MuJoCo 模型和环境基本可用。

## 4. 机器人模型

坐标系：

- `X`：前后
- `Y`：左右
- `Z`：向上

机构：

- `plate`：平板主体；
- `joint_x`：底部舵机，绕平板局部 X 轴旋转；
- `lower_rod`：下节杆；
- `joint_y`：第二个舵机，绕局部 Y 轴旋转；
- `upper_rod`：上节杆；
- `rod_tip`：杆尖碰撞球。

关键默认参数：

- 平板：`0.18 m x 0.12 m x 0.024 m`
- 平板质量：`0.28 kg`
- 下节杆长：`0.16 m`
- 上节杆长：`0.25 m`
- `joint_x`：`±78 deg`
- `joint_y`：`±115 deg`
- 舵机力矩限制：`±3.5 N*m`

这些都在 `config.py` 里。

## 5. 控制链路

策略不是直接输出力矩，而是输出两个归一化目标角：

```text
action = [target_joint_x, target_joint_y]
```

范围是：

```text
[-1, 1]^2
```

环境内部会做映射：

```text
[-1, 1] -> 实际舵机角度范围
```

再经过：

```text
目标角速度限制 -> 200 Hz PD 控制器 -> torque motor -> MuJoCo dynamics
```

这样更接近真实舵机控制，而不是让 RL 直接输出不现实的瞬时力矩。

## 6. 策略输入

当前策略输入是 8 维，全部是现实中更容易获得的信息：

```text
[
  sin(theta_x), cos(theta_x),
  sin(theta_y), cos(theta_y),
  omega_x, omega_y,
  joint_x, joint_y
]
```

解释：

- `theta_x/theta_y`：IMU 姿态倾角；
- `omega_x/omega_y`：IMU 角速度；
- `joint_x/joint_y`：两个舵机当前角度。

为什么不用 `tip_x/tip_y/tip_z`？

因为真实机器人通常没有杆尖世界坐标，也没有 MuJoCo 的完整 body pose。训练时如果用了这些“仿真特权信息”，模型在仿真里可能学得很好，但部署到真实硬件会缺输入。

所以本项目把杆尖位置、接触力、完整姿态等信息只放在 reward/debug 里，不喂给策略网络。

## 7. 手动遥控

先用键盘感受机器人是否能被人手动扶正：

```bash
python scripts/view_model.py --tilt-deg 85
```

按键：

- `A/D`：控制 `joint_x`
- `W/S`：控制 `joint_y`
- `C`：舵机目标回中
- `R`：reset
- `Space`：暂停/继续

如果人手动也完全扶不起来，通常先不要训练 RL，而是先检查：

- 平板是否太重或太轻；
- 杆是否太短或太轻；
- 舵机力矩是否太小；
- 舵机限位是否不够；
- 摩擦参数是否不合理。

## 8. 手动示教打标注

示教的目标是给模型一些“人类知道可行”的动作轨迹，让模型不必完全靠随机探索。

启动：

```bash
python scripts/record_demo.py --output data/demos/manual_demo.npz --min-tilt-deg 55 --tilt-deg 85
```

按键：

- `A/D`：控制 `joint_x`
- `W/S`：控制 `joint_y`
- `C`：回中
- `K`：保存当前 episode 并 reset
- `X` 或 `R`：丢弃当前 episode 并 reset
- `Space`：暂停/继续

建议：

- 只保存你觉得有教学价值的 episode；
- 失败、乱摆、明显错误的轨迹不要保存；
- 长边接地和短边接地都录一些；
- 至少录 20 条以上，越多越稳；
- 如果某一类姿态模型学不会，就多录那类姿态。

检查示教数据：

```bash
python scripts/inspect_demo.py --demo-path data/demos/manual_demo.npz
```

重点看：

- `episodes` 是否够多；
- `actions` 是否覆盖两个舵机；
- `action_min/action_max` 是否有足够范围；
- episode 长度是否合理。

## 9. 行为克隆 BC

BC 的任务很简单：给定 observation，让策略输出尽量接近你的键盘 action。

训练：

```bash
python scripts/train_bc.py ^
  --demo-path data/demos/manual_demo.npz ^
  --save-path runs/ppo_self_righting_bc_init.zip ^
  --epochs 1500 ^
  --batch-size 512 ^
  --device cuda
```

训练日志里关注：

```text
mean_abs_error
```

这个值表示预测动作和你示教动作的平均绝对误差。比如 `0.01` 表示在 `[-1, 1]` action 空间里平均差约 1%。

BC 只是在模仿你，不会自动超过你。它的价值是给 PPO 一个更好的起点。

## 10. PPO 强化学习

从零训练：

```bash
python train.py ^
  --total-timesteps 3000000 ^
  --save-path runs/ppo_self_righting_scratch ^
  --checkpoint-dir runs/checkpoints_scratch ^
  --checkpoint-freq 10000 ^
  --device cuda
```

从 BC 初始化继续训练：

```bash
python train.py ^
  --load-model-path runs/ppo_self_righting_bc_init.zip ^
  --total-timesteps 3000000 ^
  --save-path runs/ppo_self_righting_bc_ppo ^
  --checkpoint-dir runs/checkpoints_bc_ppo ^
  --checkpoint-freq 10000 ^
  --device cuda
```

从某个 checkpoint 继续训练：

```bash
python train.py ^
  --load-model-path runs/checkpoints_bc_ppo/ppo_self_righting_100000_steps.zip ^
  --total-timesteps 3000000 ^
  --save-path runs/ppo_self_righting_continue ^
  --checkpoint-dir runs/checkpoints_continue ^
  --checkpoint-freq 10000 ^
  --device cuda
```

默认网络是 actor/critic 各 3 层 512 hidden，总参数约 1.06M。你也可以实验不同网络规模：

```bash
python train.py --hidden-sizes 64 64
python train.py --hidden-sizes 512 512 512
```

注意：从已有 checkpoint 继续训练时，网络结构由 checkpoint 决定，`--hidden-sizes` 只对从零创建模型生效。

## 11. TensorBoard

启动：

```bash
tensorboard --logdir runs/logs/tensorboard --host 127.0.0.1 --port 6006
```

常看指标：

- `rollout/ep_rew_mean`：平均 episode reward；
- `rollout/success_rate`：Monitor 记录的成功比例；
- `train/entropy_loss`：策略探索程度；
- `train/std`：连续动作分布的标准差；
- `train/approx_kl`：每轮策略更新幅度；
- `train/explained_variance`：value function 拟合情况。

注意：只要改了 reward，`ep_rew_mean` 的绝对数值就不能和旧 run 直接比较。此时更应该看 viewer 里的动作质量和 success。

## 12. 加载权重看效果

找最新 checkpoint：

```bash
python scripts/latest_checkpoint.py runs/checkpoints_bc_ppo
```

加载：

```bash
python evaluate.py --model-path runs/checkpoints_bc_ppo/ppo_self_righting_100000_steps.zip --episodes 30 --tilt-deg 85
```

如果想打印动作和舵机调试信息：

```bash
python evaluate.py --model-path runs/checkpoints_bc_ppo/ppo_self_righting_100000_steps.zip --debug-every 10
```

## 13. Reward 设计

reward 目前由这些部分组成：

- `upright_weight`：鼓励平板法向朝上；
- `upright_progress_weight`：鼓励从倒地向站立进步；
- `rod_upright_weight`：鼓励上节杆朝上；
- `joint_center_weight`：鼓励舵机回中；
- `stand_joint_center_weight`：站起来后更强地鼓励回中；
- `action_magnitude_weight`：减少无意义大动作；
- `action_change_weight`：减少抖动；
- `stand_joint_velocity_weight`：站起来后减少舵机速度；
- `stand_angular_velocity_weight`：站起来后减少底板晃动；
- `stillness_bonus_weight`：站起来后越稳定奖励越高；
- `success_bonus`：满足成功条件并保持一段时间给奖励。

一个重要技巧是 `stand_gate`：

```text
倒地时 stand_gate 接近 0
接近站立时 stand_gate 接近 1
```

很多“稳定、回中、少抖动”的约束只在 `stand_gate` 高时生效。这样模型倒地时仍然敢大幅摆杆，站起来后才被要求安静。

## 14. Success 判定

默认成功条件：

- `abs(theta_x) < 7 deg`
- `abs(theta_y) < 7 deg`
- `abs(omega_x), abs(omega_y) < 0.35 rad/s`
- `abs(joint_x), abs(joint_y) < 10 deg`
- 上节杆方向距离世界竖直小于 `12 deg`
- 连续保持 `0.35 s`

这比“某一瞬间站起来”严格，因为真实机器人需要稳定站住。

## 15. 常见问题和调参方向

### 训练总是短边成功，长边失败

原因通常是短边更容易，PPO 优先学到局部最优。

处理：

- 多录长边示教；
- reset 过采样长边姿态；
- 降低倒地阶段 action penalty；
- 增加探索；
- 从长边示教 BC 初始化。

### 站起来后杆在 ±10 度抖动

处理：

- 增大 `stand_joint_velocity_weight`；
- 增大 `stand_angular_velocity_weight`；
- 增大 `action_change_weight`；
- 增大 `stillness_bonus_weight`；
- 注意不要在倒地阶段过早限制动作。

### 训练 reward 变高但 viewer 看起来不好

说明 reward 可能被模型钻空子。处理方式：

- 把成功条件写严格；
- 删除不真实的 reward hack；
- 用 viewer 观察多个随机 reset；
- 单独统计不同倒地方向的成功率。

### BC 训练误差很低，但 PPO 初期 success 还是 0

PPO 训练时动作是随机采样，不是 deterministic 复现 BC，所以初期 success 可能仍然低。看趋势，不要只看第一分钟。

### 想接真实机器人

优先保证策略输入只包含真实硬件可得数据。本项目当前输入已经限制为：

- IMU 姿态/角速度；
- 两个舵机角度。

真实部署前还需要考虑：

- IMU 噪声和延迟；
- 舵机角度反馈误差；
- 电池电压变化；
- 摩擦和质量偏差；
- 舵机速度/力矩限制；
- 安全限位。

可以在 `config.py` 的 `IMUConfig` 中逐步增加 noise/bias/delay 做 domain randomization。

## 16. 推荐实验顺序

1. `physics_smoke_tests.py` 确认物理模型能跑。
2. `view_model.py` 确认人手动能扶正。
3. `record_demo.py` 录 20-50 条有效轨迹。
4. `inspect_demo.py` 检查示教质量。
5. `train_bc.py` 做 BC 初始化。
6. `evaluate.py` 看 BC 是否复现人类动作。
7. `train.py --load-model-path ...` 从 BC 继续 PPO。
8. TensorBoard 看趋势。
9. 每隔一段加载 checkpoint 看 viewer。
10. 根据失败模式调 reward/reset/机械参数。

## 17. 推送仓库注意

仓库只提交代码、文档和轻量资源。

不会提交：

- `runs/`
- checkpoint `.zip`
- TensorBoard logs
- `data/demos/*.npz`
- Python 缓存

这样仓库适合作为教学框架，训练结果由每个学习者本地生成。

## 18. 扩展任务：直立遥控走路

自动扶正学会以后，可以把任务换成“直立起步，然后按遥控指令移动”。这个任务对应：

```text
env/walking_env.py
train_walking.py
evaluate_walking.py
scripts/walking_smoke_tests.py
```

### 18.1 WalkingEnv 和 SelfRightingEnv 的区别

`SelfRightingEnv`：

- reset 时随机倒地；
- reward 主要鼓励站起来、伸直、稳定；
- episode 成功后终止。

`WalkingEnv`：

- reset 时从直立状态开始；
- 输入里额外加入遥控速度指令和 gait phase；
- reward 鼓励平面速度跟踪指令；
- 如果明显倒下则终止。

walking 策略输入是 12 维：

```text
[
  sin(theta_x), cos(theta_x),
  sin(theta_y), cos(theta_y),
  omega_x, omega_y,
  joint_x, joint_y,
  command_vx, command_vy,
  sin(phase), cos(phase)
]
```

其中：

- 前 8 维仍然是可部署传感器信息；
- `command_vx/command_vy` 是遥控器给的速度指令；
- `phase` 是控制器内部的周期信号，相当于给两舵机一个可学习的节律参考。

注意：平板真实位置和速度不喂给 policy，只在仿真 reward 中使用。

### 18.2 训练 walking

先做 smoke test：

```bash
python scripts/walking_smoke_tests.py
```

从零训练：

```bash
python train_walking.py ^
  --total-timesteps 3000000 ^
  --save-path runs/ppo_walking ^
  --checkpoint-dir runs/checkpoints_walking ^
  --checkpoint-freq 10000 ^
  --device cuda
```

可调参数：

```bash
python train_walking.py --command-speed 0.05
python train_walking.py --command-speed 0.10
python train_walking.py --command-change-seconds 3.0
python train_walking.py --hidden-sizes 256 256
```

建议先用小速度，例如 `0.05 m/s` 或 `0.08 m/s`。如果速度指令太大，两个舵机可能很难产生稳定位移。

### 18.3 可视化遥控

训练出 checkpoint 后：

```bash
python evaluate_walking.py --model-path runs/checkpoints_walking/ppo_walking_100000_steps.zip
```

按键：

- `W/S`：前进/后退；
- `A/D`：左移/右移；
- `C`：停止；
- `R`：reset；
- `Space`：暂停。

### 18.4 walking reward

walking reward 的核心是速度跟踪：

```text
xy_velocity 接近 command_velocity -> 奖励高
```

同时保留：

- 直立奖励；
- 倾角惩罚；
- 角速度惩罚；
- 舵机速度惩罚；
- action 幅度和变化惩罚；
- 倒下终止惩罚。

### 18.5 重要限制

当前机器人只有两个舵机，且没有轮子或腿。它的“走路”更接近通过摆杆和地面摩擦产生小范围挪动，不应期待像四足机器人一样稳定、快速地行走。

如果训练不出明显位移，优先尝试：

- 降低 `--command-speed`；
- 增加 episode 时长；
- 调整杆尖和平板摩擦；
- 增大舵机力矩限制；
- 加入更多可控自由度；
- 设计专门的节律动作或示教轨迹。
