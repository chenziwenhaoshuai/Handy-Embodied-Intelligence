# 第三方来源与许可说明

## 1. 上游项目

本项目的仿真环境、MuJoCo 模型与评估标准来自：

- 项目：`SelfRisingRobot` — 2 自由度起立机器人 "poco"
- 作者：homemadegarbage
- 仓库：https://github.com/homemadegarbage/SelfRisingRobot
- 配套文章：https://homemadegarbage.com/rl13

具体使用了以下文件（**未做任何修改**，仅调整了读取路径）：

| 本仓库路径 | 上游路径 | 说明 |
| --- | --- | --- |
| `rl/env/robo1.xml` | `RL/robo1.xml` | MuJoCo 机器人模型 |
| `rl/env/robo1_env.py` | `RL/robo1_env.py` | Gymnasium 环境（观测/奖励/达标判定） |
| `rl/env/assets/*.stl` | `RL/assets/*.stl` | 模型引用的网格 |
| `rl/eval_robo1_policy.py` | `RL/eval_robo1_policy.py` | 四姿态评估标准 |
| `rl/play_robo1_policy.py` | `RL/play_robo1_policy.py` | 回放脚本 |

### ⚠️ 重要：上游没有许可证

截至 2026-09，上游仓库**没有 LICENSE 文件**。按照默认版权规则，这意味着
原作者保留所有权利，未经许可再分发这些文件是有法律风险的。

发布前请三选一：

1. **联系原作者取得许可**（最稳妥），并在 README 里注明授权方式；
2. **不随包分发上游文件**：把 `rl/env/` 从仓库里移除，改为在 README 里说明
   "请先从上游仓库获取 `robo1.xml` / `robo1_env.py` / `assets/`，放到 `rl/env/`"，
   或提供脚本让用户自行克隆；
3. 在得到许可前，把仓库设为**私有**或明确标注"仅供学习，不构成再分发授权"。

我们的建议是第 1 条：这类教学项目的作者通常乐意授权，只要你在 README 里
清楚注明来源和用途。

## 2. 本项目新增的部分

以下部分为原创，按 MIT 许可发布（见 `LICENSE`）：

- `rl/train_pure_rl.py` — 从零开始的纯 RL 训练（含奖励重塑与退火）
- `rl/watch_latest.py` — 实时可视化最新权重
- `rl/verify_pure_rl.py` — 标准姿态 + 摆放偏差鲁棒性终验
- `reproduce.py` — 一键 训练→验证→导出
- `docs/` — 训练报告与踩坑记录
- `firmware/` 中对上游固件的改造（IMU 机身坐标换算、`wifi_config.h` 配置化等）

## 3. 模型权重的归属

`rl/weights/eei_pure_rl_v1.zip` 与 `firmware/main/policy_network.h` 是本项目
**从零训练**得到的（训练只用 `rl/env/` 里的仿真环境和奖励函数，未使用上游
发布的任何权重）。它们按 MIT 许可发布；但如上所述，训练环境的版权归属上游，
再分发时请一并考虑这一点。
