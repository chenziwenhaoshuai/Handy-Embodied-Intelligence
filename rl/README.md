# rl/ — 强化学习部分

## 文件

| 文件 | 作用 |
| --- | --- |
| `train_pure_rl.py` | 从零训练的 PPO 主脚本（默认参数即制胜配方） |
| `watch_latest.py` | MuJoCo 窗口，自动加载 `ckpt/` 里最新的权重并循环播放四个姿态 |
| `verify_pure_rl.py` | 终验：四姿态达标 + 摆放偏差鲁棒性网格 |
| `export_policy_header.py` | 把权重导出成固件用的 C 头文件（上游脚本） |
| `eval_robo1_policy.py` | 上游评估标准（判定条件未改） |
| `play_robo1_policy.py` | 回放单个权重（MuJoCo 窗口） |
| `env/` | 上游 MuJoCo 模型 `robo1.xml`、环境 `robo1_env.py`、`assets/` |
| `weights/` | 训练好的权重 |
| `ckpt/`、`tb/` | 训练产物（已 gitignore） |

## 观测 / 动作 / 达标条件（沿用上游）

* 观测 `4`：`[roll, pitch, target1, target2]`（弧度）
* 动作 `2`：`[-1, 1]`，`target += action × 0.08`，限幅 ±1.55 rad
* 控制频率：`frame_skip 20 × timestep 0.001` = **50 Hz**，单回合上限 700 步（14 s）
* `goal_pose` 达标（需连续保持 50 步）：
  `upright > 0.95`、`|roll|, |pitch| < 0.25`、`max|servo| < 0.12`、
  `|foot_height_err| < 0.02`、`|qvel[0:6]| < 0.25`

## 训练输出

| 路径 | 内容 |
| --- | --- |
| `ckpt/<tag>_<N>_steps.zip` | 每 100k 步一个 checkpoint |
| `ckpt/<tag>_best.zip` | 评估达标数最高时自动保存 |
| `ckpt/<tag>_eval_history.json` | 每次评估的步数、达标数、四姿态明细 |
| `ckpt/<tag>_last.zip` | 训练结束时的权重 |
| `tb/<tag>_N/` | TensorBoard 事件（含 `eval/poses_ok` 自定义标量） |

## 常用命令

```powershell
python rl/train_pure_rl.py --steps 3000000 --seed 0 --tag my_run
python rl/train_pure_rl.py --reward author     # 用上游原始奖励做对比实验（会卡在局部最优）
python rl/train_pure_rl.py --init rl/ckpt/my_run_600000_steps.zip   # 从自己的 checkpoint 续训
python rl/train_pure_rl.py --eval-only rl/weights/eei_pure_rl_v1.zip
python rl/verify_pure_rl.py "--offsets=-10,-5,0,5,10"
```

> `--offsets` 的值以 `-` 开头时，PowerShell 下请用 `--offsets=...` 的写法。
