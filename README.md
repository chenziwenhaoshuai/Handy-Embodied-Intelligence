# Handy-Embodied-Intelligence

**一个两自由度（单足 + 双臂）小机器人的完整强化学习实践：用纯 RL 从零训练它从任意倒伏姿态站起来，
再把策略烧进一块 ESP32-S3 上跑。**

硬件与仿真模型基于开源项目 [SelfRisingRobot](https://github.com/homemadegarbage/SelfRisingRobot)（作者 homemadegarbage）。
我们把它的模型完整跑通、修掉了从零训练时会卡住的奖励陷阱，并整理成可复现的教学流程。

## 仓库结构

| 目录 | 内容 | 文档 |
| --- | --- | --- |
| `rl/` | **起立**：纯 RL（PPO）从随机初始化训到四方向都能站起来 | [rl/README.md](rl/README.md) |
| `firmware/` | **固件**：ESP32-S3（ESP-IDF），内置起立策略 + IMU 机身坐标换算 + 网页控制台 | [firmware/README.md](firmware/README.md)、[firmware/开发说明.md](firmware/开发说明.md) |
| `docs/` | 从零训练起立的完整实验记录（配方、两个奖励陷阱、全部实测数字） | [docs/训练报告.md](docs/训练报告.md) |



---

# 起立部分（EEI-RL）

**用纯强化学习从零训练，并把它烧进 ESP32-S3。**

这部分不依赖任何示教轨迹、动作脚本或预训练权重：PPO 从随机初始化开始，
只靠仿真环境里的奖励函数学会「从四个方向倒下的姿态都站起来，并把手臂收回零位」。

---

## 1. 结果

训练出的 `rl/weights/eei_pure_rl_v1.zip`（3M 步纯 PPO，无示教）实测：

| 验证项 | 结果 |
| --- | --- |
| 四个标准倒伏姿态（上游原始达标判定） | **4/4** |
| 摆放偏差网格 ±10°/±5°/0（每姿态 25 组，共 100 组） | **100/100** |
| 达标后舵机指令摆幅（峰峰值） | **0.27° – 1.12°**（旧版为 9°–13°，见 §4） |
| 达标后动作幅度 | 0.003 – 0.009（基本为零输出） |
| 首次 4/4 | 50 万步（从零、无示教） |
| 上机实测 | 已烧录 ESP32-S3，策略在板上 50 Hz 运行，能把 20° 倾斜拉平并保持直立 |

> 作为对照：上游作者发布的权重是 4/4 标准姿态 + 摆放偏差 35/36。

---

## 2. 目录

```
EEI-RL/
├── rl/                         # 强化学习
│   ├── train_pure_rl.py        # ★ 从零训练（默认参数就是能跑到 4/4 的配方）
│   ├── watch_latest.py         # MuJoCo 实时窗口：自动加载最新权重循环播放
│   ├── verify_pure_rl.py       # 终验：标准姿态 + 摆放偏差鲁棒性
│   ├── export_policy_header.py # 权重 → C 头文件
│   ├── eval_robo1_policy.py    # 上游评估标准（未改判定条件）
│   ├── play_robo1_policy.py    # 回放单个权重
│   ├── env/                    # 上游 MuJoCo 模型与环境（未改物理/奖励）
│   └── weights/                # 训练好的权重
├── firmware/                   # ESP32-S3 固件（ESP-IDF）
│   ├── main/main.c             # IMU 机身坐标换算 + 50Hz 策略任务 + HTTP/WS
│   ├── main/policy_network.h   # 由 rl/export_policy_header.py 生成
│   ├── main/wifi_config.example.h
│   └── 开发说明.md              # 固件完整设计文档（引脚、接口、标定）
├── docs/训练报告.md             # 配方、曲线、踩坑与调参记录
├── reproduce.py                # 一键：训练 → 验证 → 导出固件头文件
└── requirements.txt
```

---

## 3. 快速开始

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3.1 一键复现（推荐）

```powershell
python reproduce.py                 # 训练 3M 步 → 终验 → 导出 policy_network.h
python reproduce.py --steps 30000   # 只想先冒烟测一下链路
```

跑完会打印终验表格，并把 C 头文件直接写到 `firmware/main/policy_network.h`。

### 3.2 分步执行

```powershell
# 训练（默认 3M 步、5 个并行环境，约 20–40 分钟，取决于 CPU）
python rl/train_pure_rl.py

# 边训边看（另开一个窗口）：自动加载最新 checkpoint 循环播放四个姿态
python rl/watch_latest.py --speed 1.5

# 训练曲线
tensorboard --logdir rl/tb

# 终验
python rl/verify_pure_rl.py

# 导出到固件
python rl/export_policy_header.py rl/weights/eei_pure_rl_v1.zip -o firmware/main/policy_network.h
```

---

## 4. 为什么从零训练能成功（关键在奖励）

上游的奖励函数是为「先示教克隆、再 PPO 微调」设计的。**直接拿它从零训练会卡住**，
我们实测到过这个现象：四个姿态的直立度在 0.70 冻住 20 万步不动。

原因是奖励里 `servo_zero_cost / target_zero_cost / height_cost` 被一个门控乘上：

```
stand_gate = clip((upright - 0.65) / 0.35, 0, 1)
```

机器人一旦越过 0.65 的直立度、手臂还伸着，就要付约 3.2/步的代价，
而完全直立只给 2/步 —— **「撑起来但没收回手臂」是净负收益**。
于是策略学会卡在门的下方（直立度 ~0.70）白拿 `2·upright`，
永远不去够那个「收手臂才给 +5」的达标奖励。

`train_pure_rl.py` 的做法（只改奖励，物理与达标判定一字未动）：

1. **`gate_scale` 退火**：先用 0 让它专心学会「站起来」，再随训练进度升到 1，
   逼它把手臂收回、满足完整达标条件；
2. **门函数改成 smoothstep 且从 0.75 起**，去掉 0.65 处的收益悬崖；
3. **加 `6·upright⁴`**，把「0.7 直立」和「1.0 直立」的收益拉开。

结果：四个姿态在 10 万步就已经全部直立度 1.0，50 万步达到完整达标。

### 4.1 第二个陷阱：站稳之后舵机在抖

第一版跑通 4/4 后上机，发现机器人**站直以后舵机持续抽动**。打出轨迹看，
是达标状态下动作在 `±0.7` 之间来回跳、`target` 以约 5 Hz 摆动造成的
高增益抖振——原奖励对动作的惩罚只有 `0.005·Σa²`，而待在达标态每步有约 13 的收益，
抖动的代价可以忽略。

于是达标时追加两项：`+3.0·exp(−10·Σa²)`（越静止给得越多）与 `−1.5·Σa²`（直接抑制抖动）。

标定时有一条硬约束：**达标状态的收益必须始终为正**。达标本身给 `+5/步`，
抖动惩罚一旦超过它，策略就会宁愿卡在"直立但没达标"——实测惩罚设成 `6.0` 时
`roll_pos` 一直停在 `upright=0.99, servo=1.4` 进不去门。改成 `1.5` 后：
抖动时约 `+3.5/步`，完全静止时约 `+8.0/步`，方向明确又不破坏达标动机。

效果（同一套验证脚本，达标状态下的舵机指令摆幅）：

| 姿态 | 加静止奖励前 | 加静止奖励后 |
| --- | --- | --- |
| roll_pos | 12.41° | **0.27°** |
| roll_neg | 9.24° | **0.59°** |
| pitch_pos | 12.70° | **0.75°** |
| pitch_neg | 9.36° | **1.12°** |

细节与全部实验数据见 [`docs/训练报告.md`](docs/训练报告.md)。

---

## 5. 烧录到实机

```powershell
# 1. 准备 Wi-Fi 凭据（wifi_config.h 不进版本库）
copy firmware\main\wifi_config.example.h firmware\main\wifi_config.h
#    然后填上你的 SSID / 密码；连不上时板子会开热点 BMI160-Setup / 12345678

# 2. 编译
cd firmware
idf.py set-target esp32s3      # 首次
idf.py build

# 3. 烧录（端口按实际改）
idf.py -p COM8 flash monitor
```

上电行为：

* 板子连上 Wi-Fi 后，网页在 `http://<串口日志里的 IP>/`；
* 策略任务 50 Hz：`obs = [roll, pitch, target1, target2]` → `target += action × 0.08`（限幅 ±1.55）；
* 舵机指令 `angle = 90 − deg(target)`，输出到 GPIO8 / GPIO9；
* 默认 `auto` 模式：倾斜 > 43° 持续 0.3 s 自动开始起立，站直（< 14.3°）持续 2 s 停止，单次上限 30 s；
* 也可手动：`POST /getup?value=1|0|auto`。

固件的 IMU 安装矩阵、引脚定义、全部 HTTP 接口见 [`firmware/开发说明.md`](firmware/开发说明.md)。

> ⚠️ 发布前检查：`firmware/main/wifi_config.h` 里放的是**你自己的 Wi-Fi 密码**。
> 它已经在 `.gitignore` 里（`git push` 不会带上），但如果你是把整个文件夹打包上传，
> 请先删掉这个文件——别人拿到后从 `wifi_config.example.h` 复制一份填自己的即可。

---

## 6. 复现性

* 我们做过**三次独立的从零训练，全部跑到 4/4**；其中两次（含静止奖励的版本）
  验证结果逐项相同（4/4、100/100、摆幅 0.27°–1.12°），权重哈希不同但行为一致，
  说明是配方稳健收敛而非种子运气；
* 默认参数（`rl/train_pure_rl.py` 不加任何参数）就是实测能到 4/4 的配方；
* 随机种子固定为 `--seed 0`，环境按 rank 派生种子，CPU 训练确定性强；
* CPU 越少训练越慢，但**结果不受影响**（并行环境数改变会影响数据配比，
  想严格复现请保持默认 `--n-envs 5`）；
* 训练过程中每个 checkpoint 都会用上游原始判定做一次四姿态评估，
  结果存在 `rl/ckpt/<tag>_eval_history.json`，最佳权重自动存为 `<tag>_best.zip`。

---

## 7. 致谢与许可


仿真环境、MuJoCo 模型与评估标准来自 [homemadegarbage/SelfRisingRobot](https://github.com/homemadegarbage/SelfRisingRobot)，
请在使用和再分发前阅读 [`NOTICE.md`](NOTICE.md)：**上游目前没有 LICENSE 文件**，
再分发前建议先联系原作者取得许可。

本项目新增的代码按 MIT 发布，见 [`LICENSE`](LICENSE)。
