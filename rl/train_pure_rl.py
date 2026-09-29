r"""从零开始的纯强化学习：不采样任何示教数据，只用 PPO + 环境奖励。

与上游作者流程的区别（也是这个脚本存在的理由）：
  * 上游流程是 搜索轨迹 → 监督克隆 → PPO 微调；克隆阶段本身不含 RL。
  * 这里完全跳过前两步，PPO 从随机初始化开始，只靠 `robo1_env.py` 的奖励学起立。

环境与判定标准沿用上游原版（`env/robo1_env.py` / `eval_robo1_policy.py`），
一字未改；本脚本只负责向量化并行、域随机化、评估回调和日志。

默认参数就是本项目实测能跑到 4/4 的配方，直接运行即可：
    python rl/train_pure_rl.py

其它用法：
    python rl/train_pure_rl.py --seed 1                # 换种子
    python rl/train_pure_rl.py --reward author         # 用上游原始奖励（可对比）
    python rl/train_pure_rl.py --eval-only rl/weights/eei_pure_rl_v1.zip
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENV_DIR = (HERE / "env").resolve()
sys.path.insert(0, str(ENV_DIR))
os.chdir(ENV_DIR)  # eval_pose / Robo1GetupEnv 里用的是相对路径 "robo1.xml"
try:  # Windows 管道重定向时默认 GBK 会吞掉 emoji/中文
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback  # noqa: E402
from stable_baselines3.common.vec_env import (  # noqa: E402
    SubprocVecEnv,
    VecMonitor,
    VecNormalize,
)

from eval_robo1_policy import eval_pose  # noqa: E402
from eval_metrics import measure_steadiness  # noqa: E402
from robo1_env import FALLEN_POSES, Robo1GetupEnv, roll_pitch_to_quat  # noqa: E402

POSES = ("roll_pos", "roll_neg", "pitch_pos", "pitch_neg")


class JitteredEnv(Robo1GetupEnv):
    """作者环境的原样子类，只在重置时给倒伏姿态加一点随机扰动（域随机化）。

    物理、奖励、观测全都继承自作者原版，不做任何修改。
    """

    jitter_deg = 0.0

    def _set_fixed_fallen_pose(self, pose_name=None):
        super()._set_fixed_fallen_pose(pose_name)
        if self.jitter_deg <= 0.0:
            return

        roll, pitch = FALLEN_POSES[self.current_pose_name]
        j = math.radians(self.jitter_deg)
        roll += float(self.np_random.uniform(-j, j))
        pitch += float(self.np_random.uniform(-j, j))

        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.qpos[0:3] = [0.0, 0.0, 0.08]
        self.data.qpos[3:7] = roll_pitch_to_quat(roll, pitch)
        self.data.qpos[self.servo1_qpos_id] = 0.0
        self.data.qpos[self.servo2_qpos_id] = 0.0
        self.target[:] = 0.0
        self.data.ctrl[:] = self.target
        mujoco.mj_forward(self.model, self.data)

        for _ in range(self.settle_steps):
            self.data.ctrl[:] = self.target
            mujoco.mj_step(self.model, self.data)

        self.target[:] = 0.0
        self.data.ctrl[:] = self.target


class ShapedEnv(JitteredEnv):
    """为「从零学习」重写奖励的版本；物理/观测/达标判定完全沿用作者原版。

    为什么需要它：作者原奖励把 `servo_zero_cost / target_zero_cost / height_cost`
    乘上 `stand_gate = clip((upright-0.65)/0.35)`。从零开始学的时候，机器人一旦
    超过 0.65 直立度、手臂还伸着，就要付约 3.2/步的代价，而完全直立只有 2/步 ——
    "撑起来但没收回手臂"是净负收益，于是策略卡在门的下面（实测直立度冻在 0.70）。

    改动（只改奖励，不改物理）：
      1. `gate_scale`：手臂/目标/高度代价的系数，先用 0 让它先学会"站起来"，
         再随训练进度退火到 1，逼它把手臂收回、满足完整达标条件。
      2. 门的形状用 smoothstep 从 0.75 起，去掉 0.65 处的收益悬崖。
      3. 增加 `up_bonus · upright⁴`，把 0.7 直立和 1.0 直立拉开差距。
      4. **静止奖励**：达标（goal_pose）之后，动作越小、target 变化越小，给得越多。
         目的是消掉"站直之后舵机还在抖"的现象——那种抖动对任务没有任何好处，
         但原奖励里几乎没有对应的惩罚（`0.005·Σa²` 的量级太小）。
    """

    gate_scale = 0.0
    up_bonus = 6.0
    gate_start = 0.75
    # --- 静止奖励（达标后别乱动舵机） ---
    #
    # 实测现象：达标之后策略输出的是"高增益抖振"——action 在 ±0.9 之间来回跳，
    # target 以约 5 Hz 在 ±0.1 rad 摆动，实物上就是舵机不停抽动。原因是原奖励里
    # 对动作的惩罚只有 0.005·Σa²（量级 0.005），而待在达标状态每步有 ~13 的收益，
    # 抖动的代价可以忽略不计。下面三项就是把这个代价补上：
    #
    # 标定这三项时要守住一条：**达标状态的收益必须始终为正**。
    # 达标本身给 +5/步，如果抖动惩罚比它还大，策略会宁愿卡在"直立但没达标"的
    # 状态（实测过：hold_action_penalty=6.0 时 roll_pos 一直停在 up=0.99、servo=1.4）。
    hold_bonus = 3.0              # 达标时"完全静止"能拿到的额外奖励
    hold_sharp = 10.0             # 静止奖励的敏感度：exp(-sharp·Σa²)，越大越偏好"完全不动"
    hold_action_penalty = 1.5     # 达标时对 Σa² 的惩罚：抖动(Σa²≈1)扣 1.5/步，
                                  # 完全静止(Σa²≈0)几乎不扣，差值被 hold_bonus 放大到 ~4.5/步
    still_action_penalty = 0.0    # 默认关闭：直立段的持续动作是"把手臂收回来"所必需的，
                                  # 在这里加惩罚会拖住收臂过程（实测 0.5 会让 roll_pos 卡住）

    def reset(self, *args, **kwargs):
        obs, info = super().reset(*args, **kwargs)
        self._prev_up = self._upright()
        self._prev_target = self.target.copy()
        return obs, info

    def step(self, action):
        prev_target = self.target.copy()
        obs, _reward, terminated, truncated, info = super().step(action)

        up = float(info["upright"])
        d_up = up - self._prev_up
        self._prev_up = up
        roll, pitch = float(obs[0]), float(obs[1])
        servo = np.asarray(info["servo_angles"], dtype=np.float64)
        target = np.asarray(info["target"], dtype=np.float64)
        foot_err = float(info["foot_height_error"])
        act = np.asarray(action, dtype=np.float64)

        # smoothstep 门：0.75 → 1.0 之间平滑打开
        raw = (up - self.gate_start) / max(1.0 - self.gate_start, 1e-6)
        raw = float(np.clip(raw, 0.0, 1.0))
        gate = raw * raw * (3.0 - 2.0 * raw)

        act_sq = float(np.sum(act * act))
        d_target_sq = float(np.sum((target - prev_target) ** 2))

        reward = 2.0 * up + 8.0 * d_up + self.up_bonus * max(up, 0.0) ** 4
        reward -= 0.2 * (roll * roll + pitch * pitch)
        reward -= self.gate_scale * gate * (
            0.5 * float(np.sum(servo * servo))
            + 0.2 * float(np.sum(target * target))
            + 5.0 * foot_err * foot_err
        )
        reward -= 0.01 * float(np.sum(self.data.qvel[0:6] ** 2))
        reward -= 0.005 * float(np.sum(self._servo_velocities() ** 2))
        reward -= 0.005 * act_sq
        reward -= 0.02 * d_target_sq

        # 接近直立时给一个连续的"别抖"梯度，避免只有达标瞬间才有信号
        hi = float(np.clip((up - 0.90) / 0.10, 0.0, 1.0))
        hi = hi * hi * (3.0 - 2.0 * hi)
        reward -= self.still_action_penalty * hi * act_sq

        if info["goal_pose"]:
            reward += 5.0
            # ★ 静止奖励 + 抖振惩罚：达标后动作越接近 0 越好
            reward += self.hold_bonus * math.exp(-self.hold_sharp * act_sq)
            reward -= self.hold_action_penalty * act_sq

        return obs, float(reward), terminated, truncated, info


def make_env(rank: int, base_seed: int, jitter_deg: float, reward: str,
             reward_params: dict | None = None):
    env_cls = ShapedEnv if reward == "shaped" else JitteredEnv

    def _init():
        env = env_cls("robo1.xml", fallen_poses=POSES)
        env.jitter_deg = jitter_deg
        for key, value in (reward_params or {}).items():
            setattr(env, key, value)
        # 用 rank 派生种子初始化 np_random，之后的重置会沿用这条随机流
        env.reset(seed=base_seed * 1000 + rank)
        return env

    return _init


class RewardAnnealCallback(BaseCallback):
    """把 ShapedEnv.gate_scale 从 0 线性退火到 1（跨进程生效）。"""

    def __init__(self, start_frac: float = 0.20, end_frac: float = 0.55, verbose: int = 0):
        super().__init__(verbose)
        self.start_frac = start_frac
        self.end_frac = end_frac
        self._ts0: int | None = None
        self._last = None

    def _on_step(self) -> bool:
        if self._ts0 is None:
            self._ts0 = self.num_timesteps
        span = max(self.model._total_timesteps - self._ts0, 1)
        frac = (self.num_timesteps - self._ts0) / span
        if frac <= self.start_frac:
            value = 0.0
        elif frac >= self.end_frac:
            value = 1.0
        else:
            value = (frac - self.start_frac) / (self.end_frac - self.start_frac)
        if value != self._last:
            try:
                self.training_env.set_attr("gate_scale", float(value))
            except Exception as exc:  # 向量环境不支持时不要打断训练
                if self._last is None:
                    print(f"  [警告] gate_scale 无法下发到子进程：{exc}", flush=True)
            self.logger.record("reward/gate_scale", float(value))
            self._last = value
        return True


def evaluate_poses(model) -> tuple[int, dict]:
    """用作者原版 eval_pose 逐姿态评估，返回 (达标数, 明细)。"""
    detail, n_ok = {}, 0
    for pose in POSES:
        best, final = eval_pose(model, pose)
        ok = bool(final.get("goal_pose")) or bool(best.get("goal_pose"))
        detail[pose] = {
            "upright": round(float(best["upright"]), 4),
            "goal": ok,
            "final_upright": round(float(final.get("upright", -1.0)), 4),
            "servo": [round(float(v), 3) for v in np.atleast_1d(final.get("servo_angles", [0, 0]))],
            "steps": int(best.get("step", -1)),
        }
        n_ok += int(ok)
    return n_ok, detail


class PoseEvalCallback(BaseCallback):
    """每 eval_every 步评估四个姿态：达标数 + 达标后的静止程度。

    评估用上游原版判定（`eval_robo1_policy.eval_pose`），额外用
    `eval_metrics.measure_steadiness` 量出「达标后舵机指令的峰峰摆幅」。
    最优权重按 (达标数, −抖动) 的字典序挑选，所以自动存下来的
    `<tag>_best.zip` 就是「四个姿态都达标且不抖舵机」的那一版。
    """

    def __init__(self, eval_every: int, ckpt_dir: Path, tag: str, verbose: int = 0):
        super().__init__(verbose)
        self.eval_every = eval_every
        self.ckpt_dir = ckpt_dir
        self.tag = tag
        self.next_eval = None
        self.best_ok = -1
        self.best_swing = float("inf")
        self._env = Robo1GetupEnv("robo1.xml", fallen_poses=POSES)
        self.history: list[dict] = []

    def _on_step(self) -> bool:
        if self.next_eval is None:
            # 续训时 num_timesteps 已经很大，直接从"下一个整倍数"开始，避免刚续训就狂评估
            self.next_eval = ((self.num_timesteps // self.eval_every) + 1) * self.eval_every
        if self.num_timesteps < self.next_eval:
            return True
        self.next_eval += self.eval_every

        t0 = time.time()
        n_ok, detail = evaluate_poses(self.model)
        swings = {}
        for pose in POSES:
            metrics = measure_steadiness(self._env, self.model, pose)
            swings[pose] = metrics
            self.logger.record(f"eval/cmd_swing_{pose}", metrics["cmd_swing_deg"])
        reached = [m["cmd_swing_deg"] for m in swings.values() if m["goal_steps"] > 0]
        mean_swing = float(np.mean(reached)) if reached else float("inf")
        self.logger.record("eval/poses_ok", n_ok)
        self.logger.record("eval/cmd_swing_mean", 0.0 if mean_swing == float("inf") else mean_swing)
        for pose, d in detail.items():
            self.logger.record(f"eval/upright_{pose}", d["upright"])
        self.logger.record("eval/seconds", time.time() - t0)
        self.logger.dump(self.num_timesteps)

        entry = {"steps": int(self.num_timesteps), "ok": n_ok, "detail": detail,
                 "steadiness": swings,
                 "mean_swing_deg": None if mean_swing == float("inf") else round(mean_swing, 2)}
        self.history.append(entry)
        (self.ckpt_dir / f"{self.tag}_eval_history.json").write_text(
            json.dumps(self.history, indent=2), encoding="utf-8"
        )

        marks = " ".join(f"{p.split('_')[0][:5]}{'✅' if d['goal'] else '❌'}" for p, d in detail.items())
        swing_txt = "—" if mean_swing == float("inf") else f"{mean_swing:.2f}°"
        print(f"[{time.strftime('%H:%M:%S')}] {self.num_timesteps:>9} 步  "
              f"{n_ok}/4  {marks}  达标摆幅 {swing_txt}  ({time.time()-t0:.1f}s)", flush=True)

        if (n_ok, -mean_swing) > (self.best_ok, -self.best_swing):
            self.best_ok = n_ok
            self.best_swing = mean_swing
            self.model.save(str(self.ckpt_dir / f"{self.tag}_best"))
            np.save(str(self.ckpt_dir / f"{self.tag}_best_ok.npy"), np.array([n_ok]))
            print(f"          ↳ 新的最佳 {n_ok}/4（摆幅 {swing_txt}），已存 {self.tag}_best.zip",
                  flush=True)
        return True


def build_model(args, train_env):
    net_arch = [int(v) for v in args.net_arch.split(",") if v.strip()]
    return PPO(
        "MlpPolicy",
        train_env,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        # 注意：SB3 传给 schedule 的是 progress_remaining（1→0），所以乘它才是线性衰减。
        learning_rate=(lambda progress_remaining: args.lr * max(progress_remaining, 0.0)),
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=0.2,
        ent_coef=args.ent_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=net_arch, log_std_init=args.log_std_init),
        seed=args.seed,
        device="cpu",
        tensorboard_log=str(args.tb_dir),
        verbose=1,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="从零开始的纯 RL 起立训练")
    parser.add_argument("--steps", type=int, default=3_000_000)
    parser.add_argument("--n-envs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", default="pure_rl")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--log-std-init", type=float, default=-0.5)
    parser.add_argument("--net-arch", default="128,128")
    parser.add_argument("--pose-jitter-deg", type=float, default=0.0)
    parser.add_argument("--reward", choices=("shaped", "author"), default="shaped",
                        help="shaped=为从零学习重写的奖励（默认）；author=完全沿用作者原奖励")
    parser.add_argument("--anneal-start", type=float, default=0.0,
                        help="shaped 奖励：从总进度的这个比例开始退火 gate_scale")
    parser.add_argument("--anneal-end", type=float, default=0.40,
                        help="shaped 奖励：到这个比例时 gate_scale 达到 1")
    parser.add_argument("--hold-bonus", type=float, default=3.0,
                        help="达标后完全静止可拿到的额外奖励（0 = 关闭静止奖励）")
    parser.add_argument("--hold-sharp", type=float, default=10.0,
                        help="静止奖励对动作幅度的敏感度 exp(-sharp·Σa²)")
    parser.add_argument("--hold-action-penalty", type=float, default=1.5,
                        help="达标后对 Σa² 的惩罚（抑制舵机抖振的主力，0 = 关闭）。"
                             "注意别超过达标奖励 5.0，否则策略会宁愿不达标")
    parser.add_argument("--still-action-penalty", type=float, default=0.0,
                        help="接近直立（upright>0.9）时对动作幅度的惩罚；默认 0，"
                             "调大会拖慢收手臂过程")
    parser.add_argument("--norm-reward", action="store_true", default=True)
    parser.add_argument("--no-norm-reward", dest="norm_reward", action="store_false")
    parser.add_argument("--eval-every", type=int, default=50_000)
    parser.add_argument("--ckpt-every", type=int, default=100_000)
    parser.add_argument("--init", default=None, help="从本脚本自己的 checkpoint 继续（不是示教权重）")
    parser.add_argument("--eval-only", default=None, help="只评估一个权重，不训练")
    args = parser.parse_args()

    args.ckpt_dir = (HERE / "ckpt").resolve()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    args.tb_dir = (HERE / "tb").resolve()
    args.tb_dir.mkdir(parents=True, exist_ok=True)

    # 脚本会把 cwd 切到 env/，所以 --init / --eval-only 的相对路径在这里自行解析：
    # 依次尝试 「rl/ 下」「项目根目录下」「补 .zip」
    for attr in ("init", "eval_only"):
        value = getattr(args, attr)
        if value:
            candidate = Path(value)
            if not candidate.is_absolute():
                for base in (HERE, HERE.parent):
                    probe = base / candidate
                    if probe.exists():
                        candidate = probe
                        break
                    if probe.suffix != ".zip" and probe.with_suffix(".zip").exists():
                        candidate = probe.with_suffix(".zip")
                        break
                else:
                    candidate = HERE / candidate
            setattr(args, attr, str(candidate))

    if args.eval_only:
        model = PPO.load(args.eval_only, device="cpu")
        n_ok, detail = evaluate_poses(model)
        print(json.dumps({"weight": args.eval_only, "ok": n_ok, "detail": detail},
                         indent=2, ensure_ascii=False), flush=True)
        return

    print(f"纯 RL 从零训练：{args.steps} 步，{args.n_envs} 个并行环境，seed={args.seed}", flush=True)
    print(f"  网络 {args.net_arch} | lr {args.lr} | log_std {args.log_std_init} | "
          f"n_steps {args.n_steps}×{args.n_envs} | jitter ±{args.pose_jitter_deg}°", flush=True)
    print(f"  奖励模式 {args.reward}"
          + (f"（gate_scale 在进度 {args.anneal_start:.0%}→{args.anneal_end:.0%} 之间 0→1）"
             if args.reward == "shaped" else "（作者原版）"), flush=True)
    if args.reward == "shaped":
        print(f"  静止奖励：hold_bonus={args.hold_bonus} hold_sharp={args.hold_sharp} "
              f"hold_action_penalty={args.hold_action_penalty} "
              f"still_penalty={args.still_action_penalty}",
              flush=True)

    reward_params = {
        "hold_bonus": args.hold_bonus,
        "hold_sharp": args.hold_sharp,
        "hold_action_penalty": args.hold_action_penalty,
        "still_action_penalty": args.still_action_penalty,
    }

    train_env = SubprocVecEnv(
        [make_env(i, args.seed, args.pose_jitter_deg, args.reward, reward_params)
         for i in range(args.n_envs)],
        start_method="spawn",
    )
    train_env = VecMonitor(train_env)
    if args.norm_reward:
        train_env = VecNormalize(train_env, norm_obs=False, norm_reward=True, clip_reward=10.0)

    if args.init:
        model = PPO.load(args.init, env=train_env, device="cpu")
        model.tensorboard_log = str(args.tb_dir)
        model.learning_rate = args.lr
        model._setup_lr_schedule()
        print(f"从 {args.init} 继续训练（仍是本脚本自己的 RL 权重）", flush=True)
    else:
        model = build_model(args, train_env)

    callbacks = [
        PoseEvalCallback(args.eval_every, args.ckpt_dir, args.tag),
        CheckpointCallback(save_freq=max(args.ckpt_every // args.n_envs, 1),
                           save_path=str(args.ckpt_dir), name_prefix=args.tag),
    ]
    if args.reward == "shaped":
        callbacks.append(RewardAnnealCallback(args.anneal_start, args.anneal_end))

    t0 = time.time()
    model.learn(total_timesteps=args.steps, callback=callbacks,
                tb_log_name=args.tag, reset_num_timesteps=not args.init)
    model.save(str(args.ckpt_dir / f"{args.tag}_last"))
    print(f"\n训练结束（{(time.time()-t0)/60:.1f} 分钟）", flush=True)

    n_ok, detail = evaluate_poses(model)
    print(json.dumps({"weight": f"{args.tag}_last", "ok": n_ok, "detail": detail},
                     indent=2, ensure_ascii=False), flush=True)
    train_env.close()


if __name__ == "__main__":
    main()
