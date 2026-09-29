r"""MuJoCo 实时窗口：自动加载**最新**的训练权重并循环播放四个姿态。

用途：训练在后台跑，这个窗口隔一会儿就换成更新的 checkpoint，肉眼就能看出
「现在学到哪一步了」。窗口左上角会显示当前播放的权重文件名和本轮结果。

用法：
    python rl/watch_latest.py                      # 自动跟踪 rl/ckpt 里最新的权重
    python rl/watch_latest.py --speed 2            # 2 倍速播放
    python rl/watch_latest.py --weight rl/weights/eei_pure_rl_v1.zip

`--rounds 0` 表示一直循环到手动关窗。按 ESC 或直接关窗口退出。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENV_DIR = (HERE / "env").resolve()
sys.path.insert(0, str(ENV_DIR))
os.chdir(ENV_DIR)  # Robo1GetupEnv 用的是相对路径 "robo1.xml"
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402
import numpy as np  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

from robo1_env import Robo1GetupEnv  # noqa: E402

POSES = ("roll_pos", "roll_neg", "pitch_pos", "pitch_neg")
FONT = mujoco.mjtFontScale.mjFONTSCALE_150
GRID = mujoco.mjtGridPos.mjGRID_TOPLEFT


def newest_weight(ckpt_dir: Path, min_age_s: float) -> Path | None:
    """返回最近写入、且写完超过 min_age_s 秒的 zip（避免读到正在写一半的文件）。"""
    now = time.time()
    cands = [p for p in ckpt_dir.glob("*.zip") if now - p.stat().st_mtime >= min_age_s]
    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_mtime)


def main() -> None:
    parser = argparse.ArgumentParser(description="自动播放最新训练权重的 MuJoCo 窗口")
    parser.add_argument("--ckpt-dir", default=str(HERE / "ckpt"), help="扫描最新权重的目录")
    parser.add_argument("--weight", default=None, help="固定播放某个权重（默认自动跟踪最新）")
    parser.add_argument("--speed", type=float, default=1.0, help="播放倍速（1 = 实时 50Hz）")
    parser.add_argument("--rounds", type=int, default=0, help="跑几轮四姿态；0 = 无限循环")
    parser.add_argument("--reload-every", type=int, default=50, help="每多少个控制步检查一次新权重")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    env = Robo1GetupEnv("robo1.xml", fallen_poses=POSES)
    model = None
    current: Path | None = None
    control_dt = env.model.opt.timestep * env.frame_skip  # 0.001 * 20 = 0.02 s

    print(f"可视化窗口：每 {control_dt*1000:.0f} ms 一个控制步，{args.speed:g}× 倍速", flush=True)
    print(f"权重目录：{ckpt_dir}（自动跟踪最新 zip）", flush=True)

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        round_idx = 0
        while viewer.is_running():
            if round_idx and 0 < args.rounds <= round_idx:
                break
            for pose in POSES:
                if not viewer.is_running():
                    break
                if args.weight:
                    target = Path(args.weight)
                else:
                    target = newest_weight(ckpt_dir, min_age_s=1.0) or current
                if target is not None and target != current:
                    try:
                        model = PPO.load(str(target), device="cpu")
                        current = target
                        print(f"[{time.strftime('%H:%M:%S')}] 已加载 {target.name}", flush=True)
                    except Exception as exc:  # 文件可能还在写
                        print(f"  跳过 {target.name}：{exc}", flush=True)

                obs, _ = env.reset(seed=args.seed, options={"pose": pose})
                best_upright = -1.0
                goal = False
                last_target_cmd = ""
                step = 0
                while viewer.is_running() and step < env.max_steps:
                    if model is None:
                        action = np.zeros(2, dtype=np.float32)  # 还没权重就先摆着
                    else:
                        action, _ = model.predict(obs, deterministic=True)
                    obs, _, terminated, truncated, info = env.step(action)
                    step += 1
                    best_upright = max(best_upright, float(info["upright"]))
                    goal = goal or bool(info["goal_pose"])

                    if step % max(args.reload_every, 1) == 0 and not args.weight:
                        nxt = newest_weight(ckpt_dir, min_age_s=1.0)
                        if nxt is not None and nxt != current:
                            try:
                                model = PPO.load(str(nxt), device="cpu")
                                current = nxt
                                print(f"[{time.strftime('%H:%M:%S')}] 切换到 {nxt.name}", flush=True)
                            except Exception:
                                pass

                    t1 = f"{current.name if current else '未加载权重'}"
                    t2 = (f"{pose}  step {step:>3}/{env.max_steps}  "
                          f"直立度 {info['upright']:+.2f}  最佳 {best_upright:+.2f}  "
                          f"{'已达标' if goal else ''}")
                    if t1 + t2 != last_target_cmd:
                        viewer.set_texts([(FONT, GRID, t1, t2)])
                        last_target_cmd = t1 + t2

                    viewer.sync()
                    nxt_deadline = time.perf_counter() + control_dt / max(args.speed, 1e-6)
                    while time.perf_counter() < nxt_deadline:
                        time.sleep(0.001)
                    if terminated or truncated:
                        break

                print(f"   {pose:10s} 用了 {step:>3} 步，最佳直立度 {best_upright:+.3f}"
                      f"{'，达标 ✅' if goal else ''}", flush=True)
            round_idx += 1

    print("窗口已关闭。", flush=True)


if __name__ == "__main__":
    main()
