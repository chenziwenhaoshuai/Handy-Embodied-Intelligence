r"""终验：用作者原版判定标准评估一个权重，并加测「摆放偏差」鲁棒性。

两层验证：
  1. 四个标准倒伏姿态（作者原版 `eval_robo1_policy.eval_pose` 的逻辑，判定条件一字未改）
  2. 每个姿态叠加 roll/pitch 摆放偏差网格（默认 ±10°、±5°、0），模拟实物摆放不精确

用法：
    python rl/verify_pure_rl.py                                   # 默认验证 rl/weights/eei_pure_rl_v1.zip
    python rl/verify_pure_rl.py --weight rl/ckpt/xxx_steps.zip
    python rl/verify_pure_rl.py --offsets -10,0,10                # 只跑 9 组摆放偏差
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENV_DIR = (HERE / "env").resolve()
sys.path.insert(0, str(ENV_DIR))
os.chdir(ENV_DIR)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

from eval_robo1_policy import eval_pose  # noqa: E402
from robo1_env import FALLEN_POSES, Robo1GetupEnv, roll_pitch_to_quat  # noqa: E402

sys.path.insert(0, str(HERE))
from eval_metrics import measure_steadiness  # noqa: E402

POSES = ("roll_pos", "roll_neg", "pitch_pos", "pitch_neg")


def play_from_offset(env, model, pose: str, roll_off_deg: float, pitch_off_deg: float):
    """把机器人按 (姿态 + 偏差) 摆好，跑一整个回合，返回是否达标。"""
    roll, pitch = FALLEN_POSES[pose]
    roll += np.deg2rad(roll_off_deg)
    pitch += np.deg2rad(pitch_off_deg)

    env.data.qpos[:] = 0.0
    env.data.qvel[:] = 0.0
    env.data.qpos[0:3] = [0.0, 0.0, 0.08]
    env.data.qpos[3:7] = roll_pitch_to_quat(roll, pitch)
    env.target[:] = 0.0
    env.data.ctrl[:] = 0.0
    mujoco.mj_forward(env.model, env.data)
    for _ in range(env.settle_steps):
        env.data.ctrl[:] = 0.0
        mujoco.mj_step(env.model, env.data)
    env.target[:] = 0.0
    env.data.ctrl[:] = 0.0
    env.step_count = 0
    env.success_count = 0
    env.prev_upright = env._upright()
    env.current_pose_name = pose

    obs = env._get_obs()
    best = {"upright": -1.0}
    final = {}
    for _ in range(env.max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        if info["upright"] > best["upright"]:
            best = {"upright": float(info["upright"]), "goal_pose": bool(info["goal_pose"])}
        final = info
        if terminated or truncated:
            break
    return bool(final.get("goal_pose")) or bool(best.get("goal_pose")), float(best["upright"])


def main() -> None:
    parser = argparse.ArgumentParser(description="纯 RL 权重终验")
    parser.add_argument("--weight", default=str(HERE / "weights" / "eei_pure_rl_v1.zip"))
    parser.add_argument("--offsets", default="-10,-5,0,5,10", help="摆放偏差网格（度）")
    parser.add_argument("--out", default=str(HERE / "verify_report.json"))
    args = parser.parse_args()

    weight = Path(args.weight)
    if not weight.is_absolute():
        for base in (HERE, HERE.parent):
            probe = base / weight
            if probe.exists():
                weight = probe
                break
            if probe.suffix != ".zip" and probe.with_suffix(".zip").exists():
                weight = probe.with_suffix(".zip")
                break
        else:
            weight = HERE / weight
    model = PPO.load(str(weight), device="cpu")
    offsets = [float(v) for v in args.offsets.split(",") if v.strip()]
    report = {"weight": str(weight), "nominal": {}, "robust": {}}

    print(f"权重：{weight}", flush=True)
    print("\n=== 1. 四个标准姿态（作者原版判定） ===", flush=True)
    env = Robo1GetupEnv("robo1.xml", fallen_poses=POSES)
    n_ok = 0
    for pose in POSES:
        best, final = eval_pose(model, pose)
        ok = bool(final.get("goal_pose")) or bool(best.get("goal_pose"))
        n_ok += int(ok)
        report["nominal"][pose] = {"goal": ok, "upright": round(float(best["upright"]), 4)}
        print(f"  {pose:10s} upright={best['upright']:.4f}  goal_pose={ok}"
              f"  {'✅' if ok else '❌'}", flush=True)
    print(f"  → {n_ok}/4", flush=True)

    print(f"\n=== 2. 摆放偏差鲁棒性（每姿态 {len(offsets)}×{len(offsets)} 组） ===", flush=True)
    total_ok = total = 0
    for pose in POSES:
        pose_ok = 0
        for r_off in offsets:
            for p_off in offsets:
                ok, _up = play_from_offset(env, model, pose, r_off, p_off)
                pose_ok += int(ok)
                total_ok += int(ok)
                total += 1
        report["robust"][pose] = {"ok": pose_ok, "n": len(offsets) ** 2}
        print(f"  {pose:10s} {pose_ok}/{len(offsets)**2}", flush=True)
    print(f"  → 合计 {total_ok}/{total}", flush=True)

    report["summary"] = {"nominal_ok": n_ok, "robust_ok": total_ok, "robust_total": total}
    print("\n=== 3. 达标后的静止程度（越小越好） ===", flush=True)
    steady = {}
    for pose in POSES:
        metrics = measure_steadiness(env, model, pose)
        steady[pose] = metrics
        print(f"  {pose:10s} 达标步数={metrics['goal_steps']:>3}  "
              f"每步Δtarget={metrics['mean_d_target']:.5f}  "
              f"动作均值={metrics['mean_action']:.3f}  "
              f"变向比例={metrics['reversal_ratio']:.3f}  "
              f"指令摆幅={metrics['cmd_swing_deg']:.2f}°", flush=True)
    report["steadiness"] = steady

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = HERE / out_path  # 脚本会切到 env/，相对路径按 rl/ 解析
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n报告：{out_path}", flush=True)


if __name__ == "__main__":
    main()
