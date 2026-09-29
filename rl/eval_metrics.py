"""评估指标：达标数之外的"静止程度"量化。

为什么需要它：策略可以既达标（连续 50 步满足 goal_pose）又在不停抖舵机。
第一版跑通后上机就出现了这个问题——达标状态下动作在 ±0.7 之间跳，
固件下发的舵机角度来回摆 9°–13°，肉眼就是"站直之后还在抽"。

`cmd_swing_deg` 是达标期间舵机指令的峰峰值（度），直接对应肉眼看到的抖动幅度。
训练脚本用它来挑 checkpoint，验证脚本用它来出报告。
"""

from __future__ import annotations

import numpy as np


def measure_steadiness(env, model, pose: str) -> dict:
    """跑一个回合，统计达标之后舵机指令的抖动程度。

    * `mean_d_target`：达标状态下每步 target 的平均变化量（越小越稳）
    * `mean_action`：达标状态下策略输出的平均幅度
    * `reversal_ratio`：target 变化方向翻转的比例（逐维度统计）
    * `cmd_swing_deg`：达标期间舵机指令的峰峰值（度）—— 最直观的抖动指标
    """
    obs, _ = env.reset(seed=0, options={"pose": pose})
    prev_target = None
    prev_delta = None
    goal_steps = 0
    d_sum = 0.0
    a_sum = 0.0
    reversals = 0
    first_goal = -1
    t_min = np.full(2, np.inf)
    t_max = np.full(2, -np.inf)

    for step in range(env.max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        target = np.asarray(info["target"], dtype=np.float64)
        if info["goal_pose"]:
            if first_goal < 0:
                first_goal = step
            goal_steps += 1
            t_min = np.minimum(t_min, target)
            t_max = np.maximum(t_max, target)
            a_sum += float(np.abs(np.asarray(action, dtype=np.float64)).sum())
            if prev_target is not None:
                delta = target - prev_target
                d_sum += float(np.abs(delta).sum())
                if prev_delta is not None:
                    reversals += int(np.sum(np.sign(delta) * np.sign(prev_delta) < 0))
                prev_delta = delta
            prev_target = target
        else:
            prev_target = None
            prev_delta = None
        if terminated or truncated:
            break

    n_pairs = max(goal_steps - 1, 1)
    swing = float(np.degrees(np.max(t_max - t_min))) if goal_steps else 0.0
    return {
        "goal_steps": goal_steps,
        "first_goal_step": first_goal,
        "mean_d_target": round(d_sum / n_pairs, 6),
        "mean_action": round(a_sum / max(goal_steps, 1), 4),
        "reversal_ratio": round(reversals / (2 * n_pairs), 4),
        "cmd_swing_deg": round(swing, 2),
    }
