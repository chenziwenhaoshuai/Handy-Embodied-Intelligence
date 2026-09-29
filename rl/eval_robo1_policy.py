import os
import sys
from pathlib import Path

# 让脚本从任何目录运行都能找到 env/ 里的 MuJoCo 模型与 robo1_env
_ENV_DIR = Path(__file__).resolve().parent / "env"
sys.path.insert(0, str(_ENV_DIR))
os.chdir(_ENV_DIR)

import numpy as np
from stable_baselines3 import PPO

from robo1_env import Robo1GetupEnv


DEFAULT_WEIGHT = Path(__file__).resolve().parent / "weights" / "eei_pure_rl_v1.zip"


def eval_pose(model, pose):
    env = Robo1GetupEnv("robo1.xml", fallen_poses=(pose,))
    obs, _ = env.reset(options={"pose": pose})

    best = {"upright": -1.0}
    final_info = {}
    for step in range(env.max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if info["upright"] > best["upright"]:
            best = {
                "step": step,
                "upright": info["upright"],
                "servo_angles": info["servo_angles"],
                "target": info["target"],
                "goal_pose": info["goal_pose"],
            }
        final_info = info
        if terminated or truncated:
            break

    return best, final_info


def main():
    model_path = DEFAULT_WEIGHT
    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} is missing. Train first or point at another zip.")

    env = Robo1GetupEnv("robo1.xml")
    model = PPO.load(str(model_path), env=env, device="cpu")
    for pose in ("roll_pos", "roll_neg", "pitch_pos", "pitch_neg"):
        best, final_info = eval_pose(model, pose)
        print("pose", pose)
        print("best", best)
        print(
            "final",
            {
                "upright": final_info.get("upright"),
                "servo_angles": np.round(final_info.get("servo_angles"), 5),
                "target": np.round(final_info.get("target"), 5),
                "goal_pose": final_info.get("goal_pose"),
                "success_count": final_info.get("success_count"),
            },
        )


if __name__ == "__main__":
    main()
