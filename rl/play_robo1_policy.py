import argparse
import os
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
from stable_baselines3 import PPO

# 让脚本从任何目录运行都能找到 env/ 里的 MuJoCo 模型与 robo1_env
_ENV_DIR = Path(__file__).resolve().parent / "env"
sys.path.insert(0, str(_ENV_DIR))
os.chdir(_ENV_DIR)

from robo1_env import DEFAULT_FALLEN_POSES, Robo1GetupEnv


DEFAULT_WEIGHT = Path(__file__).resolve().parent / "weights" / "eei_pure_rl_v1.zip"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose", choices=DEFAULT_FALLEN_POSES, default=None)
    parser.add_argument("--weight", default=str(DEFAULT_WEIGHT),
                        help="要回放的权重（默认 rl/weights/eei_pure_rl_v1.zip）")
    args = parser.parse_args()

    fallen_poses = (args.pose,) if args.pose is not None else DEFAULT_FALLEN_POSES
    env = Robo1GetupEnv("robo1.xml", fallen_poses=fallen_poses)
    model = PPO.load(args.weight, device="cpu")
    reset_options = {"pose": args.pose} if args.pose is not None else None
    obs, info = env.reset(options=reset_options)
    print("initial pose:", info["pose"])

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            viewer.sync()
            time.sleep(env.model.opt.timestep * env.frame_skip)
            if terminated or truncated:
                obs, info = env.reset(options=reset_options)
                print("initial pose:", info["pose"])


if __name__ == "__main__":
    main()
