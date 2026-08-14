from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco.viewer
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from env import SelfRightingEnv  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Interactive MuJoCo viewer for the self-righting robot.")
    parser.add_argument("--tilt-deg", type=float, default=55.0)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--action-step", type=float, default=0.08)
    parser.add_argument("--auto-reset", action="store_true", help="Reset automatically at Gym episode end.")
    args = parser.parse_args()

    env = SelfRightingEnv(max_tilt_degrees=args.tilt_deg)
    env.reset(seed=args.seed)

    action = np.zeros(2, dtype=np.float32)
    paused = False

    print("Controls: A/D joint_x, W/S joint_y, C center, R reset, Space pause/resume, Esc close.")

    def key_callback(key: int) -> None:
        nonlocal action, paused
        if key == ord("A"):
            action[0] = np.clip(action[0] - args.action_step, -1.0, 1.0)
        elif key == ord("D"):
            action[0] = np.clip(action[0] + args.action_step, -1.0, 1.0)
        elif key == ord("S"):
            action[1] = np.clip(action[1] - args.action_step, -1.0, 1.0)
        elif key == ord("W"):
            action[1] = np.clip(action[1] + args.action_step, -1.0, 1.0)
        elif key == ord("C"):
            action[:] = 0.0
        elif key == ord("R"):
            env.reset()
            action[:] = 0.0
        elif key == 32:
            paused = not paused
        print(f"action={np.round(action, 3)}, paused={paused}")

    with mujoco.viewer.launch_passive(env.model, env.data, key_callback=key_callback) as viewer:
        while viewer.is_running():
            loop_start = time.time()
            if not paused:
                _, _, terminated, truncated, _ = env.step(action)
                if args.auto_reset and (terminated or truncated):
                    env.reset()
                    action[:] = 0.0
            viewer.sync()
            elapsed = time.time() - loop_start
            time.sleep(max(0.0, env.config.control.actual_policy_dt - elapsed))

    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
