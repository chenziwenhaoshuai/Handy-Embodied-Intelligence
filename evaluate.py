from __future__ import annotations

import argparse
import time
from pathlib import Path

from env import SelfRightingEnv


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a PPO policy in the MuJoCo viewer.")
    parser.add_argument("--model-path", type=Path, default=Path("runs/ppo_self_righting.zip"))
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--tilt-deg", type=float, default=60.0)
    parser.add_argument("--debug-every", type=int, default=0, help="Print action/servo diagnostics every N steps.")
    args = parser.parse_args()

    from stable_baselines3 import PPO

    env = SelfRightingEnv(render_mode="human", max_tilt_degrees=args.tilt_deg)
    model = PPO.load(args.model_path)
    for episode in range(args.episodes):
        obs, info = env.reset(seed=args.seed + episode)
        done = False
        total_reward = 0.0
        step_count = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            if args.debug_every > 0 and step_count % args.debug_every == 0:
                print(
                    f"step={step_count:04d} action={action.round(3)} "
                    f"cmd={info['target_angles'].round(3)} "
                    f"servo={info['servo_target_angles'].round(3)} "
                    f"joint={info['joint_angles'].round(3)} "
                    f"torque={info['joint_torques'].round(3)} "
                    f"reward={reward:.2f}"
                )
            step_count += 1
            time.sleep(env.config.control.actual_policy_dt)
        print(f"episode={episode} total_reward={total_reward:.2f} success_count={info['success_count']}")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
