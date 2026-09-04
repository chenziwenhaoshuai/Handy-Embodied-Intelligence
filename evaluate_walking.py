from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

import mujoco.viewer
import numpy as np

from config import CONFIG
from env import WalkingEnv


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a walking policy with keyboard velocity commands.")
    parser.add_argument("--model-path", type=Path, default=Path("runs/ppo_walking.zip"))
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--command-speed", type=float, default=0.08)
    parser.add_argument("--episode-seconds", type=float, default=8.0)
    parser.add_argument("--debug-every", type=int, default=0)
    args = parser.parse_args()

    from stable_baselines3 import PPO

    config = replace(CONFIG, episode_seconds=args.episode_seconds)
    env = WalkingEnv(
        config=config,
        command_speed=args.command_speed,
        fixed_command=(0.0, 0.0),
    )
    model = PPO.load(args.model_path)

    command = np.zeros(2, dtype=np.float64)
    paused = False
    reset_requested = False

    print("Controls: W/S forward/back, A/D left/right, C stop, R reset, Space pause/resume, Esc close.")

    def apply_command() -> None:
        env.set_command(float(command[0]), float(command[1]))
        print(f"command_v=[{command[0]:.3f}, {command[1]:.3f}] m/s")

    def key_callback(key: int) -> None:
        nonlocal paused, reset_requested
        if key == ord("W"):
            command[:] = [args.command_speed, 0.0]
            apply_command()
        elif key == ord("S"):
            command[:] = [-args.command_speed, 0.0]
            apply_command()
        elif key == ord("A"):
            command[:] = [0.0, args.command_speed]
            apply_command()
        elif key == ord("D"):
            command[:] = [0.0, -args.command_speed]
            apply_command()
        elif key == ord("C"):
            command[:] = [0.0, 0.0]
            apply_command()
        elif key == ord("R"):
            reset_requested = True
        elif key == 32:
            paused = not paused
            print(f"paused={paused}")

    obs, _ = env.reset(seed=args.seed)
    env.set_command(0.0, 0.0)
    total_reward = 0.0
    step_count = 0
    episode = 0

    with mujoco.viewer.launch_passive(env.model, env.data, key_callback=key_callback) as viewer:
        while viewer.is_running() and episode < args.episodes:
            loop_start = time.time()
            if reset_requested:
                obs, _ = env.reset(seed=args.seed + episode + 1)
                env.set_command(float(command[0]), float(command[1]))
                total_reward = 0.0
                step_count = 0
                reset_requested = False

            if not paused:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                if args.debug_every > 0 and step_count % args.debug_every == 0:
                    print(
                        f"step={step_count:04d} cmd={info['command_velocity'].round(3)} "
                        f"vel={info['xy_velocity'].round(3)} action={action.round(3)} reward={reward:.2f}"
                    )
                step_count += 1
                if terminated or truncated:
                    print(f"episode={episode} total_reward={total_reward:.2f}")
                    episode += 1
                    obs, _ = env.reset(seed=args.seed + episode)
                    env.set_command(float(command[0]), float(command[1]))
                    total_reward = 0.0
                    step_count = 0

            viewer.sync()
            elapsed = time.time() - loop_start
            time.sleep(max(0.0, env.config.control.actual_policy_dt - elapsed))

    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
