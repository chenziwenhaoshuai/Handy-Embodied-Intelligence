from __future__ import annotations

import argparse
from pathlib import Path

from env import WalkingEnv
from train import DEFAULT_HIDDEN_SIZES, make_policy_kwargs


def main() -> int:
    parser = argparse.ArgumentParser(description="Train PPO on the upright commanded-walking task.")
    parser.add_argument("--total-timesteps", type=int, default=3_000_000)
    parser.add_argument("--save-path", type=Path, default=Path("runs/ppo_walking"))
    parser.add_argument("--log-dir", type=Path, default=Path("runs/logs"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("runs/checkpoints_walking"))
    parser.add_argument("--checkpoint-freq", type=int, default=10_000)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--command-speed", type=float, default=0.08)
    parser.add_argument("--command-change-seconds", type=float, default=2.0)
    parser.add_argument("--episode-seconds", type=float, default=8.0)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load-model-path", type=Path, default=None)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=DEFAULT_HIDDEN_SIZES)
    args = parser.parse_args()

    from dataclasses import replace

    from config import CONFIG
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.monitor import Monitor

    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.save_path.parent.mkdir(parents=True, exist_ok=True)

    config = replace(CONFIG, episode_seconds=args.episode_seconds)
    env = Monitor(
        WalkingEnv(
            config=config,
            command_speed=args.command_speed,
            command_change_seconds=args.command_change_seconds,
        ),
        filename=str(args.log_dir / "walking_monitor.csv"),
    )
    callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=str(args.checkpoint_dir),
        name_prefix="ppo_walking",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    if args.load_model_path is not None:
        model = PPO.load(
            args.load_model_path,
            env=env,
            device=args.device,
            tensorboard_log=str(args.log_dir / "tensorboard"),
        )
        model.verbose = 1
        model.n_steps = args.n_steps
        model.batch_size = args.batch_size
        print(f"loaded model from {args.load_model_path}")
    else:
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            seed=args.seed,
            device=args.device,
            learning_rate=3e-4,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            gamma=0.98,
            gae_lambda=0.95,
            ent_coef=0.005,
            policy_kwargs=make_policy_kwargs(args.hidden_sizes),
            tensorboard_log=str(args.log_dir / "tensorboard"),
        )

    model.learn(total_timesteps=args.total_timesteps, callback=callback)
    model.save(args.save_path)
    env.close()
    print(f"saved model to {args.save_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
