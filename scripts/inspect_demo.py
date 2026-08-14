from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect a saved demonstration dataset.")
    parser.add_argument("--demo-path", type=Path, default=Path("data/demos/manual_demo.npz"))
    args = parser.parse_args()

    if not args.demo_path.exists():
        raise FileNotFoundError(f"demo file not found: {args.demo_path}")

    data = np.load(args.demo_path)
    observations = data["observations"]
    actions = data["actions"]
    episode_ids = data["episode_ids"]
    episode_lengths = data["episode_lengths"]

    print(f"path: {args.demo_path}")
    print(f"observations: {observations.shape}")
    print(f"actions: {actions.shape}")
    print(f"episode_ids: {episode_ids.shape}")
    print(f"episodes: {len(episode_lengths)}")
    if len(episode_lengths):
        print(
            "episode_length: "
            f"min={int(np.min(episode_lengths))}, "
            f"mean={float(np.mean(episode_lengths)):.1f}, "
            f"max={int(np.max(episode_lengths))}"
        )
    if len(actions):
        print(f"action_min: {np.min(actions, axis=0).round(3)}")
        print(f"action_mean: {np.mean(actions, axis=0).round(3)}")
        print(f"action_max: {np.max(actions, axis=0).round(3)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
