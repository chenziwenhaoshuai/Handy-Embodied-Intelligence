from __future__ import annotations

import argparse
from pathlib import Path


def _step_number(path: Path) -> int:
    stem = path.stem
    marker = "_steps"
    if marker not in stem:
        return -1
    prefix = stem[: stem.rfind(marker)]
    token = prefix.split("_")[-1]
    try:
        return int(token)
    except ValueError:
        return -1


def main() -> int:
    parser = argparse.ArgumentParser(description="Print the newest Stable-Baselines3 checkpoint in a directory.")
    parser.add_argument("checkpoint_dir", type=Path)
    parser.add_argument("--top", type=int, default=1)
    args = parser.parse_args()

    if not args.checkpoint_dir.exists():
        raise FileNotFoundError(f"checkpoint dir not found: {args.checkpoint_dir}")

    checkpoints = sorted(
        args.checkpoint_dir.glob("*.zip"),
        key=lambda path: (_step_number(path), path.stat().st_mtime),
        reverse=True,
    )
    if not checkpoints:
        raise FileNotFoundError(f"no .zip checkpoint found in {args.checkpoint_dir}")

    for path in checkpoints[: args.top]:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
