from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from env import SelfRightingEnv  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run random actions in the self-righting MuJoCo env.")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--tilt-deg", type=float, default=60.0)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.0)
    args = parser.parse_args()

    env = SelfRightingEnv(render_mode="human" if args.render else None, max_tilt_degrees=args.tilt_deg)
    obs, info = env.reset(seed=args.seed)
    rng = np.random.default_rng(args.seed)

    contact_count = 0
    max_force = 0.0
    min_tip_z = float("inf")
    for step in range(args.steps):
        action = rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        contact_count += int(info["tip_contact"])
        max_force = max(max_force, float(info["tip_contact_force"]))
        min_tip_z = min(min_tip_z, float(info["tip_position"][2]))
        if step % 50 == 0:
            print(
                f"step={step:04d} reward={reward: .3f} contact={info['tip_contact']} "
                f"force={info['tip_contact_force']:.3f} joints={np.round(info['joint_angles'], 3)} "
                f"tip={np.round(info['tip_position'], 3)}"
            )
        if terminated or truncated:
            obs, info = env.reset(seed=args.seed + step + 1)
        if args.sleep > 0.0:
            time.sleep(args.sleep)

    env.close()
    print(f"random action summary: contacts={contact_count}, max_force={max_force:.3f}, min_tip_z={min_tip_z:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
