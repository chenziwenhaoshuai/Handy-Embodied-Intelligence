from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from env import WalkingEnv  # noqa: E402


def main() -> int:
    env = WalkingEnv(command_speed=0.08)
    obs, info = env.reset(seed=1)
    assert obs.shape == (12,), obs.shape
    assert env.action_space.shape == (2,)
    assert np.isfinite(obs).all()

    total_reward = 0.0
    for _ in range(60):
        obs, reward, terminated, truncated, info = env.step(np.zeros(2, dtype=np.float32))
        assert obs.shape == (12,)
        assert np.isfinite(obs).all()
        assert np.isfinite(reward)
        total_reward += float(reward)
        if terminated or truncated:
            break

    env.close()
    print(
        "PASS: walking env reset/step - "
        f"total_reward={total_reward:.2f}, "
        f"last_cmd={info['command_velocity'].round(3)}, "
        f"last_xy_vel={info['xy_velocity'].round(3)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
