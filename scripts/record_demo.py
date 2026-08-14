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
from config import CONFIG, EnvConfig  # noqa: E402


def _load_existing_dataset(
    path: Path,
    overwrite: bool,
    observation_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    if overwrite or not path.exists():
        return (
            np.empty((0, observation_dim), dtype=np.float32),
            np.empty((0, 2), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
            [],
        )

    data = np.load(path)
    observations = data["observations"].astype(np.float32)
    if observations.ndim != 2 or observations.shape[1] < observation_dim:
        raise ValueError(f"expected observations shape (N, >= {observation_dim}), got {observations.shape}")
    if observations.shape[1] != observation_dim:
        observations = observations[:, :observation_dim]
        print(f"trimmed existing demos to first {observation_dim} deployable features")
    return observations, data["actions"].astype(np.float32), data["episode_ids"].astype(
        np.int32
    ), data["episode_lengths"].astype(np.int32).tolist()


def _save_dataset(
    path: Path,
    observations: np.ndarray,
    actions: np.ndarray,
    episode_ids: np.ndarray,
    episode_lengths: list[int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        observations=observations.astype(np.float32),
        actions=actions.astype(np.float32),
        episode_ids=episode_ids.astype(np.int32),
        episode_lengths=np.asarray(episode_lengths, dtype=np.int32),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Record keyboard demonstrations for behavior cloning.")
    parser.add_argument("--output", type=Path, default=Path("data/demos/manual_demo.npz"))
    parser.add_argument("--tilt-deg", type=float, default=85.0)
    parser.add_argument("--min-tilt-deg", type=float, default=55.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--action-step", type=float, default=0.08)
    parser.add_argument("--episode-seconds", type=float, default=3600.0)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing demo file instead of appending.")
    args = parser.parse_args()

    config = EnvConfig(
        robot=CONFIG.robot,
        control=CONFIG.control,
        reset=CONFIG.reset,
        imu=CONFIG.imu,
        reward=CONFIG.reward,
        episode_seconds=args.episode_seconds,
    )
    env = SelfRightingEnv(config=config, min_tilt_degrees=args.min_tilt_deg, max_tilt_degrees=args.tilt_deg)
    observation_dim = int(env.observation_space.shape[0])
    existing_obs, existing_actions, existing_episode_ids, episode_lengths = _load_existing_dataset(
        args.output,
        args.overwrite,
        observation_dim,
    )
    next_episode_id = int(existing_episode_ids.max() + 1) if existing_episode_ids.size else 0

    kept_obs: list[np.ndarray] = []
    kept_actions: list[np.ndarray] = []
    kept_episode_ids: list[np.ndarray] = []
    current_obs: np.ndarray
    current_episode_obs: list[np.ndarray] = []
    current_episode_actions: list[np.ndarray] = []
    current_reward = 0.0
    saved_count = 0

    current_obs, _ = env.reset(seed=args.seed)

    action = np.zeros(2, dtype=np.float32)
    paused = False
    waiting_for_decision = False

    def reset_episode() -> None:
        nonlocal current_obs, current_reward
        current_episode_obs.clear()
        current_episode_actions.clear()
        current_reward = 0.0
        action[:] = 0.0
        current_obs, _ = env.reset()

    def keep_episode(reason: str) -> None:
        nonlocal next_episode_id, saved_count, waiting_for_decision
        if not current_episode_obs:
            print("No samples in current episode; nothing saved.")
            reset_episode()
            waiting_for_decision = False
            return

        obs_array = np.asarray(current_episode_obs, dtype=np.float32)
        action_array = np.asarray(current_episode_actions, dtype=np.float32)
        kept_obs.append(obs_array)
        kept_actions.append(action_array)
        kept_episode_ids.append(np.full((len(obs_array),), next_episode_id, dtype=np.int32))
        episode_lengths.append(len(obs_array))
        saved_count += 1
        all_obs = np.concatenate([existing_obs, *kept_obs], axis=0)
        all_actions = np.concatenate([existing_actions, *kept_actions], axis=0)
        all_episode_ids = np.concatenate([existing_episode_ids, *kept_episode_ids], axis=0)
        _save_dataset(args.output, all_obs, all_actions, all_episode_ids, episode_lengths)
        print(
            f"Kept episode {next_episode_id} ({reason}): "
            f"samples={len(obs_array)}, reward={current_reward:.1f}, saved={args.output}"
        )
        next_episode_id += 1
        reset_episode()
        waiting_for_decision = False

    def discard_episode(reason: str) -> None:
        nonlocal waiting_for_decision
        print(
            f"Discarded current episode ({reason}): "
            f"samples={len(current_episode_obs)}, reward={current_reward:.1f}"
        )
        reset_episode()
        waiting_for_decision = False

    print(
        "Controls: A/D joint_x, W/S joint_y, C center, "
        "K keep+reset, X or R discard+reset, Space pause/resume, Esc close."
    )
    print(f"Writing demonstrations to {args.output}")

    def key_callback(key: int) -> None:
        nonlocal paused, waiting_for_decision
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
        elif key == ord("K"):
            keep_episode("manual")
        elif key in (ord("X"), ord("R")):
            discard_episode("manual")
        elif key == 32:
            paused = not paused
            if paused:
                waiting_for_decision = True
        print(f"action={np.round(action, 3)}, paused={paused}")

    with mujoco.viewer.launch_passive(env.model, env.data, key_callback=key_callback) as viewer:
        while viewer.is_running():
            loop_start = time.time()
            if not paused and not waiting_for_decision:
                obs_before = current_obs.copy()
                next_obs, reward, terminated, truncated, info = env.step(action)
                current_episode_obs.append(obs_before)
                current_episode_actions.append(action.copy())
                current_reward += float(reward)
                current_obs = next_obs

                if terminated:
                    waiting_for_decision = True
                    print("Success detected. Press K to keep this episode or X/R to discard.")
                elif truncated:
                    waiting_for_decision = True
                    print("Episode reached the recording time limit. Press K to keep or X/R to discard.")

            viewer.sync()
            elapsed = time.time() - loop_start
            time.sleep(max(0.0, env.config.control.actual_policy_dt - elapsed))

    env.close()

    if kept_obs:
        new_obs = np.concatenate(kept_obs, axis=0)
        new_actions = np.concatenate(kept_actions, axis=0)
        new_episode_ids = np.concatenate(kept_episode_ids, axis=0)
        observations = np.concatenate([existing_obs, new_obs], axis=0)
        actions = np.concatenate([existing_actions, new_actions], axis=0)
        episode_ids = np.concatenate([existing_episode_ids, new_episode_ids], axis=0)
    else:
        observations = existing_obs
        actions = existing_actions
        episode_ids = existing_episode_ids

    if observations.size:
        _save_dataset(args.output, observations, actions, episode_ids, episode_lengths)
        print(
            f"Saved dataset: path={args.output}, total_samples={len(observations)}, "
            f"total_episodes={len(episode_lengths)}, new_episodes={saved_count}"
        )
    else:
        print("No demonstrations saved.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
