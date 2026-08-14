from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from env import SelfRightingEnv  # noqa: E402


DEFAULT_HIDDEN_SIZES = [512, 512, 512]


def make_policy_kwargs(hidden_sizes: list[int]) -> dict:
    return {"net_arch": {"pi": hidden_sizes, "vf": hidden_sizes}}


def main() -> int:
    parser = argparse.ArgumentParser(description="Behavior-clone a PPO policy from manual demonstrations.")
    parser.add_argument("--demo-path", type=Path, default=Path("data/demos/manual_demo.npz"))
    parser.add_argument("--save-path", type=Path, default=Path("runs/ppo_self_righting_bc_init.zip"))
    parser.add_argument("--base-model-path", type=Path, default=None, help="Optional PPO model to initialize from.")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-tilt-deg", type=float, default=55.0)
    parser.add_argument("--tilt-deg", type=float, default=85.0)
    parser.add_argument(
        "--hidden-sizes",
        type=int,
        nargs="+",
        default=DEFAULT_HIDDEN_SIZES,
        help="MLP hidden sizes for both actor and critic when creating a new PPO model.",
    )
    parser.add_argument(
        "--train-all-policy-params",
        action="store_true",
        help="Also train value/log-std parameters. By default BC only fits the actor mean path.",
    )
    args = parser.parse_args()

    from stable_baselines3 import PPO

    if not args.demo_path.exists():
        raise FileNotFoundError(f"demo file not found: {args.demo_path}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data = np.load(args.demo_path)
    observations = data["observations"].astype(np.float32)
    actions = np.clip(data["actions"].astype(np.float32), -1.0, 1.0)
    env = SelfRightingEnv(min_tilt_degrees=args.min_tilt_deg, max_tilt_degrees=args.tilt_deg)
    expected_obs_dim = int(env.observation_space.shape[0])
    if observations.ndim != 2 or observations.shape[1] < expected_obs_dim:
        raise ValueError(f"expected observations shape (N, >= {expected_obs_dim}), got {observations.shape}")
    if observations.shape[1] != expected_obs_dim:
        observations = observations[:, :expected_obs_dim]
        print(f"trimmed demo observations to first {expected_obs_dim} deployable features")
    if actions.ndim != 2 or actions.shape[1] != 2:
        raise ValueError(f"expected actions shape (N, 2), got {actions.shape}")
    if len(observations) < 1:
        raise ValueError("demo dataset is empty")

    if args.base_model_path is not None:
        model = PPO.load(args.base_model_path, env=env, device=args.device)
    else:
        model = PPO(
            "MlpPolicy",
            env,
            verbose=0,
            seed=args.seed,
            device=args.device,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=512,
            gamma=0.98,
            gae_lambda=0.95,
            ent_coef=0.0,
            policy_kwargs=make_policy_kwargs(args.hidden_sizes),
        )

    device = model.policy.device
    obs_tensor = torch.as_tensor(observations, dtype=torch.float32, device=device)
    action_tensor = torch.as_tensor(actions, dtype=torch.float32, device=device)
    if args.train_all_policy_params:
        trainable_parameters = list(model.policy.parameters())
    else:
        trainable_parameters = list(model.policy.mlp_extractor.policy_net.parameters()) + list(
            model.policy.action_net.parameters()
        )
    optimizer = torch.optim.Adam(trainable_parameters, lr=args.learning_rate)

    sample_count = len(observations)
    print(
        f"Loaded demos: samples={sample_count}, episodes={len(data.get('episode_lengths', []))}, "
        f"device={device}"
    )

    for epoch in range(1, args.epochs + 1):
        indices = np.random.permutation(sample_count)
        total_loss = 0.0
        for start in range(0, sample_count, args.batch_size):
            batch_indices = torch.as_tensor(
                indices[start : start + args.batch_size],
                dtype=torch.long,
                device=device,
            )
            batch_obs = obs_tensor.index_select(0, batch_indices)
            batch_actions = action_tensor.index_select(0, batch_indices)

            distribution = model.policy.get_distribution(batch_obs)
            predicted_actions = distribution.distribution.mean
            loss = F.mse_loss(predicted_actions, batch_actions)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=0.5)
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(batch_indices)

        mean_loss = total_loss / sample_count
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            with torch.no_grad():
                distribution = model.policy.get_distribution(obs_tensor)
                predicted_actions = distribution.distribution.mean
                mean_abs_error = torch.mean(torch.abs(predicted_actions - action_tensor)).item()
            print(f"epoch={epoch:04d} mse={mean_loss:.6f} mean_abs_error={mean_abs_error:.4f}")

    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.save_path)
    env.close()
    print(f"saved BC-initialized PPO model to {args.save_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
