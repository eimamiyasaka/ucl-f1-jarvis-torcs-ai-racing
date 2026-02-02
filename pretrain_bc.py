"""
Behavioral Cloning: Pretrain policy from demonstrations.

This script trains a neural network to imitate the demonstrated behavior,
then saves it in a format compatible with stable-baselines3 PPO.

Usage:
    python pretrain_bc.py --demos demonstrations.npz --output models/pretrained
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from torcs_rl_env import TorcsRLEnv


class DemonstrationDataset:
    """Load and preprocess demonstration data."""

    def __init__(self, demo_path):
        data = np.load(demo_path)
        self.observations = data['observations']
        self.actions = data['actions']

        print(f"Loaded {len(self.observations)} demonstration steps")
        print(f"  Observation shape: {self.observations.shape}")
        print(f"  Action shape: {self.actions.shape}")

        # Basic stats
        print(f"\nAction statistics:")
        print(f"  Steering: mean={self.actions[:, 0].mean():.3f}, "
              f"std={self.actions[:, 0].std():.3f}")
        print(f"  Accel:    mean={self.actions[:, 1].mean():.3f}, "
              f"std={self.actions[:, 1].std():.3f}")
        print(f"  Brake:    mean={self.actions[:, 2].mean():.3f}, "
              f"std={self.actions[:, 2].std():.3f}")

    def get_tensors(self):
        """Return PyTorch tensors."""
        obs_tensor = torch.FloatTensor(self.observations)
        act_tensor = torch.FloatTensor(self.actions)
        return obs_tensor, act_tensor


def pretrain_ppo_policy(demo_path, output_dir, epochs=100, batch_size=64,
                        learning_rate=1e-3, validation_split=0.1):
    """
    Pretrain a PPO policy using behavioral cloning.

    Args:
        demo_path: Path to demonstration .npz file
        output_dir: Directory to save pretrained model
        epochs: Training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        validation_split: Fraction of data for validation

    Returns:
        Pretrained PPO model
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load demonstrations
    print("=" * 60)
    print("BEHAVIORAL CLONING PRETRAINING")
    print("=" * 60)
    dataset = DemonstrationDataset(demo_path)
    obs_tensor, act_tensor = dataset.get_tensors()

    # Split into train/val
    n_samples = len(obs_tensor)
    n_val = int(n_samples * validation_split)
    n_train = n_samples - n_val

    indices = torch.randperm(n_samples)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_obs = obs_tensor[train_indices]
    train_act = act_tensor[train_indices]
    val_obs = obs_tensor[val_indices]
    val_act = act_tensor[val_indices]

    print(f"\nTraining samples: {n_train}")
    print(f"Validation samples: {n_val}")

    # Create data loaders
    train_dataset = TensorDataset(train_obs, train_act)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Create a dummy environment to initialize PPO with correct spaces
    def make_dummy_env():
        return TorcsRLEnv(port=3001, max_steps=100)

    # We don't actually connect - just need the spaces
    dummy_env = DummyVecEnv([make_dummy_env])

    # Create PPO model with same architecture as training
    print("\nCreating PPO model...")
    model = PPO(
        'MlpPolicy',
        dummy_env,
        learning_rate=3e-4,  # Will be overwritten by BC training
        policy_kwargs=dict(
            net_arch=[dict(pi=[256, 256], vf=[256, 256])]
        ),
        verbose=0
    )

    # Get the policy network
    policy = model.policy

    # Create optimizer for BC training
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

    # Loss function - MSE for continuous actions
    criterion = nn.MSELoss()

    print(f"\nTraining for {epochs} epochs...")
    print("-" * 40)

    best_val_loss = float('inf')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy.to(device)
    val_obs = val_obs.to(device)
    val_act = val_act.to(device)

    for epoch in range(epochs):
        # Training
        policy.train()
        train_losses = []

        for batch_obs, batch_act in train_loader:
            batch_obs = batch_obs.to(device)
            batch_act = batch_act.to(device)

            optimizer.zero_grad()

            # Get action from policy (deterministic)
            # The policy outputs action mean from the actor network
            features = policy.extract_features(batch_obs)
            latent_pi, _ = policy.mlp_extractor(features)
            pred_actions = policy.action_net(latent_pi)

            # Compute loss
            loss = criterion(pred_actions, batch_act)

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)

        # Validation
        policy.eval()
        with torch.no_grad():
            features = policy.extract_features(val_obs)
            latent_pi, _ = policy.mlp_extractor(features)
            pred_actions = policy.action_net(latent_pi)
            val_loss = criterion(pred_actions, val_act).item()

        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1:3d}/{epochs}: "
                  f"train_loss={avg_train_loss:.6f}, val_loss={val_loss:.6f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save(os.path.join(output_dir, 'best_bc_model'))

    # Save final model
    model.save(os.path.join(output_dir, 'final_bc_model'))

    print("-" * 40)
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Models saved to: {output_dir}")
    print("  - best_bc_model.zip (best validation)")
    print("  - final_bc_model.zip (final epoch)")

    dummy_env.close()
    return model


def main():
    parser = argparse.ArgumentParser(description='Pretrain policy with behavioral cloning')
    parser.add_argument('--demos', type=str, required=True,
                        help='Path to demonstrations.npz file')
    parser.add_argument('--output', type=str, default='models/pretrained',
                        help='Output directory for pretrained model')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')

    args = parser.parse_args()

    pretrain_ppo_policy(
        demo_path=args.demos,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )

    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("To continue training with RL from the pretrained model:")
    print(f"  python train_ppo.py --resume {args.output}/best_bc_model")
    print("=" * 60)


if __name__ == "__main__":
    main()
