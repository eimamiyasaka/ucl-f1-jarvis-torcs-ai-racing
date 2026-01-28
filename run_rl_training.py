"""
Quick Start Script for TORCS RL Training

Simple interface to start training, evaluation, or testing.
"""

import os
import sys
import argparse
import subprocess


def check_dependencies():
    """Check if required packages are installed."""
    try:
        import stable_baselines3
        import torch
        import gym
        print("✓ All dependencies installed")
        return True
    except ImportError as e:
        print("✗ Missing dependencies!")
        print(f"  Error: {e}")
        print("\nPlease install requirements:")
        print("  pip install -r requirements_rl.txt")
        return False


def test_environment(port=3001):
    """Test the TORCS environment connection."""
    print("="*60)
    print("Testing TORCS Environment Connection")
    print("="*60)
    print("Make sure TORCS is running with scr_server driver!")
    print(f"Port: {port}")
    print("="*60)

    try:
        from torcs_rl_env import TorcsRLEnv
        env = TorcsRLEnv(port=port, max_steps=500)

        print("\n✓ Environment created successfully")
        print(f"Observation space: {env.observation_space.shape}")
        print(f"Action space: {env.action_space.shape}")

        print("\nTesting reset...")
        obs = env.reset()
        print(f"✓ Reset successful, obs shape: {obs.shape}")

        print("\nTesting 10 random steps...")
        for i in range(10):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            print(f"  Step {i+1}: speed={obs[0]:.1f} km/h, reward={reward:.2f}")
            if done:
                print(f"  Episode ended: {info.get('dnf_reason', 'completed')}")
                break

        env.close()
        print("\n✓ Environment test passed!")
        print("="*60)
        return True

    except Exception as e:
        print(f"\n✗ Environment test failed!")
        print(f"  Error: {e}")
        print("\nTroubleshooting:")
        print("  1. Is TORCS running?")
        print("  2. Is scr_server driver active?")
        print("  3. Is the port correct?")
        print("="*60)
        return False


def quick_train(timesteps=500000, port=3001, target_time=None, no_eval=False):
    """Start a quick training session with default settings."""
    print("="*60)
    print("Starting Quick Training Session")
    print("="*60)
    print(f"Timesteps: {timesteps:,}")
    print(f"Training port: {port}")
    if no_eval:
        print(f"Evaluation: DISABLED (single TORCS instance mode)")
    else:
        print(f"Evaluation port: {port + 1}")
        print(f"  NOTE: Requires second TORCS instance on port {port + 1}")
        print(f"  Use --no-eval if you only have one TORCS instance")
    if target_time:
        print(f"Target lap time: {target_time}s")
    print("="*60)
    print("\nTraining will start in 3 seconds...")
    print("Press Ctrl+C to stop training and save model\n")

    import time
    time.sleep(3)

    # Import and run training
    from train_ppo import train_ppo

    try:
        model, save_dir = train_ppo(
            total_timesteps=timesteps,
            port=port,
            target_lap_time=target_time,
            eval_freq=0 if no_eval else 10000,
            verbose=1
        )
        print(f"\n✓ Training completed!")
        print(f"  Model saved to: {save_dir}")
        return save_dir
    except KeyboardInterrupt:
        print("\n\n✓ Training stopped by user")
        return None


def quick_eval(model_path=None, episodes=10, port=3001):
    """Evaluate a trained model."""
    print("="*60)
    print("Evaluating Trained Model")
    print("="*60)

    # If no model path provided, find the most recent
    if model_path is None:
        models_dir = "models/ppo_torcs"
        if os.path.exists(models_dir):
            # Find most recent run
            runs = [d for d in os.listdir(models_dir) if d.startswith('ppo_')]
            if runs:
                runs.sort(reverse=True)  # Most recent first
                latest_run = runs[0]
                model_path = os.path.join(models_dir, latest_run, 'best_model', 'best_model')
                print(f"Using most recent model: {latest_run}")
            else:
                print("✗ No trained models found!")
                print("  Train a model first with: python run_rl_training.py train")
                return
        else:
            print("✗ Models directory not found!")
            return

    print(f"Model: {model_path}")
    print(f"Episodes: {episodes}")
    print(f"Port: {port}")
    print("="*60)

    from evaluate_ppo import evaluate_model

    try:
        stats = evaluate_model(
            model_path=model_path,
            n_episodes=episodes,
            port=port,
            verbose=True
        )
        return stats
    except Exception as e:
        print(f"\n✗ Evaluation failed: {e}")
        return None


def show_tensorboard():
    """Launch tensorboard."""
    log_dir = "logs/ppo_torcs"
    if not os.path.exists(log_dir):
        print(f"✗ Log directory not found: {log_dir}")
        print("  Train a model first to generate logs")
        return

    print("="*60)
    print("Launching Tensorboard")
    print("="*60)
    print(f"Log directory: {log_dir}")
    print("Open browser to: http://localhost:6006")
    print("Press Ctrl+C to stop tensorboard")
    print("="*60)

    try:
        subprocess.run(['tensorboard', '--logdir', log_dir])
    except KeyboardInterrupt:
        print("\nTensorboard stopped")
    except FileNotFoundError:
        print("\n✗ Tensorboard not found!")
        print("  Install with: pip install tensorboard")


def main():
    parser = argparse.ArgumentParser(
        description='TORCS RL Training Quick Start',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test environment connection
  python run_rl_training.py test

  # Quick training (500k steps) - requires 2 TORCS instances
  python run_rl_training.py train

  # Training with single TORCS instance (no evaluation)
  python run_rl_training.py train --no-eval

  # Extended training (1M steps)
  python run_rl_training.py train --timesteps 1000000

  # Training with target lap time
  python run_rl_training.py train --target-time 90.0

  # Evaluate most recent model
  python run_rl_training.py eval

  # Evaluate specific model
  python run_rl_training.py eval --model path/to/model

  # Launch tensorboard
  python run_rl_training.py tensorboard
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Test command
    test_parser = subparsers.add_parser('test', help='Test environment connection')
    test_parser.add_argument('--port', type=int, default=3001, help='TORCS port')

    # Train command
    train_parser = subparsers.add_parser('train', help='Start training')
    train_parser.add_argument('--timesteps', type=int, default=500000, help='Training timesteps')
    train_parser.add_argument('--port', type=int, default=3001, help='TORCS port')
    train_parser.add_argument('--target-time', type=float, default=None, help='Target lap time')
    train_parser.add_argument('--no-eval', action='store_true',
                              help='Disable evaluation (use if only one TORCS instance)')

    # Eval command
    eval_parser = subparsers.add_parser('eval', help='Evaluate trained model')
    eval_parser.add_argument('--model', type=str, default=None, help='Model path (default: most recent)')
    eval_parser.add_argument('--episodes', type=int, default=10, help='Evaluation episodes')
    eval_parser.add_argument('--port', type=int, default=3001, help='TORCS port')

    # Tensorboard command
    tb_parser = subparsers.add_parser('tensorboard', help='Launch tensorboard')

    args = parser.parse_args()

    # Show help if no command
    if args.command is None:
        parser.print_help()
        return

    # Check dependencies first
    if not check_dependencies():
        return

    # Run command
    if args.command == 'test':
        test_environment(port=args.port)

    elif args.command == 'train':
        quick_train(
            timesteps=args.timesteps,
            port=args.port,
            target_time=args.target_time,
            no_eval=args.no_eval
        )

    elif args.command == 'eval':
        quick_eval(
            model_path=args.model,
            episodes=args.episodes,
            port=args.port
        )

    elif args.command == 'tensorboard':
        show_tensorboard()


if __name__ == "__main__":
    main()
