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
        obs, info = env.reset()
        if info.get('connection_failed', False):
            print("✗ Reset failed - connection issue")
            env.close()
            return False
        print(f"✓ Reset successful, obs shape: {obs.shape}")

        print("\nTesting 10 random steps...")
        for i in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
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
        print(f"Log directory not found: {log_dir}")
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
        print("\nTensorboard not found!")
        print("  Install with: pip install tensorboard")


def collect_demos(laps=5, port=3001, output='demos/ga_demos.npz'):
    """Collect demonstrations from the state machine controller."""
    print("="*60)
    print("Collecting Demonstrations from State Machine Controller")
    print("="*60)
    print(f"Laps to collect: {laps}")
    print(f"Port: {port}")
    print(f"Output: {output}")
    print("="*60)
    print("\nMake sure TORCS is running with scr_server driver!")
    print("Collection will start in 3 seconds...")

    import time
    time.sleep(3)

    from collect_ga_demos import collect_demonstrations, save_demonstrations

    data = collect_demonstrations(n_laps=laps, port=port, verbose=True)

    if data['total_steps'] == 0:
        print("\nNo demonstrations collected!")
        return None

    save_demonstrations(data, output)
    return output


def pretrain_bc(demos_path='demos/ga_demos.npz', output_dir='models/pretrained',
                epochs=100, batch_size=64):
    """Pretrain policy using behavioral cloning."""
    if not os.path.exists(demos_path):
        print(f"Demonstrations file not found: {demos_path}")
        print("  Run 'python run_rl_training.py collect' first")
        return None

    print("="*60)
    print("Pretraining Policy with Behavioral Cloning")
    print("="*60)
    print(f"Demonstrations: {demos_path}")
    print(f"Output: {output_dir}")
    print(f"Epochs: {epochs}")
    print("="*60)

    from pretrain_bc import pretrain_ppo_policy

    model = pretrain_ppo_policy(
        demo_path=demos_path,
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size
    )

    return os.path.join(output_dir, 'best_bc_model')


def test_bc_model(model_path='models/pretrained/best_bc_model', port=3001, max_steps=10000):
    """
    Test if the BC pretrained model can drive around the track.
    """
    print("="*60)
    print("Testing BC Pretrained Model")
    print("="*60)
    print(f"Model: {model_path}")
    print(f"Port: {port}")
    print("="*60)

    if not os.path.exists(model_path + '.zip') and not os.path.exists(model_path):
        print(f"\nModel not found: {model_path}")
        print("Run 'python run_rl_training.py pretrain' first")
        return None

    from stable_baselines3 import PPO
    from torcs_rl_env import TorcsRLEnv

    print("\nLoading model...")
    model = PPO.load(model_path)

    print("Creating environment...")
    env = TorcsRLEnv(port=port, max_steps=max_steps)

    print("Resetting environment...")
    obs, info = env.reset()

    if info.get('connection_failed'):
        print("Failed to connect to TORCS!")
        return None

    print("\nRunning BC model (deterministic)...")
    print("-" * 40)

    step = 0
    total_reward = 0
    lap_completed = False

    try:
        while step < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step += 1

            # Progress update every 500 steps
            if step % 500 == 0:
                speed = info.get('speed', 0)
                dist = info.get('dist_raced', 0)
                track_pos = info.get('trackPos', 0)
                print(f"  Step {step}: speed={speed:.1f} km/h, dist={dist:.1f}m, "
                      f"trackPos={track_pos:.2f}, reward={total_reward:.1f}")

            if terminated or truncated:
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        env.close()

    # Results
    print("-" * 40)
    dist_raced = info.get('dist_raced', 0)
    dnf = info.get('dnf', False)
    dnf_reason = info.get('dnf_reason', None)
    lap_count = info.get('lap_count', 0)

    print(f"\nResults:")
    print(f"  Steps: {step}")
    print(f"  Distance: {dist_raced:.1f}m")
    print(f"  Total reward: {total_reward:.1f}")
    print(f"  Laps completed: {lap_count}")

    if lap_count > 0:
        print(f"\n  SUCCESS! BC model completed {lap_count} lap(s)")
        print("  The model should work well for RL fine-tuning.")
    elif dnf:
        print(f"\n  DNF: {dnf_reason}")
        if dnf_reason == 'off_track':
            print("  The model is going off track - BC may need more demos or epochs")
        elif dnf_reason == 'facing_backward':
            print("  The model is spinning - steering may be miscalibrated")
        elif dnf_reason == 'stalled_start':
            print("  The model isn't accelerating properly")
    else:
        print(f"\n  No lap completed in {max_steps} steps")
        print("  The model may be too slow or stuck")

    print("="*60)
    return {
        'steps': step,
        'distance': dist_raced,
        'reward': total_reward,
        'lap_count': lap_count,
        'dnf': dnf,
        'dnf_reason': dnf_reason
    }


def imitation_train(laps=5, port=3001, timesteps=500000, no_eval=False):
    """
    Full imitation learning pipeline:
    1. Collect demonstrations from state machine controller
    2. Pretrain policy with behavioral cloning
    3. Continue training with reinforcement learning
    """
    print("="*60)
    print("IMITATION LEARNING PIPELINE")
    print("="*60)
    print("This will:")
    print("  1. Collect demonstrations from the state machine controller")
    print("  2. Pretrain the RL policy to imitate those demonstrations")
    print("  3. Continue training with reinforcement learning")
    print("="*60)

    # Step 1: Collect demonstrations
    print("\n[Step 1/3] Collecting demonstrations...")
    demos_path = collect_demos(laps=laps, port=port)
    if demos_path is None:
        print("Failed to collect demonstrations. Aborting.")
        return

    # Step 2: Pretrain with BC
    print("\n[Step 2/3] Pretraining with behavioral cloning...")
    pretrained_model = pretrain_bc(demos_path=demos_path)
    if pretrained_model is None:
        print("Failed to pretrain. Aborting.")
        return

    # Step 3: Continue with RL
    print("\n[Step 3/3] Continuing with reinforcement learning...")
    print("Starting RL training in 3 seconds...")

    import time
    time.sleep(3)

    from train_ppo import resume_training

    try:
        model, save_dir = resume_training(
            model_path=pretrained_model,
            total_timesteps=timesteps,
            port=port,
            verbose=1
        )
        print(f"\nTraining completed!")
        print(f"  Model saved to: {save_dir}")
    except KeyboardInterrupt:
        print("\n\nTraining stopped by user")
    except Exception as e:
        print(f"\nTraining error: {e}")


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

  # IMITATION LEARNING (recommended for initial training):
  # Collect demonstrations from state machine controller
  python run_rl_training.py collect --laps 5

  # Pretrain policy from demonstrations
  python run_rl_training.py pretrain --demos demos/ga_demos.npz

  # Full imitation learning pipeline (collect + pretrain + train)
  python run_rl_training.py imitate --laps 5 --timesteps 500000
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Test command
    test_parser = subparsers.add_parser('test', help='Test environment connection')
    test_parser.add_argument('--port', type=int, default=3001, help='TORCS port')

    # Train command
    train_parser = subparsers.add_parser('train', help='Start training from scratch')
    train_parser.add_argument('--timesteps', type=int, default=500000, help='Training timesteps')
    train_parser.add_argument('--port', type=int, default=3001, help='TORCS port')
    train_parser.add_argument('--target-time', type=float, default=None, help='Target lap time')
    train_parser.add_argument('--no-eval', action='store_true',
                              help='Disable evaluation (use if only one TORCS instance)')

    # Collect demonstrations command
    collect_parser = subparsers.add_parser('collect', help='Collect demonstrations from state machine controller')
    collect_parser.add_argument('--laps', type=int, default=5, help='Number of laps to collect (default: 5)')
    collect_parser.add_argument('--port', type=int, default=3001, help='TORCS port')
    collect_parser.add_argument('--output', type=str, default='demos/ga_demos.npz', help='Output file path')

    # Pretrain command
    pretrain_parser = subparsers.add_parser('pretrain', help='Pretrain policy from demonstrations')
    pretrain_parser.add_argument('--demos', type=str, default='demos/ga_demos.npz', help='Demonstrations file')
    pretrain_parser.add_argument('--output', type=str, default='models/pretrained', help='Output directory')
    pretrain_parser.add_argument('--epochs', type=int, default=100, help='Training epochs (default: 100)')

    # Test BC model command
    test_bc_parser = subparsers.add_parser('test-bc', help='Test if BC pretrained model can drive')
    test_bc_parser.add_argument('--model', type=str, default='models/pretrained/best_bc_model', help='Model path')
    test_bc_parser.add_argument('--port', type=int, default=3001, help='TORCS port')
    test_bc_parser.add_argument('--max-steps', type=int, default=10000, help='Max steps to run')

    # Imitate command (full pipeline)
    imitate_parser = subparsers.add_parser('imitate', help='Full imitation learning pipeline (collect + pretrain + train)')
    imitate_parser.add_argument('--laps', type=int, default=5, help='Number of demonstration laps (default: 5)')
    imitate_parser.add_argument('--timesteps', type=int, default=500000, help='RL training timesteps')
    imitate_parser.add_argument('--port', type=int, default=3001, help='TORCS port')
    imitate_parser.add_argument('--no-eval', action='store_true',
                                help='Disable evaluation during RL training')

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

    elif args.command == 'collect':
        collect_demos(
            laps=args.laps,
            port=args.port,
            output=args.output
        )

    elif args.command == 'pretrain':
        pretrain_bc(
            demos_path=args.demos,
            output_dir=args.output,
            epochs=args.epochs
        )

    elif args.command == 'test-bc':
        test_bc_model(
            model_path=args.model,
            port=args.port,
            max_steps=args.max_steps
        )

    elif args.command == 'imitate':
        imitation_train(
            laps=args.laps,
            port=args.port,
            timesteps=args.timesteps,
            no_eval=args.no_eval
        )


if __name__ == "__main__":
    main()
