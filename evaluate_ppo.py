"""
Evaluation script for trained PPO models on TORCS.

Load a trained model and evaluate its performance over multiple episodes.
"""

import os
import argparse
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from torcs_rl_env import TorcsRLEnv


def evaluate_model(
    model_path,
    n_episodes=10,
    port=3001,
    max_steps=15000,
    render=False,
    verbose=True,
    save_replay=False
):
    """
    Evaluate a trained PPO model.

    Args:
        model_path: Path to saved model (.zip file)
        n_episodes: Number of evaluation episodes
        port: TORCS server port
        max_steps: Maximum steps per episode
        render: Whether to render (not implemented for TORCS)
        verbose: Print detailed information
        save_replay: Save replay data (not implemented)

    Returns:
        Dictionary with evaluation statistics
    """
    # Check if model exists
    if not os.path.exists(model_path):
        # Try adding .zip extension
        if os.path.exists(model_path + '.zip'):
            model_path = model_path + '.zip'
        else:
            raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"{'='*60}")
    print(f"Evaluating PPO Model")
    print(f"{'='*60}")
    print(f"Model: {model_path}")
    print(f"Episodes: {n_episodes}")
    print(f"Port: {port}")
    print(f"{'='*60}\n")

    # Load model
    print("Loading model...")
    model = PPO.load(model_path)

    # Create environment with Monitor wrapper
    def make_env():
        env = TorcsRLEnv(
            port=port,
            max_steps=max_steps,
            target_laps=1,
            reward_type='progress'
        )
        # Wrap with Monitor for episode statistics
        env = Monitor(env)
        return env

    env = DummyVecEnv([make_env])

    # Try to load normalization stats from multiple possible locations
    # Handle different directory structures (e.g., best_model/best_model.zip)
    possible_paths = [
        os.path.join(os.path.dirname(model_path), 'vec_normalize.pkl'),
        os.path.join(os.path.dirname(os.path.dirname(model_path)), 'vec_normalize.pkl'),
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(model_path))), 'vec_normalize.pkl'),
    ]

    vec_normalize_loaded = False
    for vec_normalize_path in possible_paths:
        if os.path.exists(vec_normalize_path):
            print(f"Loading normalization stats from: {vec_normalize_path}")
            env = VecNormalize.load(vec_normalize_path, env)
            env.training = False  # Don't update stats during evaluation
            env.norm_reward = False  # Don't normalize rewards
            vec_normalize_loaded = True
            break

    if not vec_normalize_loaded:
        print("Warning: No normalization stats found - model may not perform correctly")

    # Evaluation loop
    episode_rewards = []
    episode_lengths = []
    lap_times = []
    dnf_count = 0
    completed_count = 0

    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        episode_lap_time = None
        episode_dnf = False
        episode_dnf_reason = None

        if verbose:
            print(f"\n--- Episode {episode + 1}/{n_episodes} ---")

        while not done:
            # Get action from model
            action, _states = model.predict(obs, deterministic=True)

            # Step environment
            obs, reward, done, info = env.step(action)

            episode_reward += reward[0]
            episode_length += 1

            # Extract info
            if len(info) > 0:
                ep_info = info[0]

                # Check for lap completion
                if 'lap_time' in ep_info:
                    episode_lap_time = ep_info['lap_time']
                    if verbose:
                        print(f"  Lap completed: {episode_lap_time:.3f}s")

                # Check for DNF
                if ep_info.get('dnf', False):
                    episode_dnf = True
                    episode_dnf_reason = ep_info.get('dnf_reason', 'unknown')

                # Verbose output during episode
                if verbose and episode_length % 100 == 0:
                    speed = ep_info.get('speed', 0)
                    dist = ep_info.get('dist_raced', 0)
                    track_pos = ep_info.get('trackPos', 0)
                    print(f"  Step {episode_length}: speed={speed:.1f} km/h, dist={dist:.1f}m, trackPos={track_pos:.2f}")

        # Store episode results
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        if episode_lap_time is not None:
            lap_times.append(episode_lap_time)
            completed_count += 1
        else:
            if episode_dnf:
                dnf_count += 1

        # Print episode summary
        if verbose:
            print(f"\nEpisode {episode + 1} Summary:")
            print(f"  Total reward: {episode_reward:.2f}")
            print(f"  Episode length: {episode_length} steps")
            if episode_lap_time is not None:
                print(f"  Lap time: {episode_lap_time:.3f}s")
            elif episode_dnf:
                print(f"  DNF: {episode_dnf_reason}")
            else:
                print(f"  Did not complete lap")
            print()

    # Calculate statistics
    stats = {
        'n_episodes': n_episodes,
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths),
        'completed_count': completed_count,
        'dnf_count': dnf_count,
        'completion_rate': completed_count / n_episodes * 100,
    }

    if lap_times:
        stats['best_lap_time'] = np.min(lap_times)
        stats['mean_lap_time'] = np.mean(lap_times)
        stats['std_lap_time'] = np.std(lap_times)
        stats['lap_times'] = lap_times

    # Print final statistics
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Episodes evaluated: {n_episodes}")
    print(f"Completed laps: {completed_count}/{n_episodes} ({stats['completion_rate']:.1f}%)")
    print(f"DNFs: {dnf_count}/{n_episodes}")
    print(f"\nReward Statistics:")
    print(f"  Mean: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
    print(f"\nEpisode Length:")
    print(f"  Mean: {stats['mean_length']:.1f} ± {stats['std_length']:.1f} steps")

    if lap_times:
        print(f"\nLap Time Statistics:")
        print(f"  Best: {stats['best_lap_time']:.3f}s")
        print(f"  Mean: {stats['mean_lap_time']:.3f}s ± {stats['std_lap_time']:.3f}s")
        print(f"  All lap times: {[f'{t:.3f}' for t in lap_times]}")

    print(f"{'='*60}\n")

    # Clean up
    env.close()

    return stats


def compare_models(model_paths, n_episodes=5, port=3001):
    """
    Compare multiple trained models.

    Args:
        model_paths: List of paths to models
        n_episodes: Number of episodes per model
        port: TORCS server port

    Returns:
        Dictionary mapping model names to statistics
    """
    results = {}

    print(f"{'='*60}")
    print(f"Comparing {len(model_paths)} models")
    print(f"{'='*60}\n")

    for i, model_path in enumerate(model_paths):
        model_name = os.path.basename(model_path)
        print(f"\n[{i+1}/{len(model_paths)}] Evaluating: {model_name}")
        print(f"-" * 60)

        try:
            stats = evaluate_model(
                model_path=model_path,
                n_episodes=n_episodes,
                port=port,
                verbose=False
            )
            results[model_name] = stats
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
            continue

    # Print comparison
    print(f"\n{'='*60}")
    print(f"MODEL COMPARISON")
    print(f"{'='*60}\n")

    # Sort by best lap time
    models_with_laps = [(name, stats) for name, stats in results.items() if 'best_lap_time' in stats]
    models_with_laps.sort(key=lambda x: x[1]['best_lap_time'])

    print(f"{'Model':<40} {'Best Lap':>12} {'Mean Lap':>12} {'Completion':>12}")
    print(f"{'-'*40} {'-'*12} {'-'*12} {'-'*12}")

    for name, stats in models_with_laps:
        best_lap = f"{stats['best_lap_time']:.3f}s"
        mean_lap = f"{stats['mean_lap_time']:.3f}s"
        completion = f"{stats['completion_rate']:.1f}%"
        print(f"{name:<40} {best_lap:>12} {mean_lap:>12} {completion:>12}")

    # Models that didn't complete any laps
    models_no_laps = [(name, stats) for name, stats in results.items() if 'best_lap_time' not in stats]
    if models_no_laps:
        print(f"\nModels with no completed laps:")
        for name, stats in models_no_laps:
            print(f"  {name}: {stats['dnf_count']} DNFs")

    print(f"\n{'='*60}\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate trained PPO model for TORCS')
    parser.add_argument('model_path', type=str, help='Path to trained model')
    parser.add_argument('--episodes', type=int, default=10, help='Number of evaluation episodes')
    parser.add_argument('--port', type=int, default=3001, help='TORCS server port')
    parser.add_argument('--max-steps', type=int, default=15000, help='Maximum steps per episode')
    parser.add_argument('--quiet', action='store_true', help='Reduce verbosity')
    parser.add_argument('--compare', nargs='+', help='Compare multiple models')

    args = parser.parse_args()

    if args.compare:
        # Compare multiple models
        compare_models(
            model_paths=args.compare,
            n_episodes=args.episodes,
            port=args.port
        )
    else:
        # Evaluate single model
        evaluate_model(
            model_path=args.model_path,
            n_episodes=args.episodes,
            port=args.port,
            max_steps=args.max_steps,
            verbose=not args.quiet
        )
