"""
PPO Training Script for TORCS Lap Time Optimization

Train a deep RL agent to minimize lap time from standing start.
Uses Stable-Baselines3 PPO with custom callbacks for monitoring.
"""

import os
import argparse
from datetime import datetime
import numpy as np

# Stable-baselines3 imports
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

# Local imports
from torcs_rl_env import TorcsRLEnv


class LapTimeCallback(BaseCallback):
    """
    Custom callback for logging lap times and best performance.
    Stops training when target lap time is achieved.
    """

    def __init__(self, target_lap_time=None, verbose=1):
        super(LapTimeCallback, self).__init__(verbose)
        self.target_lap_time = target_lap_time
        self.best_lap_time = None
        self.episode_lap_times = []
        self.episode_count = 0

    def _on_step(self) -> bool:
        # Access info from the environment
        infos = self.locals.get('infos', [])
        if len(infos) > 0:
            info = infos[0]

            # Check if lap completed
            if 'lap_time' in info:
                lap_time = info['lap_time']
                self.episode_lap_times.append(lap_time)

                # Update best lap time
                if self.best_lap_time is None or lap_time < self.best_lap_time:
                    self.best_lap_time = lap_time
                    if self.verbose > 0:
                        print(f"\n{'='*60}")
                        print(f"NEW BEST LAP TIME: {lap_time:.3f}s at step {self.num_timesteps}")
                        print(f"{'='*60}\n")

                # Log to tensorboard
                self.logger.record('lap_time/last', lap_time)
                self.logger.record('lap_time/best', self.best_lap_time)

                # Check if target achieved
                if self.target_lap_time is not None and lap_time <= self.target_lap_time:
                    if self.verbose > 0:
                        print(f"\n{'='*60}")
                        print(f"TARGET LAP TIME ACHIEVED: {lap_time:.3f}s <= {self.target_lap_time:.3f}s")
                        print(f"Training stopped early!")
                        print(f"{'='*60}\n")
                    return False  # Stop training

            # Log episode info from Monitor wrapper (available in 'episode' key of info)
            if 'episode' in info:
                episode_info = info['episode']
                if 'r' in episode_info:
                    self.logger.record('episode/reward', episode_info['r'])
                if 'l' in episode_info:
                    self.logger.record('episode/length', episode_info['l'])

            # Log DNF info
            if info.get('dnf', False):
                dnf_reason = info.get('dnf_reason', 'unknown')
                if self.verbose > 1:
                    print(f"DNF at step {self.num_timesteps}: {dnf_reason}")

        return True

    def _on_training_end(self) -> None:
        """Print summary at end of training."""
        if self.verbose > 0:
            print(f"\n{'='*60}")
            print("TRAINING SUMMARY")
            print(f"{'='*60}")
            print(f"Total timesteps: {self.num_timesteps}")
            if self.best_lap_time is not None:
                print(f"Best lap time: {self.best_lap_time:.3f}s")
            print(f"Total laps completed: {len(self.episode_lap_times)}")
            if self.episode_lap_times:
                print(f"Average lap time: {np.mean(self.episode_lap_times):.3f}s")
                print(f"Lap time std: {np.std(self.episode_lap_times):.3f}s")
            print(f"{'='*60}\n")


def make_env(port=3001, max_steps=15000, reward_type='progress', monitor_dir=None):
    """
    Create and return a TORCS environment wrapped with Monitor.

    Args:
        port: TORCS server port
        max_steps: Maximum steps per episode
        reward_type: 'progress' or 'time'
        monitor_dir: Directory for Monitor logs (optional)

    Returns:
        A function that creates the wrapped environment
    """
    def _init():
        env = TorcsRLEnv(
            port=port,
            max_steps=max_steps,
            target_laps=1,
            off_track_threshold=1.3,
            start_min_distance=25,
            start_check_steps=250,
            reward_type=reward_type
        )
        # Wrap with Monitor for episode statistics tracking
        # This is required for EvalCallback to properly identify best models
        env = Monitor(env, filename=monitor_dir)
        return env
    return _init


def train_ppo(
    total_timesteps=500000,
    port=3001,
    save_dir='models/ppo_torcs',
    log_dir='logs/ppo_torcs',
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    target_lap_time=None,
    reward_type='progress',
    normalize=True,
    eval_freq=10000,
    save_freq=20000,
    eval_port=None,
    verbose=1
):
    """
    Train a PPO agent for TORCS lap time optimization.

    Args:
        total_timesteps: Total training steps
        port: TORCS server port for training
        save_dir: Directory to save models
        log_dir: Directory for tensorboard logs
        learning_rate: PPO learning rate
        n_steps: Steps per rollout
        batch_size: Minibatch size
        n_epochs: Optimization epochs per rollout
        gamma: Discount factor
        gae_lambda: GAE lambda
        clip_range: PPO clip range
        ent_coef: Entropy coefficient
        vf_coef: Value function coefficient
        max_grad_norm: Max gradient norm
        target_lap_time: Stop training when this lap time is achieved
        reward_type: 'progress' or 'time'
        normalize: Use VecNormalize wrapper
        eval_freq: Evaluation frequency (steps). Set to 0 to disable evaluation.
        save_freq: Checkpoint save frequency (steps)
        eval_port: Port for evaluation environment. If None, uses port+1.
                   Set eval_freq=0 if you only have one TORCS instance.
        verbose: Verbosity level

    Returns:
        Tuple of (trained PPO model, save directory path)

    Note:
        By default, this requires TWO TORCS instances running:
        - Training instance on `port` (default 3001)
        - Evaluation instance on `eval_port` (default port+1 = 3002)

        If you only have one TORCS instance, set eval_freq=0 to disable
        periodic evaluation during training.
    """
    # Default eval_port to port+1
    if eval_port is None:
        eval_port = port + 1
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"ppo_{timestamp}"
    run_save_dir = os.path.join(save_dir, run_name)
    run_log_dir = os.path.join(log_dir, run_name)
    os.makedirs(run_save_dir, exist_ok=True)

    print(f"{'='*60}")
    print(f"PPO Training for TORCS Lap Time Optimization")
    print(f"{'='*60}")
    print(f"Run name: {run_name}")
    print(f"Total timesteps: {total_timesteps}")
    print(f"Training port: {port}")
    if eval_freq > 0:
        print(f"Evaluation port: {eval_port}")
        print(f"  NOTE: Requires second TORCS instance on port {eval_port}")
    else:
        print(f"Evaluation: DISABLED (eval_freq=0)")
    print(f"Reward type: {reward_type}")
    print(f"Normalize observations: {normalize}")
    if target_lap_time:
        print(f"Target lap time: {target_lap_time:.3f}s")
    print(f"Save directory: {run_save_dir}")
    print(f"Log directory: {run_log_dir}")
    print(f"{'='*60}\n")

    # Create monitor directory for episode logs
    monitor_dir = os.path.join(run_save_dir, 'monitor')
    os.makedirs(monitor_dir, exist_ok=True)

    # Create environment with Monitor wrapper
    env = DummyVecEnv([make_env(port=port, reward_type=reward_type,
                                monitor_dir=os.path.join(monitor_dir, 'train'))])

    # Normalize observations and rewards if requested
    if normalize:
        env = VecNormalize(
            env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=gamma
        )

    # Create evaluation environment (only if eval_freq > 0)
    eval_env = None
    if eval_freq > 0:
        eval_env = DummyVecEnv([make_env(port=eval_port, reward_type=reward_type,
                                         monitor_dir=os.path.join(monitor_dir, 'eval'))])
        if normalize:
            eval_env = VecNormalize(
                eval_env,
                norm_obs=True,
                norm_reward=False,  # Don't normalize rewards for evaluation
                clip_obs=10.0,
                gamma=gamma,
                training=False  # Don't update normalization stats during eval
            )

    # Create PPO model
    print("Creating PPO model...")
    model = PPO(
        'MlpPolicy',
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        verbose=verbose,
        tensorboard_log=run_log_dir,
        policy_kwargs=dict(
            net_arch=[dict(pi=[256, 256], vf=[256, 256])]  # Separate networks for policy and value
        )
    )

    print(f"Model architecture: {model.policy}")
    print(f"{'='*60}\n")

    # Create callbacks
    callbacks = []

    # Lap time monitoring callback
    lap_time_callback = LapTimeCallback(
        target_lap_time=target_lap_time,
        verbose=verbose
    )
    callbacks.append(lap_time_callback)

    # Evaluation callback (only if eval environment is available)
    if eval_env is not None:
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(run_save_dir, 'best_model'),
            log_path=os.path.join(run_save_dir, 'eval_logs'),
            eval_freq=eval_freq,
            deterministic=True,
            render=False,
            verbose=verbose
        )
        callbacks.append(eval_callback)

    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=os.path.join(run_save_dir, 'checkpoints'),
        name_prefix='ppo_model',
        save_replay_buffer=False,
        save_vecnormalize=normalize
    )
    callbacks.append(checkpoint_callback)

    # Train the model
    print("Starting training...")
    print(f"{'='*60}\n")

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            log_interval=10,
            tb_log_name="ppo",
            reset_num_timesteps=True
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
    except Exception as e:
        print(f"\n\nTraining error: {e}")
        import traceback
        traceback.print_exc()

    # Save final model
    final_model_path = os.path.join(run_save_dir, 'final_model')
    model.save(final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")

    # Save normalization stats if used
    if normalize:
        env.save(os.path.join(run_save_dir, 'vec_normalize.pkl'))
        print(f"Normalization stats saved to: {os.path.join(run_save_dir, 'vec_normalize.pkl')}")

    # Print training summary
    if lap_time_callback.best_lap_time:
        print(f"\n{'='*60}")
        print(f"TRAINING COMPLETED")
        print(f"{'='*60}")
        print(f"Best lap time achieved: {lap_time_callback.best_lap_time:.3f}s")
        print(f"Total laps completed: {len(lap_time_callback.episode_lap_times)}")
        if lap_time_callback.episode_lap_times:
            recent_laps = lap_time_callback.episode_lap_times[-10:]
            print(f"Recent 10 laps average: {np.mean(recent_laps):.3f}s")
        print(f"{'='*60}\n")

    # Clean up
    env.close()
    if eval_env is not None:
        eval_env.close()

    return model, run_save_dir


def resume_training(
    model_path,
    total_timesteps=500000,
    port=3001,
    target_lap_time=None,
    save_freq=20000,
    verbose=1,
    **kwargs
):
    """
    Resume training from a saved model.

    Args:
        model_path: Path to saved model (with or without .zip extension)
        total_timesteps: Additional timesteps to train
        port: TORCS server port
        target_lap_time: Stop training when this lap time is achieved
        save_freq: Checkpoint save frequency (steps)
        verbose: Verbosity level
        **kwargs: Additional arguments (reward_type, normalize)

    Returns:
        Trained model
    """
    # Normalize model path - handle both with and without .zip
    if not model_path.endswith('.zip'):
        if os.path.exists(model_path + '.zip'):
            model_path = model_path + '.zip'
        elif not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"{'='*60}")
    print(f"Resuming PPO Training")
    print(f"{'='*60}")
    print(f"Loading model from: {model_path}")

    # Load the model
    model = PPO.load(model_path)

    # Create environment
    reward_type = kwargs.get('reward_type', 'progress')
    normalize = kwargs.get('normalize', True)

    # Create save directory for resumed training
    model_dir = os.path.dirname(model_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    resume_save_dir = os.path.join(model_dir, f'resumed_{timestamp}')
    os.makedirs(resume_save_dir, exist_ok=True)

    # Create monitor directory
    monitor_dir = os.path.join(resume_save_dir, 'monitor')
    os.makedirs(monitor_dir, exist_ok=True)

    env = DummyVecEnv([make_env(port=port, reward_type=reward_type,
                                monitor_dir=os.path.join(monitor_dir, 'train'))])

    if normalize:
        # Try to load normalization stats from multiple possible locations
        vec_normalize_paths = [
            os.path.join(os.path.dirname(model_path), 'vec_normalize.pkl'),
            os.path.join(os.path.dirname(os.path.dirname(model_path)), 'vec_normalize.pkl'),
        ]
        loaded = False
        for vec_normalize_path in vec_normalize_paths:
            if os.path.exists(vec_normalize_path):
                env = VecNormalize.load(vec_normalize_path, env)
                print(f"Loaded normalization stats from: {vec_normalize_path}")
                loaded = True
                break
        if not loaded:
            print("Warning: No normalization stats found, creating new VecNormalize")
            env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)

    # Set the environment
    model.set_env(env)

    # Create callbacks for resumed training
    callbacks = []

    # Lap time monitoring callback
    lap_time_callback = LapTimeCallback(
        target_lap_time=target_lap_time,
        verbose=verbose
    )
    callbacks.append(lap_time_callback)

    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=os.path.join(resume_save_dir, 'checkpoints'),
        name_prefix='ppo_model_resumed',
        save_replay_buffer=False,
        save_vecnormalize=normalize
    )
    callbacks.append(checkpoint_callback)

    # Continue training
    print(f"Total timesteps: {total_timesteps}")
    print(f"Port: {port}")
    if target_lap_time:
        print(f"Target lap time: {target_lap_time:.3f}s")
    print(f"Save directory: {resume_save_dir}")
    print(f"{'='*60}\n")

    print("Resuming training...")

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            reset_num_timesteps=False,
            log_interval=10
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
    except Exception as e:
        print(f"\n\nTraining error: {e}")
        import traceback
        traceback.print_exc()

    # Save updated model
    final_model_path = os.path.join(resume_save_dir, 'final_model')
    model.save(final_model_path)
    print(f"\nResumed model saved to: {final_model_path}")

    # Save normalization stats
    if normalize:
        vec_normalize_save_path = os.path.join(resume_save_dir, 'vec_normalize.pkl')
        env.save(vec_normalize_save_path)
        print(f"Normalization stats saved to: {vec_normalize_save_path}")

    # Print training summary
    if lap_time_callback.best_lap_time:
        print(f"\n{'='*60}")
        print(f"RESUMED TRAINING COMPLETED")
        print(f"{'='*60}")
        print(f"Best lap time achieved: {lap_time_callback.best_lap_time:.3f}s")
        print(f"Total laps completed: {len(lap_time_callback.episode_lap_times)}")
        print(f"{'='*60}\n")

    env.close()
    return model, resume_save_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train PPO agent for TORCS')
    parser.add_argument('--timesteps', type=int, default=500000, help='Total training timesteps')
    parser.add_argument('--port', type=int, default=3001, help='TORCS server port')
    parser.add_argument('--eval-port', type=int, default=None, help='Evaluation server port (default: port+1)')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--target-time', type=float, default=None, help='Target lap time (stop when achieved)')
    parser.add_argument('--reward-type', type=str, default='progress', choices=['progress', 'time'], help='Reward type')
    parser.add_argument('--no-normalize', action='store_true', help='Disable observation normalization')
    parser.add_argument('--no-eval', action='store_true', help='Disable evaluation (use if only one TORCS instance)')
    parser.add_argument('--resume', type=str, default=None, help='Resume from model path')
    parser.add_argument('--save-dir', type=str, default='models/ppo_torcs', help='Save directory')
    parser.add_argument('--log-dir', type=str, default='logs/ppo_torcs', help='Log directory')

    args = parser.parse_args()

    if args.resume:
        # Resume training
        resume_training(
            model_path=args.resume,
            total_timesteps=args.timesteps,
            port=args.port,
            target_lap_time=args.target_time,
            reward_type=args.reward_type,
            normalize=not args.no_normalize,
            verbose=1
        )
    else:
        # Start new training
        train_ppo(
            total_timesteps=args.timesteps,
            port=args.port,
            eval_port=args.eval_port,
            save_dir=args.save_dir,
            log_dir=args.log_dir,
            learning_rate=args.lr,
            target_lap_time=args.target_time,
            reward_type=args.reward_type,
            normalize=not args.no_normalize,
            eval_freq=0 if args.no_eval else 10000,
            verbose=1
        )
