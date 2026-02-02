"""
Record human demonstrations for imitation learning.

Controls:
    W / Up Arrow    - Accelerate
    S / Down Arrow  - Brake
    A / Left Arrow  - Steer left
    D / Right Arrow - Steer right
    Q              - Quit and save
    R              - Restart episode (discard current)

Requirements:
    pip install keyboard

Usage:
    python record_demonstrations.py --output demos.npz
"""

import os
import argparse
import numpy as np
import time
import keyboard
from torcs_rl_env import TorcsRLEnv


class DemonstrationRecorder:
    """Records human demonstrations from keyboard input."""

    def __init__(self, env, output_path='demonstrations.npz'):
        self.env = env
        self.output_path = output_path

        # Recording buffers
        self.observations = []
        self.actions = []
        self.episode_observations = []
        self.episode_actions = []

        # Current action state
        self.steer = 0.0
        self.accel = 0.0
        self.brake = 0.0

        # Control parameters
        self.steer_speed = 0.08      # How fast steering changes per frame
        self.steer_return = 0.06    # How fast steering returns to center
        self.max_steer = 1.0

        # State
        self.running = True
        self.episode_count = 0
        self.total_steps = 0

    def get_keyboard_action(self):
        """Read keyboard state and return action."""
        # Steering
        left = keyboard.is_pressed('a') or keyboard.is_pressed('left')
        right = keyboard.is_pressed('d') or keyboard.is_pressed('right')

        if left and not right:
            self.steer = max(-self.max_steer, self.steer - self.steer_speed)
        elif right and not left:
            self.steer = min(self.max_steer, self.steer + self.steer_speed)
        else:
            # Return to center
            if abs(self.steer) < self.steer_return:
                self.steer = 0.0
            elif self.steer > 0:
                self.steer -= self.steer_return
            else:
                self.steer += self.steer_return

        # Throttle/Brake
        accel_pressed = keyboard.is_pressed('w') or keyboard.is_pressed('up')
        brake_pressed = keyboard.is_pressed('s') or keyboard.is_pressed('down')

        self.accel = 1.0 if accel_pressed else 0.0
        self.brake = 1.0 if brake_pressed else 0.0

        # Check for quit
        if keyboard.is_pressed('q'):
            self.running = False

        # Check for restart
        restart = keyboard.is_pressed('r')

        return np.array([self.steer, self.accel, self.brake], dtype=np.float32), restart

    def save_episode(self):
        """Save current episode to buffers."""
        if len(self.episode_observations) > 10:  # Only save if meaningful length
            self.observations.extend(self.episode_observations)
            self.actions.extend(self.episode_actions)
            print(f"  Episode saved: {len(self.episode_observations)} steps")
        else:
            print(f"  Episode too short ({len(self.episode_observations)} steps), discarded")

        self.episode_observations = []
        self.episode_actions = []

    def save_to_file(self):
        """Save all demonstrations to file."""
        if len(self.observations) == 0:
            print("No demonstrations to save!")
            return

        obs_array = np.array(self.observations, dtype=np.float32)
        act_array = np.array(self.actions, dtype=np.float32)

        np.savez(
            self.output_path,
            observations=obs_array,
            actions=act_array
        )

        print(f"\nSaved {len(self.observations)} demonstration steps to {self.output_path}")
        print(f"  Observations shape: {obs_array.shape}")
        print(f"  Actions shape: {act_array.shape}")

    def run(self):
        """Main recording loop."""
        print("=" * 60)
        print("DEMONSTRATION RECORDER")
        print("=" * 60)
        print("Controls:")
        print("  W/Up    - Accelerate")
        print("  S/Down  - Brake")
        print("  A/Left  - Steer left")
        print("  D/Right - Steer right")
        print("  Q       - Quit and save")
        print("  R       - Restart episode (discard current)")
        print("=" * 60)
        print("\nStarting in 3 seconds... (focus on TORCS window)")
        time.sleep(3)

        while self.running:
            self.episode_count += 1
            print(f"\n--- Episode {self.episode_count} ---")

            # Reset environment
            obs, info = self.env.reset()
            if info.get('connection_failed'):
                print("Connection failed! Make sure TORCS is running.")
                break

            self.episode_observations = []
            self.episode_actions = []
            self.steer = 0.0

            done = False
            step = 0

            while not done and self.running:
                # Get action from keyboard
                action, restart = self.get_keyboard_action()

                if restart:
                    print("  Restarting episode (discarding)...")
                    self.episode_observations = []
                    self.episode_actions = []
                    time.sleep(0.5)  # Debounce
                    break

                # Record before stepping
                self.episode_observations.append(obs.copy())
                self.episode_actions.append(action.copy())

                # Step environment
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                step += 1
                self.total_steps += 1

                # Display status
                speed = obs[0]
                track_pos = obs[4]
                if step % 50 == 0:
                    print(f"  Step {step}: speed={speed:.1f} km/h, "
                          f"trackPos={track_pos:.2f}, steer={self.steer:.2f}")

                # Small delay to prevent CPU spinning
                time.sleep(0.02)

            # Episode ended
            if done and self.running:
                reason = info.get('dnf_reason', 'completed')
                print(f"  Episode ended: {reason}")

                if reason is None:  # Lap completed successfully
                    lap_time = info.get('lap_time', 0)
                    print(f"  LAP COMPLETED! Time: {lap_time:.2f}s")

                self.save_episode()

        # Save all demonstrations
        self.save_to_file()
        self.env.close()
        print("\nRecording complete!")


def main():
    parser = argparse.ArgumentParser(description='Record demonstrations for TORCS')
    parser.add_argument('--output', type=str, default='demonstrations.npz',
                        help='Output file path')
    parser.add_argument('--port', type=int, default=3001,
                        help='TORCS server port')
    parser.add_argument('--max-steps', type=int, default=15000,
                        help='Max steps per episode')
    args = parser.parse_args()

    # Create environment
    env = TorcsRLEnv(
        port=args.port,
        max_steps=args.max_steps,
        target_laps=1,
        off_track_threshold=1.5,  # More lenient for human control
        launch_assist=False  # Disable launch assist for human control
    )

    # Create recorder and run
    recorder = DemonstrationRecorder(env, output_path=args.output)

    try:
        recorder.run()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        recorder.save_to_file()
        env.close()


if __name__ == "__main__":
    main()
