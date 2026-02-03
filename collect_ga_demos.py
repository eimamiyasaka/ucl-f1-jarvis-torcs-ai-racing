"""
Collect demonstrations from the GA/State Machine controller for imitation learning.

This script runs the existing state machine controller and records (observation, action)
pairs in a format compatible with pretrain_bc.py. The RL agent can then be pre-trained
to imitate this controller before fine-tuning with reinforcement learning.

Usage:
    python collect_ga_demos.py --laps 5 --output demos/ga_demos.npz
    python pretrain_bc.py --demos demos/ga_demos.npz --output models/pretrained
    python train_ppo.py --resume models/pretrained/best_bc_model
"""

import os
import argparse
import numpy as np
from datetime import datetime

from torcs_jm_par import (
    Client, LapTimeTracker, drive_modular
)
import torcs_jm_par as torcs_module
from baseline_runner import get_default_params


def build_observation(S):
    """
    Build observation vector matching TorcsRLEnv format (30 dimensions).

    Components:
        - speedX, speedY, speedZ (3)
        - angle (1)
        - trackPos (1)
        - 19 track edge sensors (19)
        - rpm (1)
        - gear (1)
        - wheelSpinVel (4)
    """
    obs = np.array([
        S.get('speedX', 0),
        S.get('speedY', 0),
        S.get('speedZ', 0),
        S.get('angle', 0),
        S.get('trackPos', 0),
        *S.get('track', [100.0] * 19)[:19],
        S.get('rpm', 0),
        S.get('gear', 1),
        *S.get('wheelSpinVel', [0, 0, 0, 0])[:4]
    ], dtype=np.float32)

    return obs


def apply_params(params):
    """Apply driving parameters to the torcs module."""
    torcs_module.TARGET_SPEED = params.get('TARGET_SPEED', 180)
    torcs_module.STEER_GAIN = params.get('STEER_GAIN', 50)
    torcs_module.CENTERING_GAIN = params.get('CENTERING_GAIN', 0.60)
    torcs_module.BRAKE_THRESHOLD = params.get('BRAKE_THRESHOLD', 0.5)
    torcs_module.GEAR_SPEEDS = params.get('GEAR_SPEEDS', [0, 50, 80, 120, 150, 200])
    torcs_module.ENABLE_TRACTION_CONTROL = params.get('ENABLE_TRACTION_CONTROL', True)
    torcs_module.THROTTLE_INCREASE = params.get('THROTTLE_INCREASE', 0.4)
    torcs_module.THROTTLE_DECREASE = params.get('THROTTLE_DECREASE', 0.2)
    torcs_module.SPEED_STEER_FACTOR = params.get('SPEED_STEER_FACTOR', 2.5)
    torcs_module.BRAKE_INTENSITY = params.get('BRAKE_INTENSITY', 0.3)
    torcs_module.TC_THRESHOLD = params.get('TC_THRESHOLD', 2)
    torcs_module.TC_REDUCTION = params.get('TC_REDUCTION', 0.1)


def collect_demonstrations(
    n_laps=5,
    port=3001,
    max_steps_per_lap=15000,
    params=None,
    verbose=True
):
    """
    Collect demonstration data from the legacy modular controller.

    Args:
        n_laps: Number of laps to collect
        port: TORCS server port
        max_steps_per_lap: Maximum steps per lap attempt
        params: Optional custom parameters dict (uses get_default_params() if None)
        verbose: Print progress

    Returns:
        dict with:
            - observations: np.array of shape (N, 30)
            - actions: np.array of shape (N, 3) [steer, accel, brake]
            - lap_times: list of completed lap times
            - total_steps: total number of steps collected
    """
    # Use default params if not provided
    if params is None:
        params = get_default_params()

    # Apply parameters to torcs module
    apply_params(params)

    observations = []
    actions = []
    lap_times = []
    total_steps = 0
    laps_completed = 0

    if verbose:
        print("=" * 60)
        print("COLLECTING DEMONSTRATIONS FROM MODULAR CONTROLLER")
        print("=" * 60)
        print(f"Target laps: {n_laps}")
        print(f"Port: {port}")
        print(f"Parameters: TARGET_SPEED={params.get('TARGET_SPEED')}, "
              f"STEER_GAIN={params.get('STEER_GAIN')}, "
              f"BRAKE_INTENSITY={params.get('BRAKE_INTENSITY')}")
        print("=" * 60)

    while laps_completed < n_laps:
        if verbose:
            print(f"\nStarting lap attempt {laps_completed + 1}/{n_laps}...")

        # Initialize client and trackers
        try:
            C = Client(p=port)
            C.maxSteps = max_steps_per_lap
        except Exception as e:
            print(f"Failed to connect: {e}")
            print("Make sure TORCS is running with scr_server driver")
            break

        tracker = LapTimeTracker()

        lap_observations = []
        lap_actions = []
        lap_complete = False
        dnf = False

        try:
            for step in range(max_steps_per_lap):
                # Get server state
                if not C.get_servers_input():
                    print("  Lost connection to server")
                    dnf = True
                    break

                if C.so is None:
                    break

                if C.S.d is None:
                    continue

                S = C.S.d
                R = C.R.d

                # Check for off-track or facing backward
                track_pos = S.get('trackPos', 0)
                angle = S.get('angle', 0)

                if abs(track_pos) > 1.3:
                    if verbose:
                        print(f"  DNF: Off track at step {step} (trackPos={track_pos:.2f})")
                    dnf = True
                    break

                if np.cos(angle) < 0 and S.get('speedX', 0) > 5:
                    if verbose:
                        print(f"  DNF: Facing backward at step {step}")
                    dnf = True
                    break

                # Record observation BEFORE action
                obs = build_observation(S)

                # Run modular controller to get action
                drive_modular(C)

                # Record the action that was set
                action = np.array([
                    R.get('steer', 0),
                    R.get('accel', 0),
                    R.get('brake', 0)
                ], dtype=np.float32)

                lap_observations.append(obs)
                lap_actions.append(action)

                # Track lap completion
                tracker.update(S)

                if tracker.lap_just_completed:
                    lap_complete = True
                    lap_time = tracker.last_lap_time
                    if verbose:
                        print(f"  Lap completed: {lap_time:.3f}s ({len(lap_observations)} steps)")
                    break

                # Send response
                C.respond_to_server()

        except Exception as e:
            print(f"  Error during collection: {e}")
            dnf = True

        finally:
            # Clean up
            try:
                if lap_complete or dnf:
                    C.restart_race()
                else:
                    C.shutdown()
            except:
                pass

        # Only keep data from successful laps
        if lap_complete and not dnf:
            observations.extend(lap_observations)
            actions.extend(lap_actions)
            lap_times.append(tracker.last_lap_time)
            total_steps += len(lap_observations)
            laps_completed += 1

            if verbose:
                print(f"  Collected {len(lap_observations)} steps (total: {total_steps})")
        elif dnf:
            if verbose:
                print(f"  Discarding {len(lap_observations)} steps from DNF lap")

        # Small delay between laps
        import time
        time.sleep(2.0)

    if verbose:
        print("\n" + "=" * 60)
        print("COLLECTION COMPLETE")
        print("=" * 60)
        print(f"Laps completed: {laps_completed}")
        print(f"Total steps: {total_steps}")
        if lap_times:
            print(f"Lap times: {[f'{t:.3f}s' for t in lap_times]}")
            print(f"Average: {np.mean(lap_times):.3f}s")
        print("=" * 60)

    return {
        'observations': np.array(observations, dtype=np.float32),
        'actions': np.array(actions, dtype=np.float32),
        'lap_times': lap_times,
        'total_steps': total_steps
    }


def save_demonstrations(data, output_path):
    """Save demonstration data to .npz file."""
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    np.savez(
        output_path,
        observations=data['observations'],
        actions=data['actions'],
        lap_times=np.array(data['lap_times']),
        metadata=np.array([
            f"collected={datetime.now().isoformat()}",
            f"total_steps={data['total_steps']}",
            f"n_laps={len(data['lap_times'])}"
        ])
    )

    print(f"\nDemonstrations saved to: {output_path}")
    print(f"  Observations shape: {data['observations'].shape}")
    print(f"  Actions shape: {data['actions'].shape}")


def main():
    parser = argparse.ArgumentParser(
        description='Collect demonstrations from GA/State Machine controller',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Collect 5 laps of demonstrations
    python collect_ga_demos.py --laps 5

    # Collect 10 laps and save to custom location
    python collect_ga_demos.py --laps 10 --output demos/my_demos.npz

    # Then pretrain the RL agent:
    python pretrain_bc.py --demos demos/ga_demos.npz --output models/pretrained

    # Finally, continue with RL training:
    python train_ppo.py --resume models/pretrained/best_bc_model
        """
    )

    parser.add_argument('--laps', type=int, default=5,
                        help='Number of laps to collect (default: 5)')
    parser.add_argument('--port', type=int, default=3001,
                        help='TORCS server port (default: 3001)')
    parser.add_argument('--output', type=str, default='demos/ga_demos.npz',
                        help='Output file path (default: demos/ga_demos.npz)')
    parser.add_argument('--max-steps', type=int, default=15000,
                        help='Max steps per lap attempt (default: 15000)')
    parser.add_argument('--quiet', action='store_true',
                        help='Reduce output verbosity')

    args = parser.parse_args()

    # Collect demonstrations
    data = collect_demonstrations(
        n_laps=args.laps,
        port=args.port,
        max_steps_per_lap=args.max_steps,
        verbose=not args.quiet
    )

    if data['total_steps'] == 0:
        print("\nNo demonstrations collected!")
        print("Make sure TORCS is running and the controller can complete laps.")
        return

    # Save to file
    save_demonstrations(data, args.output)

    # Print next steps
    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("1. Pretrain the RL policy with behavioral cloning:")
    print(f"   python pretrain_bc.py --demos {args.output} --output models/pretrained")
    print()
    print("2. Continue training with reinforcement learning:")
    print("   python train_ppo.py --resume models/pretrained/best_bc_model")
    print("=" * 60)


if __name__ == "__main__":
    main()
