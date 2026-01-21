"""
Stage 1: Baseline Runner
Provides run_episode(params) -> lap_time functionality for parameter optimization.
"""

import sys
import os
import csv
import math
from datetime import datetime

# Import from torcs_jm_par
from torcs_jm_par import (
    Client, LapTimeTracker,
    calculate_steering, calculate_throttle, apply_brakes,
    shift_gears, traction_control
)
import torcs_jm_par as torcs_module

# Base gear speeds (scaled by GEAR_SHIFT_SCALE)
BASE_GEAR_SPEEDS = [0, 50, 80, 120, 150, 200]

# DNF detection threshold: |trackPos| > this value = too far off track
# trackPos: 0 = center, +/-1 = track edge, >1 = off track
# Set > 1.0 to allow some off-track driving (e.g., 1.3 = 30% past edge)
DEFAULT_OFF_TRACK_THRESHOLD = 1.3

# DNF detection for stalled start: car must travel at least this distance
# within the first N steps, otherwise it's considered stalled/stuck
# TORCS runs at ~50Hz, so 250 steps ≈ 5 seconds
DEFAULT_START_MIN_DISTANCE = 25  # meters
DEFAULT_START_CHECK_STEPS = 250  # steps (~5 seconds at 50Hz)


def gear_shift_scale_to_speeds(scale):
    """Convert GEAR_SHIFT_SCALE to actual gear speeds list."""
    return [BASE_GEAR_SPEEDS[0]] + [BASE_GEAR_SPEEDS[i] * scale for i in range(1, 6)]


def speeds_to_gear_shift_scale(speeds):
    """Estimate GEAR_SHIFT_SCALE from gear speeds (uses gear 2 as reference)."""
    if BASE_GEAR_SPEEDS[2] > 0 and len(speeds) > 2:
        return speeds[2] / BASE_GEAR_SPEEDS[2]
    return 1.0


def run_episode(params, port=3001, max_steps=100000, target_laps=1, verbose=True,
                 off_track_threshold=None, restart=True,
                 start_min_distance=None, start_check_steps=None):
    """
    Run a single episode with given parameters and return lap time.

    Args:
        params: dict with keys:
            - TARGET_SPEED: float (50-300) - target speed in km/h
            - STEER_GAIN: float (1-100) - steering sensitivity
            - CENTERING_GAIN: float (0-2) - track centering strength
            - BRAKE_THRESHOLD: float (0.02-1.5) - angle threshold for braking
            - GEAR_SPEEDS: list of 6 floats - speed thresholds for each gear
            - ENABLE_TRACTION_CONTROL: bool
            - THROTTLE_INCREASE: float (0.2-0.8) - acceleration rate
            - THROTTLE_DECREASE: float (0.1-0.5) - deceleration rate
            - SPEED_STEER_FACTOR: float (1.0-5.0) - steering effect on target speed
            - BRAKE_INTENSITY: float (0.1-0.8) - braking force
            - TC_THRESHOLD: float (1-10) - wheel spin differential for TC
            - TC_REDUCTION: float (0.05-0.3) - throttle reduction when TC activates
        port: TORCS server port (default 3001)
        max_steps: Maximum simulation steps (default 100000)
        target_laps: Number of laps to complete before stopping (default 1)
        verbose: Print progress info (default True)
        off_track_threshold: Max |trackPos| before DNF (default 1.3, None to disable)
                            trackPos: 0=center, +/-1=edge, >1=off track
        restart: If True, request race restart for next episode. If False, just shutdown (default True)
        start_min_distance: Minimum distance (m) car must travel within start_check_steps,
                           otherwise DNF for stalled start (default 50m, None to disable)
        start_check_steps: Number of steps to check start distance (default 250 = ~5 sec)

    Returns:
        dict with:
            - lap_time: Best lap time (None if no laps completed or DNF)
            - lap_times: List of all lap times
            - lap_count: Number of laps completed
            - avg_lap_time: Average lap time (None if no laps)
            - total_steps: Steps used
            - completed: True if target_laps reached
            - dnf: True if episode ended due to DNF condition
            - dnf_reason: String describing DNF cause (None if not DNF)
    """
    # Set default off-track threshold
    if off_track_threshold is None:
        off_track_threshold = DEFAULT_OFF_TRACK_THRESHOLD

    # Set default start distance check parameters
    if start_min_distance is None:
        start_min_distance = DEFAULT_START_MIN_DISTANCE
    if start_check_steps is None:
        start_check_steps = DEFAULT_START_CHECK_STEPS

    # Apply parameters to module - core driving parameters
    torcs_module.TARGET_SPEED = params.get('TARGET_SPEED', 180)
    torcs_module.STEER_GAIN = params.get('STEER_GAIN', 50)
    torcs_module.CENTERING_GAIN = params.get('CENTERING_GAIN', 0.60)
    torcs_module.BRAKE_THRESHOLD = params.get('BRAKE_THRESHOLD', 0.5)

    # Gear speeds - use direct list
    torcs_module.GEAR_SPEEDS = params.get('GEAR_SPEEDS', BASE_GEAR_SPEEDS)

    torcs_module.ENABLE_TRACTION_CONTROL = params.get('ENABLE_TRACTION_CONTROL', True)

    # Throttle control parameters
    torcs_module.THROTTLE_INCREASE = params.get('THROTTLE_INCREASE', 0.4)
    torcs_module.THROTTLE_DECREASE = params.get('THROTTLE_DECREASE', 0.2)
    torcs_module.SPEED_STEER_FACTOR = params.get('SPEED_STEER_FACTOR', 2.5)

    # Brake control parameters
    torcs_module.BRAKE_INTENSITY = params.get('BRAKE_INTENSITY', 0.3)

    # Traction control parameters
    torcs_module.TC_THRESHOLD = params.get('TC_THRESHOLD', 2)
    torcs_module.TC_REDUCTION = params.get('TC_REDUCTION', 0.1)

    if verbose:
        print(f"Running episode with params:")
        print(f"  TARGET_SPEED={torcs_module.TARGET_SPEED}")
        print(f"  STEER_GAIN={torcs_module.STEER_GAIN}")
        print(f"  CENTERING_GAIN={torcs_module.CENTERING_GAIN}")
        print(f"  BRAKE_THRESHOLD={torcs_module.BRAKE_THRESHOLD}")
        print(f"  GEAR_SPEEDS={torcs_module.GEAR_SPEEDS}")
        print(f"  THROTTLE_INCREASE={torcs_module.THROTTLE_INCREASE}")
        print(f"  THROTTLE_DECREASE={torcs_module.THROTTLE_DECREASE}")
        print(f"  SPEED_STEER_FACTOR={torcs_module.SPEED_STEER_FACTOR}")
        print(f"  BRAKE_INTENSITY={torcs_module.BRAKE_INTENSITY}")
        print(f"  TC_THRESHOLD={torcs_module.TC_THRESHOLD}")
        print(f"  TC_REDUCTION={torcs_module.TC_REDUCTION}")
        print(f"  ENABLE_TRACTION_CONTROL={torcs_module.ENABLE_TRACTION_CONTROL}")

    # Initialize client and tracker
    C = Client(p=port)
    C.maxSteps = max_steps
    tracker = LapTimeTracker()

    steps_used = 0
    dnf = False
    dnf_reason = None
    start_check_done = False  # Track if we've checked for stalled start

    try:
        for step in range(max_steps, 0, -1):
            C.get_servers_input()
            if C.so is None:  # Connection closed
                break

            if C.S.d:
                # Drive using module functions (which use updated params)
                S, R = C.S.d, C.R.d

                # DNF check: car too far off track
                track_pos = S.get('trackPos', 0)
                if off_track_threshold and abs(track_pos) > off_track_threshold:
                    dnf = True
                    dnf_reason = f"off-track (trackPos={track_pos:.2f}, threshold={off_track_threshold})"
                    if verbose:
                        print(f"  DNF: {dnf_reason}")
                    steps_used = max_steps - step
                    break

                # DNF check: car facing backward (angle > 90 degrees from track direction)
                angle = S.get('angle', 0)
                if math.cos(angle) < 0:
                    dnf = True
                    dnf_reason = f"facing backward (angle={math.degrees(angle):.1f}°)"
                    if verbose:
                        print(f"  DNF: {dnf_reason}")
                    steps_used = max_steps - step
                    break

                # DNF check: car stalled at start (hasn't moved enough distance)
                current_step = max_steps - step
                if start_min_distance and not start_check_done and current_step >= start_check_steps:
                    dist_raced = S.get('distRaced', 0)
                    if dist_raced < start_min_distance:
                        dnf = True
                        dnf_reason = f"stalled at start (distRaced={dist_raced:.1f}m after {start_check_steps} steps, need {start_min_distance}m)"
                        if verbose:
                            print(f"  DNF: {dnf_reason}")
                        steps_used = current_step
                        break
                    start_check_done = True  # Passed the check, don't check again

                R['steer'] = calculate_steering(S)
                R['accel'] = calculate_throttle(S, R)
                R['brake'] = apply_brakes(S)
                R['accel'] = traction_control(S, R['accel'])
                R['gear'] = shift_gears(S)

                # Track lap times
                tracker.update(C.S.d)

                if tracker.lap_just_completed and verbose:
                    print(f"  Lap {tracker.lap_count}: {tracker.last_lap_time:.3f}s")

                # Stop if we've completed target laps
                if tracker.lap_count >= target_laps:
                    steps_used = max_steps - step
                    break

            C.respond_to_server()
            steps_used = max_steps - step

    finally:
        if restart:
            C.restart_race()  # Request TORCS to restart race for next episode
        else:
            C.shutdown()  # Final episode, just close connection

    # Build result
    stats = tracker.get_stats()
    result = {
        'lap_time': None if dnf else stats['best_lap'],  # DNF = no valid lap time
        'lap_times': stats['all_laps'],
        'lap_count': stats['lap_count'],
        'avg_lap_time': stats['avg_lap'],
        'total_steps': steps_used,
        'completed': stats['lap_count'] >= target_laps and not dnf,
        'dnf': dnf,
        'dnf_reason': dnf_reason
    }

    if verbose:
        if dnf:
            print(f"Episode DNF: {dnf_reason}")
        else:
            print(f"Episode complete: {stats['lap_count']} laps, best={stats['best_lap']}")

    return result


def get_default_params():
    """Return default parameter dictionary with all 15 tunable parameters."""
    # Gear speeds derived from previous best GEAR_SHIFT_SCALE of ~0.61
    gear_scale = 0.6108308839970421
    return {
        # Core driving parameters (4)
        'TARGET_SPEED': 77.30955047334417,
        'STEER_GAIN': 61.37239899636491,
        'CENTERING_GAIN': 1.7178362750637122,
        'BRAKE_THRESHOLD': 0.6794351390345678,
        # Individual gear speeds (5) - derived from gear_scale * BASE_GEAR_SPEEDS
        'GEAR_SPEEDS': [0, 50 * gear_scale, 80 * gear_scale, 120 * gear_scale,
                        150 * gear_scale, 200 * gear_scale],
        # Throttle control parameters (3)
        'THROTTLE_INCREASE': 0.4,
        'THROTTLE_DECREASE': 0.2,
        'SPEED_STEER_FACTOR': 2.5,
        # Brake control parameters (1)
        'BRAKE_INTENSITY': 0.3,
        # Traction control parameters (2)
        'TC_THRESHOLD': 2.0,
        'TC_REDUCTION': 0.1,
        # Fixed parameter
        'ENABLE_TRACTION_CONTROL': True
    }


# ================= CSV LOGGING =================
class ResultLogger:
    """
    Logs episode results to CSV file.

    CSV columns:
        timestamp, target_speed, steer_gain, centering_gain, brake_threshold,
        gear_2_speed, gear_3_speed, gear_4_speed, gear_5_speed, gear_6_speed,
        throttle_increase, throttle_decrease, speed_steer_factor,
        brake_intensity, tc_threshold, tc_reduction, traction_control,
        lap_time, avg_lap_time, lap_count, completed
    """

    def __init__(self, filepath='results/run_log.csv'):
        self.filepath = filepath
        self._ensure_dir()
        self._write_header()

    def _ensure_dir(self):
        """Create results directory if it doesn't exist."""
        dir_path = os.path.dirname(self.filepath)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path)

    def _write_header(self):
        """Write CSV header if file doesn't exist."""
        if not os.path.exists(self.filepath):
            with open(self.filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp',
                    'target_speed', 'steer_gain', 'centering_gain', 'brake_threshold',
                    'gear_2_speed', 'gear_3_speed', 'gear_4_speed', 'gear_5_speed', 'gear_6_speed',
                    'throttle_increase', 'throttle_decrease', 'speed_steer_factor',
                    'brake_intensity', 'tc_threshold', 'tc_reduction', 'traction_control',
                    'lap_time', 'avg_lap_time', 'lap_count', 'completed'
                ])

    def log(self, params, result):
        """
        Log a single episode result.

        Args:
            params: Parameter dictionary
            result: Result dictionary from run_episode()
        """
        gear_speeds = params.get('GEAR_SPEEDS', BASE_GEAR_SPEEDS)

        row = [
            datetime.now().isoformat(),
            params.get('TARGET_SPEED', 180),
            params.get('STEER_GAIN', 50),
            params.get('CENTERING_GAIN', 0.60),
            params.get('BRAKE_THRESHOLD', 0.5),
            gear_speeds[1] if len(gear_speeds) > 1 else 50,
            gear_speeds[2] if len(gear_speeds) > 2 else 80,
            gear_speeds[3] if len(gear_speeds) > 3 else 120,
            gear_speeds[4] if len(gear_speeds) > 4 else 150,
            gear_speeds[5] if len(gear_speeds) > 5 else 200,
            params.get('THROTTLE_INCREASE', 0.4),
            params.get('THROTTLE_DECREASE', 0.2),
            params.get('SPEED_STEER_FACTOR', 2.5),
            params.get('BRAKE_INTENSITY', 0.3),
            params.get('TC_THRESHOLD', 2.0),
            params.get('TC_REDUCTION', 0.1),
            1 if params.get('ENABLE_TRACTION_CONTROL', True) else 0,
            result.get('lap_time', ''),
            result.get('avg_lap_time', ''),
            result.get('lap_count', 0),
            1 if result.get('completed', False) else 0
        ]

        with open(self.filepath, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)


def run_episode_logged(params, logger=None, **kwargs):
    """
    Run episode and log results to CSV.

    Args:
        params: Parameter dictionary
        logger: ResultLogger instance (creates default if None)
        **kwargs: Additional args passed to run_episode()

    Returns:
        Result dictionary from run_episode()
    """
    if logger is None:
        logger = ResultLogger()

    result = run_episode(params, **kwargs)
    logger.log(params, result)

    return result


# ================= PARAMETER SWEEP =================
def run_parameter_sweep(sweep_config=None, target_laps=1, verbose=True):
    """
    Run a manual parameter sweep to validate the system.

    Args:
        sweep_config: dict mapping param names to lists of values to try.
                     If None, uses a default sweep configuration.
        target_laps: Number of laps per episode
        verbose: Print progress

    Returns:
        List of (params, result) tuples sorted by lap_time (best first)
    """
    if sweep_config is None:
        # Default sweep: test a few values for key parameters
        sweep_config = {
            'TARGET_SPEED': [150, 180, 210],
            'STEER_GAIN': [40, 50, 60],
            'GEAR_SHIFT_SCALE': [0.8, 1.0, 1.2],
        }

    # Generate all parameter combinations
    from itertools import product

    base_params = get_default_params()
    param_names = list(sweep_config.keys())
    param_values = [sweep_config[k] for k in param_names]

    combinations = list(product(*param_values))
    total_runs = len(combinations)

    if verbose:
        print("=" * 60)
        print("PARAMETER SWEEP")
        print("=" * 60)
        print(f"Sweeping: {param_names}")
        print(f"Total combinations: {total_runs}")
        print("=" * 60)

    logger = ResultLogger('results/sweep_log.csv')
    results = []

    for i, combo in enumerate(combinations):
        # Build params for this run
        params = base_params.copy()
        for name, value in zip(param_names, combo):
            params[name] = value

        if verbose:
            print(f"\n[{i+1}/{total_runs}] Testing: ", end="")
            print(", ".join(f"{n}={v}" for n, v in zip(param_names, combo)))

        result = run_episode_logged(params, logger=logger, target_laps=target_laps, verbose=False)
        results.append((params.copy(), result))

        if verbose:
            if result['lap_time']:
                print(f"  -> Lap time: {result['lap_time']:.3f}s")
            else:
                print(f"  -> No lap completed")

    # Sort by lap time (best first), putting None values at end
    results.sort(key=lambda x: x[1]['lap_time'] if x[1]['lap_time'] else float('inf'))

    if verbose:
        print("\n" + "=" * 60)
        print("SWEEP RESULTS (sorted by lap time)")
        print("=" * 60)
        for i, (params, result) in enumerate(results[:10]):  # Show top 10
            lap_time = result['lap_time']
            if lap_time:
                print(f"{i+1}. {lap_time:.3f}s - ", end="")
                print(", ".join(f"{n}={params[n]}" for n in param_names))
            else:
                print(f"{i+1}. DNF - ", end="")
                print(", ".join(f"{n}={params[n]}" for n in param_names))

        print(f"\nResults saved to: {logger.filepath}")

    return results


def run_quick_sweep(verbose=True):
    """
    Run a quick 3x3 sweep on TARGET_SPEED and STEER_GAIN only.
    Good for quick validation.
    """
    sweep_config = {
        'TARGET_SPEED': [160, 180, 200],
        'STEER_GAIN': [45, 50, 55],
    }
    return run_parameter_sweep(sweep_config, target_laps=1, verbose=verbose)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='TORCS Baseline Runner')
    parser.add_argument('--sweep', action='store_true', help='Run full parameter sweep (27 runs)')
    parser.add_argument('--quick-sweep', action='store_true', help='Run quick parameter sweep (9 runs)')
    parser.add_argument('--laps', type=int, default=1, help='Target laps per episode')
    args = parser.parse_args()

    # Clear sys.argv so torcs_jm_par's Client doesn't try to parse our arguments
    sys.argv = [sys.argv[0]]

    if args.sweep:
        # Full sweep: 3x3x3 = 27 combinations
        run_parameter_sweep(target_laps=args.laps)
    elif args.quick_sweep:
        # Quick sweep: 3x3 = 9 combinations
        run_quick_sweep()
    else:
        # Single test run with default parameters
        print("=" * 50)
        print("Baseline Runner - Single Test")
        print("=" * 50)
        print("Use --sweep or --quick-sweep for parameter sweep")
        print("=" * 50)

        params = get_default_params()
        logger = ResultLogger()

        result = run_episode_logged(params, logger=logger, target_laps=args.laps, verbose=True)

        print("\n" + "=" * 50)
        print("Results:")
        print(f"  Lap time: {result['lap_time']}")
        print(f"  Completed: {result['completed']}")
        print(f"  Steps used: {result['total_steps']}")
        print(f"  Results logged to: {logger.filepath}")
