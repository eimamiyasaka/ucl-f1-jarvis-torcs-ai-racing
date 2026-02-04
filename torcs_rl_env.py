"""
Gym-compatible TORCS environment for reinforcement learning.
Optimized for single lap time from standing start.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
from torcs_jm_par import Client, LapTimeTracker


class TorcsRLEnv(gym.Env):
    """
    TORCS Racing Environment for RL training.

    Observation Space:
        - speedX, speedY, speedZ (3)
        - angle (1)
        - trackPos (1)
        - 19 track edge sensors (19)
        - rpm (1)
        - gear (1)
        - wheelSpinVel (4)
        Total: 30 dimensions

    Action Space:
        - steering: continuous [-1, 1]
        - acceleration: continuous [0, 1]
        - brake: continuous [0, 1]
        Total: 3 dimensions
    """

    metadata = {'render.modes': []}

    def __init__(self, port=3001, max_steps=15000, target_laps=1,
                 off_track_threshold=1.2, start_min_distance=10,
                 start_check_steps=500, reward_type='progress',
                 launch_assist=True):
        """
        Initialize TORCS RL environment.

        Args:
            port: TORCS server port
            max_steps: Maximum steps per episode
            target_laps: Number of laps to complete (1 for single lap optimization)
            off_track_threshold: Max |trackPos| before DNF
            start_min_distance: Minimum distance to travel in start_check_steps
            start_check_steps: Steps to check for stalled start
            reward_type: 'progress' or 'time' - reward shaping strategy
            launch_assist: Enable launch assist (restricts steering at low speeds)
        """
        super(TorcsRLEnv, self).__init__()

        self.port = port
        self.max_steps = max_steps
        self.target_laps = target_laps
        self.off_track_threshold = off_track_threshold
        self.start_min_distance = start_min_distance
        self.start_check_steps = start_check_steps
        self.reward_type = reward_type
        self.launch_assist_enabled = launch_assist

        # Define action space: [steering, accel, brake]
        # Accel is [-1, 1] where 0 = cruise throttle (~70 km/h), remapped in step()
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32
        )

        # Define observation space (30 dimensions)
        # Speed: -50 to 350 km/h, angle: -pi to pi, trackPos: -2 to 2
        # track sensors: 0 to 200, rpm: 0 to 10000, gear: 0 to 6, wheelSpin: 0 to 100
        obs_high = np.array([
            350.0, 50.0, 50.0,  # speedX, speedY, speedZ
            np.pi,  # angle
            2.0,  # trackPos
            *([200.0] * 19),  # track sensors
            10000.0,  # rpm
            6.0,  # gear
            100.0, 100.0, 100.0, 100.0  # wheelSpinVel
        ], dtype=np.float32)

        obs_low = np.array([
            -50.0, -50.0, -50.0,  # speedX, speedY, speedZ
            -np.pi,  # angle
            -2.0,  # trackPos
            *([0.0] * 19),  # track sensors
            0.0,  # rpm
            -1.0,  # gear (can be -1 for reverse)
            0.0, 0.0, 0.0, 0.0  # wheelSpinVel
        ], dtype=np.float32)

        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # Episode tracking
        self.client = None
        self.lap_tracker = None
        self.step_count = 0
        self.episode_count = 0
        self.prev_dist_raced = 0.0
        self.prev_speed = 0.0
        self.prev_steer = 0.0  # Track previous steering for smoothness penalty
        self.start_check_done = False
        self.total_reward = 0.0
        self.best_lap_time = None
        self.connection_healthy = False  # Track connection health
        self.consecutive_failures = 0  # Track consecutive communication failures
        self.max_consecutive_failures = 5  # Max failures before episode termination
        self.launch_assist_complete = False  # Track if launch assist phase is done
        self.stalled_steps = 0  # Track consecutive steps with near-zero speed

    def reset(self, seed=None, options=None):
        """Reset the environment and return initial observation and info."""

        # Handle seed for gymnasium compatibility
        super().reset(seed=seed)

        # Properly shutdown previous client if exists
        if self.client is not None:
            self._cleanup_client()

        # Create new client connection with retry logic
        max_connection_attempts = 3
        for attempt in range(max_connection_attempts):
            try:
                # The Client class handles waiting for server in setup_connection()
                self.client = Client(p=self.port)
                self.client.maxSteps = self.max_steps
                self.connection_healthy = True
                self.consecutive_failures = 0
                break
            except Exception as e:
                print(f"[Episode {self.episode_count + 1}] Connection attempt {attempt + 1}/{max_connection_attempts} failed: {e}")
                if attempt == max_connection_attempts - 1:
                    print(f"Failed to connect after {max_connection_attempts} attempts")
                    self.connection_healthy = False
                    # Return zero observation with connection failure flag
                    # Mark as terminated so training loop handles this like a DNF
                    obs = np.zeros(self.observation_space.shape, dtype=np.float32)
                    return obs, {
                        'connection_failed': True,
                        'dnf': True,
                        'dnf_reason': 'connection_failed',
                        'terminal_reward': -100.0  # Signal penalty to training loop
                    }

        self.lap_tracker = LapTimeTracker()

        # Get initial state with timeout handling
        if not self.client.get_servers_input(max_retries=10):
            print(f"[Episode {self.episode_count + 1}] Failed to get initial server state")
            self.connection_healthy = False
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            return obs, {
                'connection_failed': True,
                'dnf': True,
                'dnf_reason': 'connection_failed',
                'terminal_reward': -100.0  # Signal penalty to training loop
            }

        # Reset episode tracking
        self.step_count = 0
        self.episode_count += 1
        self.prev_dist_raced = 0.0
        self.prev_speed = 0.0
        self.prev_steer = 0.0  # Reset steering history
        self.start_check_done = False
        self.total_reward = 0.0
        self.launch_assist_complete = False  # Reset launch assist for new episode
        self.stalled_steps = 0  # Reset stalled counter

        # Debug: show initial state
        if self.client.S.d:
            print(f"[Episode {self.episode_count}] Reset complete - "
                  f"speed={self.client.S.d.get('speedX', 0):.1f}, "
                  f"trackPos={self.client.S.d.get('trackPos', 0):.2f}, "
                  f"distRaced={self.client.S.d.get('distRaced', 0):.1f}")

        # Return initial observation and info dict (gymnasium API)
        obs = self._get_observation()
        info = {}
        return obs, info

    def _cleanup_client(self):
        """Properly cleanup client connection and request race restart."""
        import time

        if self.client is None:
            return

        try:
            # Try to request race restart first
            self.client.restart_race()
        except Exception as e:
            print(f"Warning: restart_race failed: {e}")

        # Ensure socket is closed even if restart_race failed
        try:
            if self.client.so is not None:
                self.client.so.close()
                self.client.so = None
        except Exception as e:
            print(f"Warning: socket close failed: {e}")

        self.client = None

        # Give TORCS time to process the restart before reconnecting
        # This is crucial - without this delay, rapid episode resets can
        # overwhelm TORCS's restart mechanism
        time.sleep(5.0)

    def step(self, action):
        """
        Execute one step in the environment.

        Args:
            action: [steering, accel, brake]

        Returns:
            observation, reward, terminated, truncated, info (gymnasium API)
        """
        self.step_count += 1

        # Check if connection is healthy before proceeding
        if not self.connection_healthy or self.client is None or self.client.so is None:
            obs = self._get_observation()
            return obs, -100.0, True, False, {'dnf': True, 'dnf_reason': 'connection_lost'}

        # Apply action
        steer = np.clip(action[0], -1.0, 1.0)
        brake = np.clip(action[2], 0.0, 1.0)

        # Remap acceleration: agent's 0 = cruise throttle (~70 km/h)
        # -1 → 0 (no throttle), 0 → 0.4 (cruise), +1 → 1 (full throttle)
        raw_accel = np.clip(action[1], -1.0, 1.0)
        if raw_accel >= 0:
            accel = 0.4 + raw_accel * 0.6  # 0→0.4, 1→1.0
        else:
            accel = 0.4 + raw_accel * 0.4  # -1→0, 0→0.4

        # Get current state
        current_speed = self.client.S.d.get('speedX', 0) if self.client.S.d else 0

        # Steering damping at high speeds to prevent spins
        # Reduce max steering as speed increases, but allow enough for corners
        if current_speed > 50:
            # At 50 km/h: max steer = 1.0, at 150 km/h: max steer = 0.4, at 250 km/h: max steer = 0.2
            max_steer = max(0.2, 1.0 - (current_speed - 50) / 200)
            steer = np.clip(steer, -max_steer, max_steer)

        # LAUNCH ASSIST DISABLED - let agent learn from scratch
        # The reward system now handles stall prevention via low speed penalties
        # if self.launch_assist_enabled and not self.launch_assist_complete:
        #     if current_speed >= 100:
        #         # Launch phase complete - agent now has full control
        #         self.launch_assist_complete = True
        #     elif current_speed >= -5:
        #         # Still in launch phase - apply assists
        #         # Phase 1: Steering restriction at very low speeds (0-30 km/h)
        #         if current_speed < 30:
        #             speed_factor = max(0, current_speed) / 30.0
        #             launch_max_steer = 0.1 + speed_factor * 0.4  # 0.1 at 0, 0.5 at 30 km/h
        #             steer = np.clip(steer, -launch_max_steer, launch_max_steer)
        #
        #         # Phase 2: Minimum acceleration floor - ensures car keeps moving
        #         steer_penalty = 1.0 - abs(steer) * 0.4
        #         min_accel = np.clip(steer_penalty, 0.5, 1.0) * 0.8  # Min 40-80% throttle
        #         accel = max(accel, min_accel)
        #
        #         # No braking during launch
        #         brake = 0.0

        # Resolve accelerator + brake conflict: braking takes priority
        # This prevents the agent from wasting actions by pressing both
        if brake > 0.1:
            accel = accel * (1.0 - brake)  # Reduce accel proportionally to brake

        # MINIMUM ACCELERATION AT LOW SPEEDS
        # Without this, random exploration produces near-zero throttle and car never moves
        # This gives the agent something to learn from (movement = reward signal)
        # if current_speed < 50 and current_speed >= 0:
        #     min_accel = 0.5  # Minimum 50% throttle when slow
        #     accel = max(accel, min_accel)
        #     brake = 0.0  # No braking when trying to get moving

        if current_speed < 100 and current_speed >= 0:
            min_accel = 0.5
            accel = max(accel, min_accel)
            if current_speed < 50:
                brake = 0.0

        # If car is going backwards, don't accelerate (let it stop naturally)
        if current_speed < -5:
            accel = 0.0
            brake = 0.5  # Apply some brake to stop

        self.client.R.d['steer'] = steer
        self.client.R.d['accel'] = accel
        self.client.R.d['brake'] = brake

        # Simple automatic gear shifting
        self.client.R.d['gear'] = self._auto_gear()

        # Debug: print actions every 50 steps
        if self.step_count % 50 == 1:
            print(f"  [Step {self.step_count}] Actions: accel={accel:.2f}, brake={brake:.2f}, "
                  f"steer={steer:.2f}, gear={self.client.R.d['gear']}, speed={current_speed:.1f}")

        # Send action and get response with failure tracking
        send_success = self.client.respond_to_server()
        if not send_success:
            self.consecutive_failures += 1
            if self.consecutive_failures >= self.max_consecutive_failures:
                self.connection_healthy = False
                obs = self._get_observation()
                return obs, -100.0, True, False, {'dnf': True, 'dnf_reason': 'connection_lost'}

        recv_success = self.client.get_servers_input(max_retries=10)
        if not recv_success:
            self.consecutive_failures += 1
            if self.consecutive_failures >= self.max_consecutive_failures:
                self.connection_healthy = False
                obs = self._get_observation()
                return obs, -100.0, True, False, {'dnf': True, 'dnf_reason': 'connection_timeout'}
        else:
            self.consecutive_failures = 0  # Reset on success

        # # Debug: print state every 50 steps
        # if self.step_count % 50 == 1 and self.client.S.d:
        #     S = self.client.S.d
        #     print(f"  [Step {self.step_count}] State: speed={S.get('speedX', 0):.1f}, "
        #           f"dist={S.get('distRaced', 0):.1f}, gear={S.get('gear', 0)}")

        # Check if connection closed
        if self.client.so is None or self.client.S.d is None:
            self.connection_healthy = False
            obs = self._get_observation()
            return obs, -100.0, True, False, {'dnf': True, 'dnf_reason': 'connection_lost'}

        S = self.client.S.d

        # Update lap tracker
        self.lap_tracker.update(S)

        # Get current state
        obs = self._get_observation()

        # Check termination conditions
        done, dnf_reason = self._check_done(S)

        # Gymnasium uses terminated and truncated instead of done
        # terminated = episode ended due to environment rules (lap complete, crash, etc.)
        # truncated = episode ended due to time limit
        truncated = (dnf_reason == 'max_steps')
        terminated = done and not truncated

        # Calculate reward
        reward = self._calculate_reward(S, done, dnf_reason)
        self.total_reward += reward

        # Build info dict
        info = {
            'step': self.step_count,
            'speed': S.get('speedX', 0),
            'dist_raced': S.get('distRaced', 0),
            'trackPos': S.get('trackPos', 0),
            'lap_count': self.lap_tracker.lap_count,
            'total_reward': self.total_reward,
            'dnf': dnf_reason is not None,
            'dnf_reason': dnf_reason
        }

        # Debug: log episode termination
        if done:
            reason = dnf_reason if dnf_reason else "lap_complete"
            print(f"[Episode {self.episode_count}] Ended at step {self.step_count}: {reason} "
                  f"(dist={S.get('distRaced', 0):.1f}m, speed={S.get('speedX', 0):.1f}km/h, reward={self.total_reward:.1f})")

        # Add lap time info if lap completed
        if self.lap_tracker.lap_just_completed:
            info['lap_time'] = self.lap_tracker.last_lap_time
            info['best_lap_time'] = self.lap_tracker.best_lap_time

            # Update best lap time
            if self.best_lap_time is None or self.lap_tracker.last_lap_time < self.best_lap_time:
                self.best_lap_time = self.lap_tracker.last_lap_time

        return obs, reward, terminated, truncated, info

    def _get_observation(self):
        """Extract observation vector from server state."""
        if self.client.S.d is None:
            # Return zeros if no state available
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        S = self.client.S.d

        obs = np.array([
            S.get('speedX', 0),
            S.get('speedY', 0),
            S.get('speedZ', 0),
            S.get('angle', 0),
            S.get('trackPos', 0),
            *S.get('track', [100.0] * 19)[:19],  # Ensure 19 elements
            S.get('rpm', 0),
            S.get('gear', 1),
            *S.get('wheelSpinVel', [0, 0, 0, 0])[:4]  # Ensure 4 elements
        ], dtype=np.float32)

        # Ensure correct shape
        assert obs.shape == (30,), f"Observation shape mismatch: {obs.shape}"

        return obs

    def _auto_gear(self):
        """
        Automatic gear shifting with hysteresis to prevent oscillation.
        Never uses reverse gear - if car is going backwards, stay in gear 1
        and let brakes/physics handle it.
        """
        if self.client.S.d is None:
            return 1

        speed = self.client.S.d.get('speedX', 0)
        current_gear = self.client.S.d.get('gear', 1)

        # Gear shift thresholds: (upshift_speed, downshift_speed)
        gear_thresholds = {
            1: (55, None),    # Upshift to 2 at 55, no downshift from 1
            2: (85, 45),      # Upshift to 3 at 85, downshift to 1 at 45
            3: (125, 70),     # Upshift to 4 at 125, downshift to 2 at 70
            4: (160, 110),    # Upshift to 5 at 160, downshift to 3 at 110
            5: (200, 145),    # Upshift to 6 at 200, downshift to 4 at 145
            6: (None, 185),   # No upshift from 6, downshift to 5 at 185
        }

        # Never use reverse - always stay in gear 1 at low/negative speeds
        # This prevents the car from accelerating backwards after a spin
        if speed < 20:
            return 1

        # If somehow in reverse/neutral, go to gear 1
        if current_gear <= 0:
            return 1

        # Clamp to valid gear range
        current_gear = max(1, min(6, current_gear))

        upshift_speed, downshift_speed = gear_thresholds.get(current_gear, (None, None))

        # Check for upshift
        if upshift_speed is not None and speed > upshift_speed:
            return current_gear + 1

        # Check for downshift
        if downshift_speed is not None and speed < downshift_speed:
            return current_gear - 1

        # Stay in current gear
        return current_gear

    def _check_done(self, S):
        """
        Check if episode should terminate.

        Returns:
            (done: bool, dnf_reason: str or None)
        """
        # Max steps reached
        if self.step_count >= self.max_steps:
            return True, 'max_steps'

        # Target laps completed (success!)
        if self.lap_tracker.lap_count >= self.target_laps:
            return True, None  # Not a DNF, completed successfully

        # Give the car some time to stabilize after race start before checking termination
        # TORCS may have weird initial values in the first few frames
        # Car needs ~2-3 seconds to reach 150 km/h with launch assist
        grace_period = 150  # ~3 seconds of grace period at 50 FPS
        if self.step_count < grace_period:
            return False, None

        # Off track
        track_pos = S.get('trackPos', 0)
        if abs(track_pos) > self.off_track_threshold:
            return True, 'off_track'

        # Facing backward - only check after grace period and with significant speed
        # to avoid false positives from initial spawn orientation
        angle = S.get('angle', 0)
        speed = S.get('speedX', 0)
        if math.cos(angle) < 0 and speed > 5:  # Only if actually moving backward
            return True, 'facing_backward'

        # Stalled mid-race - car stopped moving for too long
        if abs(speed) < 5:
            self.stalled_steps += 1
            if self.stalled_steps >= 100:  # ~2 seconds at 50 FPS
                return True, 'stalled'
        else:
            self.stalled_steps = 0  # Reset if moving again

        # Stalled at start - be more lenient
        if not self.start_check_done and self.step_count >= self.start_check_steps:
            dist_raced = S.get('distRaced', 0)
            if dist_raced < self.start_min_distance:
                return True, 'stalled_start'
            self.start_check_done = True

        return False, None

    def _calculate_reward(self, S, done, dnf_reason):
        """
        Calculate reward for current step.

        SIMPLIFIED reward shaping focused on:
        1. Progress (distance) - main learning signal
        2. Track position - stay on track
        3. Steering smoothness - no jerky inputs
        4. Speed maintenance - don't stall
        5. Terminal bonuses/penalties
        """
        reward = 0.0

        # Get current state
        dist_raced = S.get('distRaced', 0)
        speed = S.get('speedX', 0)
        track_pos = abs(S.get('trackPos', 0))
        angle = abs(S.get('angle', 0))
        current_steer = self.client.R.d.get('steer', 0) if self.client.R.d else 0

        if self.reward_type == 'progress':
            # ===================
            # 1. PROGRESS REWARD (main signal)
            # ===================
            # Reward forward movement - this is what we ultimately want
            dist_progress = dist_raced - self.prev_dist_raced
            reward += dist_progress * 1.5

            # ===================
            # 2. TRACK POSITION PENALTY (graduated)
            # ===================
            # Start penalizing earlier and more strongly as agent approaches edge
            if track_pos > 0.5:
                # Quadratic penalty: gentle at 0.5, strong near 1.0
                track_penalty = -((track_pos - 0.5) ** 2) * 4.0
                reward += track_penalty

            # ===================
            # 3. STEERING SMOOTHNESS PENALTY
            # ===================
            # Penalize rapid changes in steering (bang-bang control)
            steer_change = abs(current_steer - self.prev_steer)
            if steer_change > 0.1:  # Allow small adjustments
                smoothness_penalty = -(steer_change - 0.1) * 2.0
                reward += smoothness_penalty

            # ===================
            # 4. SPEED MAINTENANCE (anti-stall)
            # ===================
            # Strong penalty for low speed to prevent stalling
            if speed < 20:
                # Severe penalty when nearly stopped
                reward -= 1.0 * (1.0 - speed / 20.0)
            elif speed < 50:
                # Mild penalty for crawling
                reward -= 0.3 * (1.0 - speed / 50.0)

            # Small bonus for maintaining good speed (encourages forward progress)
            if speed > 60:
                reward += 0.1

            # ===================
            # 5. ANGLE PENALTY (face forward)
            # ===================
            # Penalize being sideways - scaled by speed (worse at high speed)
            if angle > 0.1:  # ~6 degrees tolerance
                speed_factor = np.clip(speed / 100.0, 0.3, 1.0)
                angle_penalty = -(angle - 0.1) * 0.5 * speed_factor
                reward += angle_penalty

            # ===================
            # 6. LAP COMPLETION BONUS
            # ===================
            if self.lap_tracker.lap_just_completed:
                lap_time = self.lap_tracker.last_lap_time
                # Base bonus + time bonus (faster = more reward)
                lap_bonus = 500.0 + 500.0 * (60.0 / max(lap_time, 30.0))
                reward += lap_bonus

            # ===================
            # 7. TERMINAL PENALTIES
            # ===================
            if done and dnf_reason is not None:
                progress_fraction = min(dist_raced / 3610.0, 1.0)
                if dnf_reason == 'off_track':
                    # Penalty scales with how early the crash
                    reward -= 50.0 * (1.0 - 0.5 * progress_fraction)
                elif dnf_reason == 'facing_backward':
                    reward -= 50.0 * (1.0 - 0.5 * progress_fraction)
                elif dnf_reason in ('stalled_start', 'stalled'):
                    # Strong stall penalty to encourage movement
                    reward -= 75.0
                elif dnf_reason == 'max_steps':
                    reward -= 10.0

        elif self.reward_type == 'time':
            # Sparse reward - mainly for lap completion
            if self.lap_tracker.lap_just_completed:
                lap_time = self.lap_tracker.last_lap_time
                reward = 100.0 * (60.0 / max(lap_time, 30.0))
            else:
                dist_progress = dist_raced - self.prev_dist_raced
                reward = dist_progress * 0.02

            if done and dnf_reason is not None:
                reward -= 50.0

        # Update previous values for next step
        self.prev_dist_raced = dist_raced
        self.prev_speed = speed
        self.prev_steer = current_steer

        return reward

    def close(self):
        """Clean up environment."""
        self.connection_healthy = False
        if self.client is not None:
            try:
                self.client.shutdown()
            except Exception as e:
                print(f"Warning: shutdown failed: {e}")
            # Ensure socket is closed
            try:
                if self.client.so is not None:
                    self.client.so.close()
            except:
                pass
            self.client = None


class TorcsRLEnvVec:
    """
    Vectorized TORCS environment for parallel training.
    Creates multiple TORCS instances on different ports.

    Note: For use with Stable-Baselines3, prefer using DummyVecEnv or
    SubprocVecEnv wrappers around TorcsRLEnv instead, as they provide
    better integration with SB3's training infrastructure.

    This class implements auto-reset: when an episode ends, the environment
    is automatically reset and the new observation is returned.
    """

    def __init__(self, n_envs=4, base_port=3001, **env_kwargs):
        """
        Create multiple TORCS environments.

        Args:
            n_envs: Number of parallel environments
            base_port: Starting port number (each env gets base_port + i)
            **env_kwargs: Arguments passed to TorcsRLEnv
        """
        self.n_envs = n_envs
        self.envs = []
        self.env_kwargs = env_kwargs
        self.base_port = base_port

        for i in range(n_envs):
            port = base_port + i
            env = TorcsRLEnv(port=port, **env_kwargs)
            self.envs.append(env)

    def reset(self):
        """Reset all environments."""
        obs_list = []
        for env in self.envs:
            obs, info = env.reset()
            obs_list.append(obs)
        return np.array(obs_list)

    def step(self, actions):
        """
        Step all environments with corresponding actions.
        Implements auto-reset: when an episode ends, the environment is
        automatically reset and the new initial observation is returned.
        """
        obs_list = []
        rewards = []
        dones = []
        infos = []

        for i, (env, action) in enumerate(zip(self.envs, actions)):
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Auto-reset on episode end
            if done:
                # Store terminal observation in info for algorithms that need it
                info['terminal_observation'] = obs
                # Reset and get new initial observation
                obs, reset_info = env.reset()
                # Check if reset failed due to connection issues
                if reset_info.get('connection_failed', False):
                    info['connection_failed'] = True

            obs_list.append(obs)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)

        return np.array(obs_list), np.array(rewards), np.array(dones), infos

    def close(self):
        """Close all environments."""
        for env in self.envs:
            env.close()

    @property
    def observation_space(self):
        """Return observation space (same for all envs)."""
        return self.envs[0].observation_space

    @property
    def action_space(self):
        """Return action space (same for all envs)."""
        return self.envs[0].action_space


# Test the environment
if __name__ == "__main__":
    print("Testing TORCS RL Environment...")
    print("=" * 60)

    env = TorcsRLEnv(port=3001, max_steps=1000)

    print("Observation space:", env.observation_space.shape)
    print("Action space:", env.action_space.shape)
    print("=" * 60)

    # Test reset
    print("Resetting environment...")
    obs, info = env.reset()
    if info.get('connection_failed', False):
        print("Connection failed during reset!")
    else:
        print(f"Initial observation shape: {obs.shape}")
        print(f"Initial speed: {obs[0]:.2f} km/h")
    print("=" * 60)

    # Test random actions for a few steps
    print("Testing random actions...")
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        print(f"Step {i+1}: speed={obs[0]:.1f}, reward={reward:.2f}, done={done}")

        if done:
            print(f"Episode ended: {info.get('dnf_reason', 'completed')}")
            break

    env.close()
    print("=" * 60)
    print("Environment test completed!")
