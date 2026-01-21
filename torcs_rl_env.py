"""
Gym-compatible TORCS environment for reinforcement learning.
Optimized for single lap time from standing start.
"""

import gym
from gym import spaces
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
                 off_track_threshold=1.3, start_min_distance=25,
                 start_check_steps=250, reward_type='progress'):
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
        """
        super(TorcsRLEnv, self).__init__()

        self.port = port
        self.max_steps = max_steps
        self.target_laps = target_laps
        self.off_track_threshold = off_track_threshold
        self.start_min_distance = start_min_distance
        self.start_check_steps = start_check_steps
        self.reward_type = reward_type

        # Define action space: [steering, accel, brake]
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0]),
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
        self.start_check_done = False
        self.total_reward = 0.0
        self.best_lap_time = None

    def reset(self):
        """Reset the environment and return initial observation."""
        # Shutdown previous client if exists
        if self.client is not None:
            try:
                self.client.restart_race()
            except:
                pass

        # Create new client connection
        self.client = Client(p=self.port)
        self.client.maxSteps = self.max_steps
        self.lap_tracker = LapTimeTracker()

        # Get initial state
        self.client.get_servers_input()

        # Reset episode tracking
        self.step_count = 0
        self.episode_count += 1
        self.prev_dist_raced = 0.0
        self.prev_speed = 0.0
        self.start_check_done = False
        self.total_reward = 0.0

        # Return initial observation
        obs = self._get_observation()
        return obs

    def step(self, action):
        """
        Execute one step in the environment.

        Args:
            action: [steering, accel, brake]

        Returns:
            observation, reward, done, info
        """
        self.step_count += 1

        # Apply action
        self.client.R.d['steer'] = np.clip(action[0], -1.0, 1.0)
        self.client.R.d['accel'] = np.clip(action[1], 0.0, 1.0)
        self.client.R.d['brake'] = np.clip(action[2], 0.0, 1.0)

        # Simple automatic gear shifting
        self.client.R.d['gear'] = self._auto_gear()

        # Send action and get response
        self.client.respond_to_server()
        self.client.get_servers_input()

        # Check if connection closed
        if self.client.so is None or self.client.S.d is None:
            obs = self._get_observation()
            return obs, -100.0, True, {'dnf': True, 'dnf_reason': 'connection_lost'}

        S = self.client.S.d

        # Update lap tracker
        self.lap_tracker.update(S)

        # Get current state
        obs = self._get_observation()

        # Check termination conditions
        done, dnf_reason = self._check_done(S)

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

        # Add lap time info if lap completed
        if self.lap_tracker.lap_just_completed:
            info['lap_time'] = self.lap_tracker.last_lap_time
            info['best_lap_time'] = self.lap_tracker.best_lap_time

            # Update best lap time
            if self.best_lap_time is None or self.lap_tracker.last_lap_time < self.best_lap_time:
                self.best_lap_time = self.lap_tracker.last_lap_time

        return obs, reward, done, info

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
        """Simple automatic gear shifting based on speed."""
        if self.client.S.d is None:
            return 1

        speed = self.client.S.d.get('speedX', 0)

        if speed < 0:
            return -1
        elif speed < 50:
            return 1
        elif speed < 80:
            return 2
        elif speed < 120:
            return 3
        elif speed < 150:
            return 4
        elif speed < 200:
            return 5
        else:
            return 6

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

        # Off track
        track_pos = S.get('trackPos', 0)
        if abs(track_pos) > self.off_track_threshold:
            return True, 'off_track'

        # Facing backward
        angle = S.get('angle', 0)
        if math.cos(angle) < 0:
            return True, 'facing_backward'

        # Stalled at start
        if not self.start_check_done and self.step_count >= self.start_check_steps:
            dist_raced = S.get('distRaced', 0)
            if dist_raced < self.start_min_distance:
                return True, 'stalled_start'
            self.start_check_done = True

        return False, None

    def _calculate_reward(self, S, done, dnf_reason):
        """
        Calculate reward for current step.

        Reward shaping for lap time optimization from standing start:
        1. Progress reward: encourage forward movement
        2. Speed bonus: reward maintaining high speed
        3. Track position penalty: stay on racing line
        4. Lap completion bonus: big reward for completing lap
        5. DNF penalty: heavy penalty for crashes/off-track
        """
        reward = 0.0

        # Get current state
        dist_raced = S.get('distRaced', 0)
        speed = S.get('speedX', 0)
        track_pos = abs(S.get('trackPos', 0))
        angle = abs(S.get('angle', 0))

        if self.reward_type == 'progress':
            # 1. Progress reward (main driver)
            # Reward distance traveled weighted by speed efficiency
            dist_progress = dist_raced - self.prev_dist_raced
            reward += dist_progress * 0.1

            # 2. Speed bonus (encourage maintaining high speed)
            # Normalized speed reward (0 to 1 for speeds 0 to 300 km/h)
            speed_normalized = np.clip(speed / 300.0, 0, 1)
            reward += speed_normalized * 0.05

            # 3. Track position penalty (stay near center)
            # Quadratic penalty for being off-center
            track_penalty = -track_pos ** 2 * 0.5
            reward += track_penalty

            # 4. Angle penalty (face forward)
            angle_penalty = -abs(angle) * 0.2
            reward += angle_penalty

            # 5. Lap completion bonus
            if self.lap_tracker.lap_just_completed:
                # Big bonus inversely proportional to lap time
                # Assuming 30-120s lap times, give 500-2000 bonus
                lap_time = self.lap_tracker.last_lap_time
                lap_bonus = 1000.0 / max(lap_time, 30.0) * 30.0
                reward += lap_bonus

            # 6. DNF penalty
            if done and dnf_reason is not None:
                if dnf_reason == 'off_track':
                    reward -= 100.0
                elif dnf_reason == 'facing_backward':
                    reward -= 100.0
                elif dnf_reason == 'stalled_start':
                    reward -= 100.0
                elif dnf_reason == 'max_steps':
                    # Mild penalty for not completing
                    reward -= 10.0

        elif self.reward_type == 'time':
            # Alternative: sparse reward based on lap time
            # Only reward at lap completion
            if self.lap_tracker.lap_just_completed:
                lap_time = self.lap_tracker.last_lap_time
                # Negative time (lower is better)
                reward = -lap_time
            else:
                # Small progress reward to guide learning
                dist_progress = dist_raced - self.prev_dist_raced
                reward = dist_progress * 0.01

            # DNF penalty
            if done and dnf_reason is not None:
                reward -= 200.0

        # Update previous values
        self.prev_dist_raced = dist_raced
        self.prev_speed = speed

        return reward

    def close(self):
        """Clean up environment."""
        if self.client is not None:
            try:
                self.client.shutdown()
            except:
                pass
            self.client = None


class TorcsRLEnvVec:
    """
    Vectorized TORCS environment for parallel training.
    Creates multiple TORCS instances on different ports.
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

        for i in range(n_envs):
            port = base_port + i
            env = TorcsRLEnv(port=port, **env_kwargs)
            self.envs.append(env)

    def reset(self):
        """Reset all environments."""
        obs_list = []
        for env in self.envs:
            obs = env.reset()
            obs_list.append(obs)
        return np.array(obs_list)

    def step(self, actions):
        """Step all environments with corresponding actions."""
        obs_list = []
        rewards = []
        dones = []
        infos = []

        for env, action in zip(self.envs, actions):
            obs, reward, done, info = env.step(action)
            obs_list.append(obs)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)

        return np.array(obs_list), np.array(rewards), np.array(dones), infos

    def close(self):
        """Close all environments."""
        for env in self.envs:
            env.close()


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
    obs = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial speed: {obs[0]:.2f} km/h")
    print("=" * 60)

    # Test random actions for a few steps
    print("Testing random actions...")
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(f"Step {i+1}: speed={obs[0]:.1f}, reward={reward:.2f}, done={done}")

        if done:
            print(f"Episode ended: {info.get('dnf_reason', 'completed')}")
            break

    env.close()
    print("=" * 60)
    print("Environment test completed!")
