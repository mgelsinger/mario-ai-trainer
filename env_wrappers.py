"""
Environment wrappers for Super Mario Bros RL training.
Bridges gym-super-mario-bros (old gym API) to gymnasium and applies
standard Atari-style preprocessing plus custom reward shaping.
"""

import collections
import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Lazy-loaded movement type mapping (imports from gym_super_mario_bros)
_MOVEMENT_MAP = None

def _get_movement_map():
    global _MOVEMENT_MAP
    if _MOVEMENT_MAP is None:
        from gym_super_mario_bros.actions import RIGHT_ONLY, SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
        _MOVEMENT_MAP = {
            "right_only": RIGHT_ONLY,
            "simple": SIMPLE_MOVEMENT,
            "complex": COMPLEX_MOVEMENT,
        }
    return _MOVEMENT_MAP


class GymToGymnasiumAdapter(gym.Env):
    """Adapts old gym API (4-return step) to gymnasium API (5-return step)."""

    def __init__(self, old_env):
        super().__init__()
        self._env = old_env
        self.observation_space = self._convert_space(old_env.observation_space)
        self.action_space = self._convert_space(old_env.action_space)
        self.metadata = getattr(old_env, 'metadata', {})

    def _convert_space(self, space):
        import gym as old_gym
        if isinstance(space, old_gym.spaces.Box):
            return spaces.Box(low=space.low, high=space.high, shape=space.shape, dtype=space.dtype)
        elif isinstance(space, old_gym.spaces.Discrete):
            return spaces.Discrete(space.n)
        elif isinstance(space, old_gym.spaces.MultiBinary):
            return spaces.MultiBinary(space.n)
        elif isinstance(space, old_gym.spaces.MultiDiscrete):
            return spaces.MultiDiscrete(space.nvec)
        return space

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        return obs, reward, done, False, info

    def reset(self, *, seed=None, options=None):
        obs = self._env.reset()
        return obs, {}

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()


class FrameSkipWrapper(gym.Wrapper):
    """Repeat action for N frames, accumulate reward, return max of last 2 frames.

    The max-pooling prevents NES sprite flickering artifacts where enemies
    can be invisible on alternating frames.
    """

    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip
        self._obs_buffer = [None, None]

    def step(self, action):
        total_reward = 0.0
        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            self._obs_buffer[i % 2] = obs
            if terminated or truncated:
                break
        # Max of last 2 frames to prevent sprite flickering
        if self._obs_buffer[0] is not None and self._obs_buffer[1] is not None:
            obs = np.maximum(self._obs_buffer[0], self._obs_buffer[1])
        return obs, total_reward, terminated, truncated, info


class GrayscaleResizeWrapper(gym.ObservationWrapper):
    """Convert to grayscale and resize to 84x84."""

    def __init__(self, env, width=84, height=84):
        super().__init__(env)
        self._width = width
        self._height = height
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(height, width, 1), dtype=np.uint8
        )

    def observation(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (self._width, self._height), interpolation=cv2.INTER_AREA)
        return resized[:, :, np.newaxis]


class NormalizeObservationWrapper(gym.ObservationWrapper):
    """Normalize pixel values to [0, 1] float32."""

    def __init__(self, env):
        super().__init__(env)
        old_space = env.observation_space
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=old_space.shape, dtype=np.float32
        )

    def observation(self, obs):
        return obs.astype(np.float32) / 255.0


class FrameStackWrapper(gym.Wrapper):
    """Stack N consecutive frames along the last axis."""

    def __init__(self, env, n_frames=4):
        super().__init__(env)
        self._n_frames = n_frames
        self._frames = collections.deque(maxlen=n_frames)
        old_space = env.observation_space
        h, w, c = old_space.shape
        self.observation_space = spaces.Box(
            low=np.repeat(old_space.low, n_frames, axis=-1),
            high=np.repeat(old_space.high, n_frames, axis=-1),
            shape=(h, w, c * n_frames),
            dtype=old_space.dtype,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self._n_frames):
            self._frames.append(obs)
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        return np.concatenate(list(self._frames), axis=-1)


class RewardShapingWrapper(gym.Wrapper):
    """Custom reward shaping for Mario.

    Keeps rewards in a stable range via clipping (default [-15, 15]),
    matching the scale of the built-in gym-super-mario-bros reward.
    """

    def __init__(self, env, progress_weight=1.0, time_penalty=0.0,
                 death_penalty=15.0, flag_bonus=15.0, reward_clip=15.0):
        super().__init__(env)
        self._progress_weight = progress_weight
        self._time_penalty = time_penalty
        self._death_penalty = death_penalty
        self._flag_bonus = flag_bonus
        self._reward_clip = reward_clip
        self._last_x_pos = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_x_pos = info.get("x_pos", 0)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Progress reward: reward for moving right
        x_pos = info.get("x_pos", 0)
        x_delta = x_pos - self._last_x_pos
        shaped_reward = x_delta * self._progress_weight

        # Time penalty (per decision step — default 0 since standing still
        # already yields x_delta=0 which is its own punishment)
        shaped_reward -= self._time_penalty

        # Death penalty
        if terminated and not info.get("flag_get", False):
            shaped_reward -= self._death_penalty

        # Flag bonus
        if info.get("flag_get", False):
            shaped_reward += self._flag_bonus

        # Clip to prevent extreme values that destabilize training
        if self._reward_clip > 0:
            shaped_reward = max(-self._reward_clip, min(self._reward_clip, shaped_reward))

        self._last_x_pos = x_pos
        info["shaped_reward"] = shaped_reward

        return obs, shaped_reward, terminated, truncated, info


class EpisodeRecorderWrapper(gym.Wrapper):
    """Records raw color frames for replay viewing. Keeps best recording separately."""

    def __init__(self, env, record_interval=10, max_recordings=10):
        super().__init__(env)
        self.record_interval = record_interval
        self.max_recordings = max_recordings
        self._recordings_list = []
        self._best_recording = None
        self._episode_count = 0
        self._current_frames = []
        self._recording = False
        self._episode_x_pos = 0
        self._episode_reward = 0.0

    @property
    def recordings(self):
        """Return all recordings: regular + best (if not already included)."""
        result = list(self._recordings_list)
        if self._best_recording:
            best_ep = self._best_recording["episode"]
            if not any(r["episode"] == best_ep for r in result):
                result.append(self._best_recording)
        return result

    def _save_recording(self, recording):
        """Save a recording, maintaining best and capacity limits."""
        if self._best_recording is None or recording["x_pos"] > self._best_recording.get("x_pos", 0):
            self._best_recording = recording
        self._recordings_list.append(recording)
        max_regular = max(1, self.max_recordings - 1)
        while len(self._recordings_list) > max_regular:
            best_ep = self._best_recording["episode"] if self._best_recording else -1
            removed = False
            for i, r in enumerate(self._recordings_list):
                if r["episode"] != best_ep:
                    self._recordings_list.pop(i)
                    removed = True
                    break
            if not removed:
                break

    def reset(self, **kwargs):
        if self._recording and self._current_frames:
            self._save_recording({
                "episode": self._episode_count,
                "frames": self._current_frames,
                "x_pos": self._episode_x_pos,
                "reward": self._episode_reward,
            })
            self._current_frames = []

        self._episode_count += 1
        self._recording = (self._episode_count % self.record_interval == 0)
        self._episode_x_pos = 0
        self._episode_reward = 0.0

        obs, info = self.env.reset(**kwargs)

        if self._recording:
            raw_frame = self._get_raw_frame()
            if raw_frame is not None:
                self._current_frames.append(raw_frame)

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._episode_x_pos = info.get("x_pos", self._episode_x_pos)
        self._episode_reward += reward

        if self._recording:
            raw_frame = self._get_raw_frame()
            if raw_frame is not None:
                self._current_frames.append(raw_frame)

        if (terminated or truncated) and self._recording and self._current_frames:
            self._save_recording({
                "episode": self._episode_count,
                "frames": self._current_frames,
                "x_pos": self._episode_x_pos,
                "reward": self._episode_reward,
            })
            self._current_frames = []
            self._recording = False

        return obs, reward, terminated, truncated, info

    def _get_raw_frame(self):
        """Get the raw color frame from the underlying NES environment."""
        return get_raw_frame_from_env(self.env)


def get_raw_frame_from_env(env):
    """Get the raw color frame from a wrapped NES environment.

    Walks the wrapper chain to find the underlying NES screen buffer.
    Works with any wrapper depth.
    """
    current = env
    while hasattr(current, 'env'):
        if hasattr(current, '_env') and hasattr(current._env, 'screen'):
            return current._env.screen.copy() if current._env.screen is not None else None
        if hasattr(current, 'render'):
            try:
                frame = current.render()
                if frame is not None and isinstance(frame, np.ndarray):
                    return frame.copy()
            except Exception:
                pass
        current = current.env
    # Try the innermost env
    if hasattr(current, '_env') and hasattr(current._env, 'screen'):
        return current._env.screen.copy() if current._env.screen is not None else None
    if hasattr(current, 'screen'):
        return current.screen.copy() if current.screen is not None else None
    try:
        frame = current.render()
        if frame is not None and isinstance(frame, np.ndarray):
            return frame.copy()
    except Exception:
        pass
    return None


def make_mario_env(world=1, stage=1, movement_type="right_only", skip_frames=4,
                   frame_stack=4, progress_weight=1.0, time_penalty=0.0,
                   death_penalty=15.0, flag_bonus=15.0, reward_clip=15.0,
                   record_interval=10, env_index=0):
    """Create a fully wrapped Mario environment.

    Returns the wrapped env and a reference to the recorder wrapper.
    """
    from nes_py.wrappers import JoypadSpace
    from gym_super_mario_bros import SuperMarioBrosEnv

    actions = _get_movement_map().get(movement_type, _get_movement_map()["right_only"])

    raw_env = SuperMarioBrosEnv(rom_mode="vanilla", target=(world, stage))
    raw_env = JoypadSpace(raw_env, actions)

    # Bridge to gymnasium API
    env = GymToGymnasiumAdapter(raw_env)

    # Record raw frames BEFORE preprocessing (only on env 0)
    recorder = None
    if env_index == 0:
        recorder = EpisodeRecorderWrapper(env, record_interval=record_interval)
        env = recorder

    # Frame skip
    env = FrameSkipWrapper(env, skip=skip_frames)

    # Reward shaping
    env = RewardShapingWrapper(env, progress_weight=progress_weight,
                                time_penalty=time_penalty, death_penalty=death_penalty,
                                flag_bonus=flag_bonus, reward_clip=reward_clip)

    # Grayscale + resize (keeps uint8 0-255 for SB3 CnnPolicy compatibility)
    env = GrayscaleResizeWrapper(env)

    # Frame stacking
    env = FrameStackWrapper(env, n_frames=frame_stack)

    return env, recorder


def make_mario_training_env(world=1, stage=1, movement_type="right_only", skip_frames=4,
                            frame_stack=4, progress_weight=1.0, time_penalty=0.0,
                            death_penalty=15.0, flag_bonus=15.0, reward_clip=15.0):
    """Create a Mario environment for training (no recorder).

    Used by SubprocVecEnv where each env runs in a child process.
    Returns just the env (not a tuple), since SubprocVecEnv expects a callable
    returning a single env.
    """
    from nes_py.wrappers import JoypadSpace
    from gym_super_mario_bros import SuperMarioBrosEnv

    actions = _get_movement_map().get(movement_type, _get_movement_map()["right_only"])

    raw_env = SuperMarioBrosEnv(rom_mode="vanilla", target=(world, stage))
    raw_env = JoypadSpace(raw_env, actions)

    env = GymToGymnasiumAdapter(raw_env)
    env = FrameSkipWrapper(env, skip=skip_frames)
    env = RewardShapingWrapper(env, progress_weight=progress_weight,
                                time_penalty=time_penalty, death_penalty=death_penalty,
                                flag_bonus=flag_bonus, reward_clip=reward_clip)
    env = GrayscaleResizeWrapper(env)
    env = FrameStackWrapper(env, n_frames=frame_stack)

    return env
