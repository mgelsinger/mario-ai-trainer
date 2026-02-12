"""
Training pipeline for Mario AI using PPO from Stable-Baselines3.
Runs in a background thread, streams metrics via callback.
"""

import json as _json
import os
import time
import threading
import numpy as np
from collections import deque
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from env_wrappers import make_mario_env, make_mario_training_env, get_raw_frame_from_env


def get_best_device():
    """Auto-detect best available device."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def linear_schedule(initial_value: float):
    """Linear LR schedule — decays from initial_value to 0 over training."""
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


class MetricsCallback(BaseCallback):
    """Streams per-episode metrics to a callback function."""

    def __init__(self, on_metrics=None, on_log=None, total_timesteps=0, verbose=0):
        super().__init__(verbose)
        self.on_metrics = on_metrics
        self.on_log = on_log
        self.total_timesteps = total_timesteps
        self._episode_count = 0
        self._episode_rewards = deque(maxlen=100)
        self._episode_lengths = deque(maxlen=100)
        self._episode_x_positions = deque(maxlen=100)
        self._max_x_pos = 0
        self._start_time = None
        self._last_fps_time = None
        self._last_fps_steps = 0
        self._current_fps = 0.0
        self._env_rewards = {}
        self._env_lengths = {}
        self._env_x_pos = {}

    def _on_training_start(self):
        self._start_time = time.time()
        self._last_fps_time = time.time()
        self._last_fps_steps = 0
        n_envs = self.training_env.num_envs
        for i in range(n_envs):
            self._env_rewards[i] = 0.0
            self._env_lengths[i] = 0
            self._env_x_pos[i] = 0

    def _on_step(self):
        now = time.time()
        if now - self._last_fps_time >= 1.0:
            steps_delta = self.num_timesteps - self._last_fps_steps
            time_delta = now - self._last_fps_time
            self._current_fps = steps_delta / time_delta if time_delta > 0 else 0
            self._last_fps_time = now
            self._last_fps_steps = self.num_timesteps

        n_envs = self.training_env.num_envs
        for i in range(n_envs):
            reward = self.locals["rewards"][i]
            self._env_rewards[i] = self._env_rewards.get(i, 0) + float(reward)
            self._env_lengths[i] = self._env_lengths.get(i, 0) + 1

            info = self.locals["infos"][i]
            x_pos = info.get("x_pos", 0)
            self._env_x_pos[i] = x_pos

            done = self.locals["dones"][i]
            if done:
                self._episode_count += 1
                ep_reward = self._env_rewards[i]
                ep_length = self._env_lengths[i]
                ep_x_pos = self._env_x_pos[i]

                self._episode_rewards.append(ep_reward)
                self._episode_lengths.append(ep_length)
                self._episode_x_positions.append(ep_x_pos)
                self._max_x_pos = max(self._max_x_pos, ep_x_pos)

                self._env_rewards[i] = 0.0
                self._env_lengths[i] = 0
                self._env_x_pos[i] = 0

                if self.on_metrics:
                    elapsed = now - self._start_time if self._start_time else 0
                    metrics = {
                        "episode": int(self._episode_count),
                        "timestep": int(self.num_timesteps),
                        "reward": round(float(ep_reward), 2),
                        "episode_length": int(ep_length),
                        "x_pos": int(ep_x_pos),
                        "avg_reward": round(float(np.mean(self._episode_rewards)), 2) if self._episode_rewards else 0,
                        "avg_length": round(float(np.mean(self._episode_lengths)), 2) if self._episode_lengths else 0,
                        "avg_x_pos": round(float(np.mean(self._episode_x_positions)), 2) if self._episode_x_positions else 0,
                        "max_x_pos": int(self._max_x_pos),
                        "fps": round(float(self._current_fps), 1),
                        "elapsed": round(float(elapsed), 1),
                        "total_timesteps": int(self.total_timesteps),
                    }
                    self.on_metrics(metrics)

        return True


class CheckpointCallback(BaseCallback):
    """Saves model checkpoints periodically with metadata."""

    def __init__(self, save_freq=50000, save_path="checkpoints", metrics_cb=None,
                 on_log=None, world=1, stage=1, movement_type="right_only", verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.metrics_cb = metrics_cb
        self.on_log = on_log
        self.world = world
        self.stage = stage
        self.movement_type = movement_type
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self):
        if self.num_timesteps % self.save_freq < self.training_env.num_envs:
            path = os.path.join(self.save_path, f"mario_ppo_{self.num_timesteps}")
            self.model.save(path)
            meta = {
                "timestep": int(self.num_timesteps),
                "timestamp": time.time(),
                "world": self.world,
                "stage": self.stage,
                "movement_type": self.movement_type,
            }
            if self.metrics_cb and self.metrics_cb._episode_rewards:
                meta["avg_reward"] = round(float(np.mean(self.metrics_cb._episode_rewards)), 2)
                meta["max_x_pos"] = int(self.metrics_cb._max_x_pos)
                meta["episode"] = int(self.metrics_cb._episode_count)
            with open(path + ".meta.json", "w") as f:
                _json.dump(meta, f, indent=2)
            if self.on_log:
                self.on_log(f"Checkpoint saved: {path}")
        return True


class BestModelCallback(BaseCallback):
    """Auto-saves model when avg_reward exceeds previous best."""

    def __init__(self, metrics_cb, save_path="checkpoints", on_log=None,
                 world=1, stage=1, movement_type="right_only", verbose=0):
        super().__init__(verbose)
        self.metrics_cb = metrics_cb
        self.save_path = save_path
        self.on_log = on_log
        self.world = world
        self.stage = stage
        self.movement_type = movement_type
        self.best_avg_reward = float("-inf")
        self._check_interval = 0
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self):
        # Only check every ~100 steps to avoid excessive IO
        self._check_interval += 1
        if self._check_interval % 100 != 0:
            return True
        if len(self.metrics_cb._episode_rewards) < 20:
            return True
        avg = float(np.mean(self.metrics_cb._episode_rewards))
        if avg > self.best_avg_reward:
            self.best_avg_reward = avg
            path = os.path.join(self.save_path, f"mario_ppo_best_w{self.world}_s{self.stage}")
            self.model.save(path)
            meta = {
                "avg_reward": round(avg, 2),
                "max_x_pos": int(self.metrics_cb._max_x_pos),
                "avg_x_pos": round(float(np.mean(self.metrics_cb._episode_x_positions)), 2),
                "episode": int(self.metrics_cb._episode_count),
                "timestep": int(self.num_timesteps),
                "timestamp": time.time(),
                "type": "best",
                "world": self.world,
                "stage": self.stage,
                "movement_type": self.movement_type,
            }
            with open(path + ".meta.json", "w") as f:
                _json.dump(meta, f, indent=2)
            if self.on_log:
                self.on_log(f"New best model! avg_reward={avg:.1f}, max_x={self.metrics_cb._max_x_pos}")
        return True


class StopTrainingCallback(BaseCallback):
    """Checks a stop event to allow graceful training interruption."""

    def __init__(self, stop_event, verbose=0):
        super().__init__(verbose)
        self.stop_event = stop_event

    def _on_step(self):
        return not self.stop_event.is_set()


class EvalRecordCallback(BaseCallback):
    """Periodically runs eval episodes in a separate env with recording."""

    def __init__(self, eval_env, recorder, n_eval_episodes=3,
                 eval_freq=50000, on_metrics=None, on_log=None, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.recorder = recorder
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.on_metrics = on_metrics
        self.on_log = on_log
        self._last_eval = 0

    def _on_step(self):
        if self.num_timesteps - self._last_eval < self.eval_freq:
            return True

        self._last_eval = self.num_timesteps

        rewards = []
        x_positions = []
        flag_gets = []

        for ep in range(self.n_eval_episodes):
            obs, info = self.eval_env.reset()
            done = False
            ep_reward = 0.0
            ep_x_pos = 0
            ep_flag = False

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                # predict() returns a numpy array; raw gym env needs a plain int
                action = int(action)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated
                ep_reward += float(reward)
                ep_x_pos = info.get("x_pos", ep_x_pos)
                if info.get("flag_get", False):
                    ep_flag = True

            rewards.append(ep_reward)
            x_positions.append(ep_x_pos)
            flag_gets.append(ep_flag)

        avg_reward = float(np.mean(rewards))
        avg_x_pos = float(np.mean(x_positions))
        flag_rate = sum(flag_gets) / len(flag_gets)

        if self.on_log:
            self.on_log(
                f"Eval ({self.n_eval_episodes} eps): "
                f"avg_reward={avg_reward:.1f}, avg_x={avg_x_pos:.0f}, "
                f"flag_rate={flag_rate:.0%}"
            )

        if self.on_metrics:
            self.on_metrics({
                "eval": True,
                "eval_reward": round(avg_reward, 2),
                "eval_x_pos": round(avg_x_pos, 1),
                "eval_flag_rate": round(flag_rate, 2),
                "timestep": int(self.num_timesteps),
            })

        return True


class MarioTrainer:
    """Manages the training lifecycle."""

    def __init__(self):
        self.model = None
        self.env = None
        self.eval_env = None
        self.recorder = None
        self.training_thread = None
        self.stop_event = threading.Event()
        self.is_training = False
        self.device = "cpu"
        self.current_params = {}
        self._on_metrics = None
        self._on_log = None
        self._checkpoint_path = "checkpoints"
        self._resume_checkpoint = None
        # Live play state
        self.play_env = None
        self.play_thread = None
        self.play_stop_event = threading.Event()
        self.is_playing = False
        self._on_play_frame = None
        self._on_play_end = None

    def set_callbacks(self, on_metrics=None, on_log=None):
        self._on_metrics = on_metrics
        self._on_log = on_log

    def start_training(self, params, checkpoint_path=None):
        """Start training in a background thread."""
        if self.is_training:
            return False

        self.stop_event.clear()
        self.current_params = params
        self.device = get_best_device()
        self._resume_checkpoint = checkpoint_path

        self.training_thread = threading.Thread(target=self._train, args=(params,), daemon=True)
        self.training_thread.start()
        self.is_training = True
        return True

    def stop_training(self):
        """Signal training to stop."""
        if not self.is_training:
            return False
        self.stop_event.set()
        return True

    def _train(self, params):
        """Training loop - runs in background thread."""
        try:
            if self._on_log:
                self._on_log(f"Initializing training on device: {self.device}")

            n_envs = params.get("n_envs", 8)

            # Common env params (no recorder for training envs)
            env_kwargs = dict(
                world=params.get("world", 1),
                stage=params.get("stage", 1),
                movement_type=params.get("movement_type", "right_only"),
                skip_frames=params.get("skip_frames", 4),
                frame_stack=params.get("frame_stack", 4),
                progress_weight=params.get("progress_weight", 1.0),
                time_penalty=params.get("time_penalty", 0.0),
                death_penalty=params.get("death_penalty", 15.0),
                flag_bonus=params.get("flag_bonus", 15.0),
                reward_clip=params.get("reward_clip", 15.0),
            )

            def make_env(idx):
                def _init():
                    return make_mario_training_env(**env_kwargs)
                return _init

            if self._on_log:
                self._on_log(f"Initializing {n_envs} environments...")

            env_fns = [make_env(i) for i in range(n_envs)]

            # Try SubprocVecEnv for true parallelism, fall back to DummyVecEnv
            try:
                self.env = SubprocVecEnv(env_fns)
                if self._on_log:
                    self._on_log(f"Created {n_envs} parallel environments (SubprocVecEnv)")
            except Exception as exc:
                if self._on_log:
                    self._on_log(f"SubprocVecEnv failed ({exc}), falling back to DummyVecEnv")
                env_fns = [make_env(i) for i in range(n_envs)]
                self.env = DummyVecEnv(env_fns)
                if self._on_log:
                    self._on_log(f"Created {n_envs} sequential environments (DummyVecEnv)")

            # Separate eval env in main process (with recorder for replays).
            # Uses raw gymnasium env (no DummyVecEnv) to avoid auto-reset
            # double-counting episodes in the recorder. record_interval=1
            # because eval frequency is controlled by eval_freq instead.
            self.eval_env, self.recorder = make_mario_env(
                **env_kwargs,
                record_interval=1,
                env_index=0,
            )

            # Determine learning rate (with optional annealing)
            lr_value = params.get("learning_rate", 1e-4)
            lr_annealing = params.get("lr_annealing", "linear")
            if lr_annealing == "linear":
                lr = linear_schedule(lr_value)
                if self._on_log:
                    self._on_log(f"Using linear LR annealing: {lr_value} -> 0")
            else:
                lr = lr_value

            # Extract level params early (needed for checkpoint mismatch check)
            world = params.get("world", 1)
            stage = params.get("stage", 1)
            movement_type = params.get("movement_type", "right_only")

            # Check for checkpoint to resume from
            resume_path = self._resume_checkpoint or self._find_latest_checkpoint()
            loaded_checkpoint = False

            if resume_path and os.path.exists(resume_path + ".zip"):
                try:
                    if self._on_log:
                        self._on_log(f"Resuming from checkpoint: {resume_path}")
                    # Check for world/stage mismatch
                    meta_path = resume_path + ".meta.json"
                    if os.path.exists(meta_path):
                        try:
                            with open(meta_path) as mf:
                                cp_meta = _json.load(mf)
                            cp_world = cp_meta.get("world")
                            cp_stage = cp_meta.get("stage")
                            if cp_world and cp_stage:
                                if cp_world != world or cp_stage != stage:
                                    if self._on_log:
                                        self._on_log(
                                            f"Note: Checkpoint trained on W{cp_world}-{cp_stage}, "
                                            f"now training on W{world}-{stage} (transfer learning)"
                                        )
                        except Exception:
                            pass
                    self.model = PPO.load(resume_path, env=self.env, device=self.device)
                    self.model.learning_rate = lr
                    self.model.ent_coef = params.get("ent_coef", 0.01)
                    self.model.clip_range = params.get("clip_range", 0.2)
                    loaded_checkpoint = True
                except Exception as load_err:
                    if self._on_log:
                        self._on_log(f"Failed to load checkpoint ({load_err}). Starting fresh model.")

            if not loaded_checkpoint:
                self.model = PPO(
                    "CnnPolicy",
                    self.env,
                    learning_rate=lr,
                    n_steps=params.get("n_steps", 512),
                    batch_size=params.get("batch_size", 64),
                    n_epochs=params.get("n_epochs", 10),
                    gamma=params.get("gamma", 0.9),
                    gae_lambda=params.get("gae_lambda", 0.95),
                    clip_range=params.get("clip_range", 0.2),
                    ent_coef=params.get("ent_coef", 0.01),
                    vf_coef=params.get("vf_coef", 0.5),
                    max_grad_norm=params.get("max_grad_norm", 0.5),
                    verbose=0,
                    device=self.device,
                )

            if self._on_log:
                self._on_log(f"PPO model initialized (CnnPolicy, device={self.device})")
                self._on_log("Training started...")

            # Callbacks
            total_timesteps = params.get("total_timesteps", 5_000_000)
            metrics_cb = MetricsCallback(on_metrics=self._on_metrics, on_log=self._on_log, total_timesteps=total_timesteps)
            checkpoint_cb = CheckpointCallback(
                save_freq=50000, save_path=self._checkpoint_path,
                metrics_cb=metrics_cb, on_log=self._on_log,
                world=world, stage=stage, movement_type=movement_type,
            )
            best_cb = BestModelCallback(
                metrics_cb, save_path=self._checkpoint_path, on_log=self._on_log,
                world=world, stage=stage, movement_type=movement_type,
            )
            stop_cb = StopTrainingCallback(self.stop_event)
            eval_cb = EvalRecordCallback(
                eval_env=self.eval_env,
                recorder=self.recorder,
                n_eval_episodes=3,
                eval_freq=50000,
                on_metrics=self._on_metrics,
                on_log=self._on_log,
            )

            self.model.learn(
                total_timesteps=total_timesteps,
                callback=[metrics_cb, checkpoint_cb, best_cb, eval_cb, stop_cb],
                progress_bar=False,
            )

            # Save final model with metadata
            final_path = os.path.join(self._checkpoint_path, "mario_ppo_final")
            self.model.save(final_path)
            meta = {
                "timestamp": time.time(),
                "type": "final",
                "world": world,
                "stage": stage,
                "movement_type": movement_type,
            }
            if metrics_cb._episode_rewards:
                meta["avg_reward"] = round(float(np.mean(metrics_cb._episode_rewards)), 2)
                meta["max_x_pos"] = int(metrics_cb._max_x_pos)
                meta["episode"] = int(metrics_cb._episode_count)
            with open(final_path + ".meta.json", "w") as f:
                _json.dump(meta, f, indent=2)
            if self._on_log:
                self._on_log(f"Training complete. Final model saved: {final_path}")

        except Exception as e:
            if self._on_log:
                self._on_log(f"Training error: {str(e)}")
            if self._on_metrics:
                self._on_metrics({"type": "training_error", "error": str(e)})
            import traceback
            traceback.print_exc()
        finally:
            self.is_training = False
            self._resume_checkpoint = None
            # Broadcast stopped status so frontend exits training state
            if self._on_metrics:
                self._on_metrics({"type": "status", "is_training": False})
            if self.env:
                try:
                    self.env.close()
                except Exception:
                    pass
                self.env = None
            if self.eval_env:
                try:
                    self.eval_env.close()
                except Exception:
                    pass
                self.eval_env = None

    def _find_latest_checkpoint(self):
        """Find the most recent checkpoint file."""
        if not os.path.exists(self._checkpoint_path):
            return None
        checkpoints = []
        for f in os.listdir(self._checkpoint_path):
            if f.startswith("mario_ppo_") and f.endswith(".zip") and "final" not in f and "best" not in f and "snapshot" not in f:
                try:
                    steps = int(f.replace("mario_ppo_", "").replace(".zip", ""))
                    checkpoints.append((steps, f))
                except ValueError:
                    continue
        if not checkpoints:
            return None
        checkpoints.sort(reverse=True)
        return os.path.join(self._checkpoint_path, checkpoints[0][1].replace(".zip", ""))

    def save_snapshot(self, name=None):
        """Save a named snapshot of the current model."""
        if self.model is None:
            return None
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        safe_name = (name or "snapshot").replace(" ", "_")[:40]
        filename = f"mario_ppo_{safe_name}_{timestamp}"
        path = os.path.join(self._checkpoint_path, filename)
        os.makedirs(self._checkpoint_path, exist_ok=True)
        self.model.save(path)
        meta = {
            "name": safe_name,
            "timestamp": time.time(),
            "type": "snapshot",
            "world": self.current_params.get("world", 1),
            "stage": self.current_params.get("stage", 1),
            "movement_type": self.current_params.get("movement_type", "right_only"),
        }
        with open(path + ".meta.json", "w") as f:
            _json.dump(meta, f, indent=2)
        return filename

    def get_recordings(self):
        """Get list of available recordings with is_best flag."""
        if self.recorder is None:
            return []
        try:
            recordings = list(self.recorder.recordings)
            best_ep = self.recorder._best_recording["episode"] if self.recorder._best_recording else None
            return [
                {
                    "episode": int(r["episode"]),
                    "x_pos": int(r["x_pos"]),
                    "reward": round(float(r["reward"]), 2),
                    "frame_count": len(r["frames"]),
                    "is_best": r["episode"] == best_ep,
                }
                for r in recordings
            ]
        except Exception:
            return []

    def get_recording_frames(self, episode):
        """Get frames for a specific recording."""
        if self.recorder is None:
            return None
        for r in self.recorder.recordings:
            if r["episode"] == episode:
                return r["frames"]
        return None

    def get_checkpoints(self):
        """List saved checkpoints with metadata."""
        if not os.path.exists(self._checkpoint_path):
            return []
        checkpoints = []
        for f in os.listdir(self._checkpoint_path):
            if not f.endswith(".zip"):
                continue
            path = os.path.join(self._checkpoint_path, f)
            name = f.replace(".zip", "")
            size_mb = os.path.getsize(path) / (1024 * 1024)
            entry = {
                "name": name,
                "size_mb": round(size_mb, 2),
                "modified": os.path.getmtime(path),
            }
            meta_path = path.replace(".zip", ".meta.json")
            if os.path.exists(meta_path):
                try:
                    with open(meta_path) as mf:
                        meta = _json.load(mf)
                    entry.update(meta)
                except Exception:
                    pass
            entry["is_best"] = name.startswith("mario_ppo_best")
            checkpoints.append(entry)
        checkpoints.sort(key=lambda x: x["modified"], reverse=True)
        return checkpoints

    # --- Live Play ---

    def start_play(self, checkpoint_path, world=1, stage=1, movement_type="right_only",
                   skip_frames=4, frame_stack=4, on_frame=None, on_end=None, on_log=None):
        """Load a checkpoint and start playing in a background thread."""
        if self.is_playing:
            return False, "Already playing"

        # Validate checkpoint exists (PPO.load handles .zip extension automatically)
        cp = checkpoint_path
        if not os.path.exists(cp) and not os.path.exists(cp + ".zip"):
            return False, f"Checkpoint not found: {checkpoint_path}"

        self.play_stop_event.clear()
        self._on_play_frame = on_frame
        self._on_play_end = on_end
        self.is_playing = True

        self.play_thread = threading.Thread(
            target=self._play_loop,
            args=(cp, world, stage, movement_type, skip_frames, frame_stack, on_log),
            daemon=True,
        )
        self.play_thread.start()
        return True, "Play started"

    def stop_play(self):
        """Stop the live play session."""
        if not self.is_playing:
            return False
        self.play_stop_event.set()
        return True

    def _play_loop(self, checkpoint_path, world, stage, movement_type,
                   skip_frames, frame_stack, on_log):
        """Inference loop — runs in background thread."""
        import base64
        import io
        from PIL import Image

        env = None
        try:
            if on_log:
                on_log(f"Live play: loading {os.path.basename(checkpoint_path)} on W{world}-{stage}")

            # Create env (same pipeline as eval, with recorder for frame capture)
            env, _ = make_mario_env(
                world=world, stage=stage, movement_type=movement_type,
                skip_frames=skip_frames, frame_stack=frame_stack,
                progress_weight=1.0, time_penalty=0.0, death_penalty=15.0,
                flag_bonus=15.0, reward_clip=15.0,
                record_interval=999999,  # don't auto-record
                env_index=0,
            )
            self.play_env = env

            # Load model on CPU to avoid GPU conflicts with training
            model = PPO.load(checkpoint_path, device="cpu")

            if on_log:
                on_log(f"Live play: model loaded, starting episode")

            obs, info = env.reset()
            done = False
            step_count = 0
            total_reward = 0.0
            max_steps = 5000  # safety timeout

            while not done and not self.play_stop_event.is_set() and step_count < max_steps:
                action, _ = model.predict(obs, deterministic=True)
                action = int(action)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += float(reward)
                step_count += 1

                # Capture and stream raw frame
                raw_frame = get_raw_frame_from_env(env)
                if raw_frame is not None and self._on_play_frame:
                    img = Image.fromarray(raw_frame)
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=75)
                    b64 = base64.b64encode(buf.getvalue()).decode("ascii")

                    self._on_play_frame({
                        "type": "play_frame",
                        "frame": b64,
                        "x_pos": int(info.get("x_pos", 0)),
                        "reward": round(total_reward, 2),
                        "step": step_count,
                        "time": int(info.get("time", 0)),
                        "flag_get": bool(info.get("flag_get", False)),
                    })

                # Throttle to ~15 FPS
                time.sleep(1.0 / 15)

            # Send end message
            result = {
                "type": "play_end",
                "x_pos": int(info.get("x_pos", 0)),
                "reward": round(total_reward, 2),
                "flag_get": bool(info.get("flag_get", False)),
                "steps": step_count,
                "stopped": self.play_stop_event.is_set(),
            }
            if self._on_play_end:
                self._on_play_end(result)

            if on_log:
                status = "CLEAR!" if info.get("flag_get") else "GAME OVER"
                on_log(f"Live play: {status} — x={info.get('x_pos', 0)}, reward={total_reward:.1f}, steps={step_count}")

        except Exception as e:
            if on_log:
                on_log(f"Live play error: {str(e)}")
            if self._on_play_end:
                self._on_play_end({"type": "play_error", "message": str(e)})
            import traceback
            traceback.print_exc()
        finally:
            self.is_playing = False
            if env:
                try:
                    env.close()
                except Exception:
                    pass
            self.play_env = None
