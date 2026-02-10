"""
FastAPI server for Mario AI Trainer.
REST endpoints + WebSocket for real-time metrics and replay streaming.
"""

import asyncio
import base64
import io
import json
import os
import subprocess
import time
from typing import Optional

import psutil
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from PIL import Image

from trainer import MarioTrainer

app = FastAPI(title="Mario AI Trainer")
trainer = MarioTrainer()

# WebSocket connections
ws_connections: list[WebSocket] = []
log_messages: list[str] = []
MAX_LOG_MESSAGES = 500
_event_loop: Optional[asyncio.AbstractEventLoop] = None


def add_log(message: str):
    """Add a log message and broadcast to WebSocket clients."""
    timestamp = time.strftime("%H:%M:%S")
    entry = f"[{timestamp}] {message}"
    log_messages.append(entry)
    if len(log_messages) > MAX_LOG_MESSAGES:
        log_messages.pop(0)
    broadcast({"type": "log", "message": entry})


def on_metrics(metrics: dict):
    """Called from training thread with episode metrics."""
    broadcast({"type": "metrics", "data": metrics})


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        import numpy as np
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def broadcast(data: dict):
    """Send data to all connected WebSocket clients (thread-safe)."""
    if not _event_loop or not ws_connections:
        return
    msg = json.dumps(data, cls=_NumpyEncoder)

    async def _send_all():
        dead = []
        for ws in ws_connections[:]:
            try:
                await ws.send_text(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            if ws in ws_connections:
                ws_connections.remove(ws)

    try:
        asyncio.run_coroutine_threadsafe(_send_all(), _event_loop)
    except Exception:
        pass


trainer.set_callbacks(on_metrics=on_metrics, on_log=add_log)


# --- REST Endpoints ---

@app.get("/")
async def index():
    return FileResponse("static/index.html")


@app.get("/api/system-check")
async def system_check():
    """Check system readiness for training."""
    result = {
        "gpu": {"available": False, "name": "N/A", "vram_total_mb": 0, "vram_free_mb": 0},
        "cuda": {"available": False, "version": "N/A"},
        "cudnn": {"available": False},
        "cpu": {"cores": psutil.cpu_count(logical=False) or 0, "threads": psutil.cpu_count(logical=True) or 0},
        "ram": {"total_gb": round(psutil.virtual_memory().total / (1024**3), 1),
                "available_gb": round(psutil.virtual_memory().available / (1024**3), 1)},
        "dependencies": {},
        "ready": False,
        "warnings": [],
        "device": "cpu",
    }

    # Check GPU via nvidia-smi
    try:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10)
        if smi.returncode == 0:
            # Take first line only (first GPU if multi-GPU)
            first_line = smi.stdout.strip().split("\n")[0]
            parts = first_line.split(",")
            if len(parts) >= 3:
                result["gpu"]["available"] = True
                result["gpu"]["name"] = parts[0].strip()
                try:
                    result["gpu"]["vram_total_mb"] = int(parts[1].strip())
                    result["gpu"]["vram_free_mb"] = int(parts[2].strip())
                except ValueError:
                    pass
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Check CUDA/cuDNN via PyTorch
    try:
        import torch
        result["cuda"]["available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            result["cuda"]["version"] = torch.version.cuda or "N/A"
            result["cudnn"]["available"] = torch.backends.cudnn.is_available()
            result["device"] = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            result["device"] = "mps"
    except ImportError:
        pass

    # Check Python dependencies
    deps = {
        "torch": "torch",
        "gym_super_mario_bros": "gym_super_mario_bros",
        "stable_baselines3": "stable_baselines3",
        "cv2": "cv2",
        "gymnasium": "gymnasium",
        "nes_py": "nes_py",
    }
    all_deps_ok = True
    for name, module in deps.items():
        try:
            __import__(module)
            result["dependencies"][name] = True
        except ImportError:
            result["dependencies"][name] = False
            all_deps_ok = False

    # Warnings
    if not result["gpu"]["available"]:
        result["warnings"].append("No NVIDIA GPU detected. Training will use CPU (much slower).")
    elif result["gpu"]["vram_free_mb"] < 2000:
        result["warnings"].append(f"Low VRAM: {result['gpu']['vram_free_mb']}MB free. Consider closing other GPU apps.")

    if not result["cuda"]["available"] and result["gpu"]["available"]:
        result["warnings"].append("GPU found but CUDA not available in PyTorch. Reinstall PyTorch with CUDA support.")

    if not all_deps_ok:
        missing = [k for k, v in result["dependencies"].items() if not v]
        result["warnings"].append(f"Missing packages: {', '.join(missing)}. Run: pip install -r requirements.txt")

    if result["ram"]["available_gb"] < 4:
        result["warnings"].append(f"Low RAM: {result['ram']['available_gb']}GB available.")

    result["ready"] = all_deps_ok and (not result["warnings"] or
                                        all("Low" in w or "CPU" in w for w in result["warnings"]))

    return result


@app.get("/api/hyperparameters")
async def get_hyperparameters():
    """Return current/default hyperparameters."""
    return {
        "learning_rate": trainer.current_params.get("learning_rate", 1e-4),
        "n_steps": trainer.current_params.get("n_steps", 512),
        "batch_size": trainer.current_params.get("batch_size", 64),
        "n_epochs": trainer.current_params.get("n_epochs", 10),
        "gamma": trainer.current_params.get("gamma", 0.9),
        "gae_lambda": trainer.current_params.get("gae_lambda", 0.95),
        "clip_range": trainer.current_params.get("clip_range", 0.2),
        "ent_coef": trainer.current_params.get("ent_coef", 0.01),
        "vf_coef": trainer.current_params.get("vf_coef", 0.5),
        "max_grad_norm": trainer.current_params.get("max_grad_norm", 0.5),
        "n_envs": trainer.current_params.get("n_envs", 8),
        "world": trainer.current_params.get("world", 1),
        "stage": trainer.current_params.get("stage", 1),
        "movement_type": trainer.current_params.get("movement_type", "right_only"),
        "skip_frames": trainer.current_params.get("skip_frames", 4),
        "frame_stack": trainer.current_params.get("frame_stack", 4),
        "progress_weight": trainer.current_params.get("progress_weight", 1.0),
        "time_penalty": trainer.current_params.get("time_penalty", 0.0),
        "death_penalty": trainer.current_params.get("death_penalty", 15.0),
        "flag_bonus": trainer.current_params.get("flag_bonus", 15.0),
        "reward_clip": trainer.current_params.get("reward_clip", 15.0),
        "total_timesteps": trainer.current_params.get("total_timesteps", 5_000_000),
        "record_interval": trainer.current_params.get("record_interval", 10),
        "lr_annealing": trainer.current_params.get("lr_annealing", "linear"),
    }


@app.post("/api/training/start")
async def start_training(params: dict):
    """Start training with given parameters."""
    if trainer.is_training:
        return {"status": "error", "message": "Training already in progress"}

    checkpoint_path = params.pop("checkpoint_path", None)
    add_log("Starting training...")
    success = trainer.start_training(params, checkpoint_path=checkpoint_path)
    if success:
        return {"status": "ok", "message": "Training started", "device": trainer.device}
    return {"status": "error", "message": "Failed to start training"}


@app.post("/api/training/stop")
async def stop_training():
    """Stop training."""
    if not trainer.is_training:
        return {"status": "error", "message": "No training in progress"}

    add_log("Stopping training...")
    trainer.stop_training()
    return {"status": "ok", "message": "Training stop requested"}


@app.get("/api/training/status")
async def training_status():
    """Get current training status."""
    return {
        "is_training": trainer.is_training,
        "is_playing": trainer.is_playing,
        "device": trainer.device,
    }


@app.get("/api/replays")
async def list_replays():
    """List available replay recordings."""
    return {"replays": trainer.get_recordings()}


@app.get("/api/checkpoints")
async def list_checkpoints():
    """List saved model checkpoints."""
    return {"checkpoints": trainer.get_checkpoints()}


@app.post("/api/snapshots/save")
async def save_snapshot(body: dict):
    """Save a named snapshot of the current model."""
    name = body.get("name", "snapshot")
    filename = trainer.save_snapshot(name)
    if filename:
        add_log(f"Snapshot saved: {filename}")
        return {"status": "ok", "filename": filename}
    return {"status": "error", "message": "No model loaded to snapshot"}


@app.get("/api/logs")
async def get_logs():
    """Get recent log messages."""
    return {"logs": log_messages[-100:]}


@app.post("/api/play/start")
async def start_play(body: dict):
    """Start a live play session with a checkpoint."""
    checkpoint_path = body.get("checkpoint_path")
    if not checkpoint_path:
        return {"status": "error", "message": "checkpoint_path required"}

    # Build full path if just a name
    if not os.path.sep in checkpoint_path and not checkpoint_path.endswith(".zip"):
        checkpoint_path = os.path.join("checkpoints", checkpoint_path)

    world = body.get("world", 1)
    stage = body.get("stage", 1)
    movement_type = body.get("movement_type", "right_only")

    success, msg = trainer.start_play(
        checkpoint_path=checkpoint_path,
        world=world,
        stage=stage,
        movement_type=movement_type,
        on_frame=broadcast,
        on_end=broadcast,
        on_log=add_log,
    )
    if success:
        add_log(f"Live play started: {os.path.basename(checkpoint_path)} on W{world}-{stage}")
        return {"status": "ok"}
    return {"status": "error", "message": msg}


@app.post("/api/play/stop")
async def stop_play():
    """Stop the live play session."""
    if trainer.stop_play():
        add_log("Live play stopped")
        return {"status": "ok"}
    return {"status": "error", "message": "Not playing"}


@app.get("/api/levels/summary")
async def level_summary():
    """Get training progress summary per level."""
    checkpoints = trainer.get_checkpoints()
    levels = {}
    for cp in checkpoints:
        w = cp.get("world")
        s = cp.get("stage")
        if w is None or s is None:
            continue
        key = f"{w}-{s}"
        if key not in levels:
            levels[key] = {
                "world": int(w), "stage": int(s),
                "checkpoints": 0,
                "best_reward": None,
                "max_x_pos": 0,
                "best_checkpoint": None,
                "total_timesteps": 0,
            }
        levels[key]["checkpoints"] += 1
        reward = cp.get("avg_reward")
        if reward is not None:
            if levels[key]["best_reward"] is None or reward > levels[key]["best_reward"]:
                levels[key]["best_reward"] = round(float(reward), 2)
                levels[key]["best_checkpoint"] = cp["name"]
        x_pos = cp.get("max_x_pos", 0)
        if x_pos and int(x_pos) > levels[key]["max_x_pos"]:
            levels[key]["max_x_pos"] = int(x_pos)
        timestep = cp.get("timestep", 0)
        if timestep and int(timestep) > levels[key]["total_timesteps"]:
            levels[key]["total_timesteps"] = int(timestep)

    return {"levels": list(levels.values())}


# --- WebSocket ---

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    global _event_loop
    await ws.accept()
    _event_loop = asyncio.get_event_loop()
    ws_connections.append(ws)

    # Send current state
    await ws.send_text(json.dumps({
        "type": "status",
        "is_training": trainer.is_training,
        "is_playing": trainer.is_playing,
        "device": trainer.device,
    }))

    try:
        while True:
            data = await ws.receive_text()
            msg = json.loads(data)

            if msg.get("type") == "get_replay":
                episode = msg.get("episode")
                frames = trainer.get_recording_frames(episode)
                if frames is None:
                    await ws.send_text(json.dumps({
                        "type": "replay_error",
                        "message": f"Recording for episode {episode} not found",
                    }))
                    continue

                # Send frames in chunks
                await ws.send_text(json.dumps({
                    "type": "replay_start",
                    "episode": episode,
                    "total_frames": len(frames),
                }))

                chunk_size = 10
                for i in range(0, len(frames), chunk_size):
                    chunk_frames = []
                    for frame in frames[i:i + chunk_size]:
                        # Encode frame as base64 JPEG
                        img = Image.fromarray(frame)
                        buf = io.BytesIO()
                        img.save(buf, format="JPEG", quality=80)
                        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
                        chunk_frames.append(b64)

                    await ws.send_text(json.dumps({
                        "type": "replay_frames",
                        "episode": episode,
                        "start_index": i,
                        "frames": chunk_frames,
                    }))

                await ws.send_text(json.dumps({
                    "type": "replay_end",
                    "episode": episode,
                }))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        if ws in ws_connections:
            ws_connections.remove(ws)


if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("  Mario AI Trainer")
    print("  Open http://localhost:8000 in your browser")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000)
