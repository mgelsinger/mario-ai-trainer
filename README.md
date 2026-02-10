# Mario AI Trainer

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Train an AI to play Super Mario Bros using reinforcement learning (PPO), with a real-time browser dashboard to monitor training, watch live play, and progress through all 32 levels.

![Dashboard](screenshots/dashboard.png)

## Features

- **Real-time dashboard** — watch training metrics, charts, and replays update live in your browser
- **Live Play** — select any saved checkpoint and watch the AI play Mario in real-time
- **Level Progression** — train on World 1-1, then transfer the model to 1-2, 1-3, and beyond
- **Checkpoint management** — auto-saves best models per level, manual snapshots, resume anytime
- **Configurable everything** — 20+ hyperparameters adjustable from the UI
- **No build step** — single HTML file frontend, just run the server and go

![Live Play](screenshots/live-play.png)

## Requirements

- **Windows 10/11** (tested on Windows; Linux/Mac may work with adjustments)
- **Python 3.9+**
- **NVIDIA GPU with CUDA** (recommended, CPU works but is much slower)
- **Visual C++ Build Tools** (required for nes-py compilation)

## Quick Start

### 1. Install Visual C++ Build Tools (if not already installed)

nes-py requires C++ compilation. Download from:
https://visualstudio.microsoft.com/visual-cpp-build-tools/

Select **"Desktop development with C++"** during installation.

### 2. Setup

**Option A: Double-click setup**
```
setup.bat
```

**Option B: PowerShell**
```powershell
.\setup.ps1
```

**Option C: Manual**
```bash
python -m venv venv
venv\Scripts\activate

# With NVIDIA GPU:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# CPU only:
pip install torch torchvision

pip install -r requirements.txt
```

### 3. Run

```bash
venv\Scripts\activate
python server.py
```

Open **http://localhost:8000** in your browser.

### 4. Train

1. Check the **System Health** panel — make sure dependencies are green
2. Adjust hyperparameters if desired (defaults are good for starting out)
3. Click **START NEW**
4. Watch the charts update in real-time
5. Check replays as they become available

### 5. Watch the AI Play

1. After training (or with a saved checkpoint), open the **LIVE PLAY** panel
2. Select a checkpoint from the dropdown
3. Choose a world/stage (auto-detected from checkpoint metadata)
4. Click **PLAY** to watch the model run the level in real-time
5. Click **RUN AGAIN** to watch another attempt

### 6. Progress to the Next Level

1. Train until the eval flag rate reaches 80%+ (a banner will appear)
2. Click **TRAIN ON NEXT LEVEL** or manually set the world/stage
3. Load your best checkpoint from the previous level
4. Start training — the model transfers its learned skills to the new level
5. Repeat until all 32 levels are beaten

## What to Expect at Each Stage

Default settings: 8 parallel envs, gamma=0.9, linear LR annealing, reward clipping at 15.

### 10 minutes (~100K steps)
- Mario learns to move right consistently
- Still mostly dying to early obstacles
- Average reward near zero or slightly positive
- X position: ~300-600

### 30 minutes (~500K steps)
- Mario navigates past the first few obstacles
- Starting to learn basic jumping patterns
- Average reward trending upward
- X position: ~600-1200

### 1 hour (~1M steps)
- Mario reliably reaches mid-level
- Learning enemy avoidance and gap jumping
- Reward trend clearly upward
- X position: ~1200-2000

### 2-3 hours (~2-3M steps)
- Mario reaches the flagpole on some runs
- Developing consistent strategies
- X position: ~2000-3000+

### 4-6 hours (~5M steps)
- Mario completes the level regularly
- Consistent high x_positions
- Reward curve plateauing near maximum

### 8+ hours (~10M steps)
- Mario completes the level most of the time
- Optimized pathing and timing
- Near-maximum reward per episode

**Note:** Results vary based on GPU speed, hyperparameters, and randomness. The above are rough guidelines for World 1-1 with default settings. The best model is auto-saved whenever average reward improves.

## Troubleshooting

### "No NVIDIA GPU detected"
- Install the latest NVIDIA drivers from https://www.nvidia.com/drivers
- Training will work on CPU but will be 5-10x slower

### "CUDA not available" (but GPU is present)
- PyTorch was installed without CUDA support. Reinstall:
  ```
  pip uninstall torch torchvision
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
  ```

### nes-py fails to install
- You need Visual C++ Build Tools. See step 1 above.
- Make sure you selected "Desktop development with C++"
- Restart your terminal after installing build tools

### Out of memory (CUDA OOM)
- Reduce `batch_size` (try 32)
- Reduce `n_envs` (try 2)
- Reduce `n_steps` (try 256)
- Close other GPU applications (games, other ML training)

### Training is very slow
- Check that CUDA is being used (device badge should show "CUDA")
- Reduce `n_envs` if CPU-bound
- Increase `skip_frames` to 6 (fewer decisions per episode)

### Blank dashboard / page won't load
- Check browser console (F12) for JavaScript errors
- Try a hard refresh (Ctrl+Shift+R)
- Make sure the server is running and accessible at localhost:8000

### WebSocket disconnects frequently
- The dashboard auto-reconnects. Brief disconnects are normal.
- If persistent, check that no firewall is blocking WebSocket connections.

## Hyperparameter Tuning Guide

### Default settings (recommended for World 1-1)
These defaults are based on the most successful public Mario PPO implementations:
- `learning_rate`: 1e-4 with linear annealing
- `gamma`: 0.9 (lower than typical RL — Mario is reactive, not strategic)
- `n_epochs`: 10
- `ent_coef`: 0.01
- `n_envs`: 8
- `reward_clip`: 15 (prevents gradient instability from extreme rewards)

### For faster initial learning
- Increase `learning_rate` to 2.5e-4
- Increase `ent_coef` to 0.02 (more exploration)
- Use `right_only` movement (fewer actions to learn)

### For better final performance
- Lower `learning_rate` to 7e-5
- Increase `n_steps` to 1024-2048
- Increase `total_timesteps` to 10M

### For speedrunner Mario
- Add a small `time_penalty` of 0.05-0.1
- Increase `progress_weight` to 2.0
- Decrease `death_penalty` to 5

### For harder levels (World 2+)
- Use `simple` or `complex` movement
- Increase `total_timesteps` to 10M
- Increase `ent_coef` to 0.02 (more exploration needed)
- Lower `learning_rate` to 7e-5

## Project Structure

```
mario-ai-trainer/
  server.py           # FastAPI server (REST + WebSocket)
  trainer.py          # PPO training pipeline + live play
  env_wrappers.py     # Mario environment wrappers
  requirements.txt    # Python dependencies
  setup.bat           # Windows setup script (batch)
  setup.ps1           # Windows setup script (PowerShell)
  static/
    index.html        # Dashboard (React 18 + Recharts, no build step)
  screenshots/        # README images
  checkpoints/        # Saved model checkpoints (auto-created)
```

## Tech Stack

- **RL Algorithm**: PPO (Proximal Policy Optimization) via Stable-Baselines3
- **Game Environment**: gym-super-mario-bros + nes-py
- **Backend**: FastAPI + uvicorn
- **Frontend**: React 18 + Recharts (plain JS, no build step)
- **ML Framework**: PyTorch

## License

[MIT](LICENSE)
