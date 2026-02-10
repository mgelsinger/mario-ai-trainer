Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "   Mario AI Trainer - Setup Script (PowerShell)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Check Python
try {
    $pyVer = python --version 2>&1
    Write-Host "[OK] $pyVer" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Python is not installed or not in PATH." -ForegroundColor Red
    Write-Host "Download from https://www.python.org/downloads/"
    exit 1
}

# Create virtual environment
if (-not (Test-Path "venv")) {
    Write-Host "`nCreating virtual environment..."
    python -m venv venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Failed to create virtual environment." -ForegroundColor Red
        exit 1
    }
    Write-Host "[OK] Virtual environment created" -ForegroundColor Green
} else {
    Write-Host "[OK] Virtual environment already exists" -ForegroundColor Green
}

# Activate venv
& ".\venv\Scripts\Activate.ps1"

# Upgrade pip
Write-Host "`nUpgrading pip..."
python -m pip install --upgrade pip 2>&1 | Out-Null

# Check for NVIDIA GPU
Write-Host "`nChecking for NVIDIA GPU..."
$hasGpu = $false
try {
    nvidia-smi 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) { $hasGpu = $true }
} catch {}

if ($hasGpu) {
    Write-Host "[OK] NVIDIA GPU detected" -ForegroundColor Green
    Write-Host "Installing PyTorch with CUDA support..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[WARN] CUDA install failed, falling back to CPU" -ForegroundColor Yellow
        pip install torch torchvision
    }
} else {
    Write-Host "[WARN] No NVIDIA GPU detected - installing CPU-only PyTorch" -ForegroundColor Yellow
    pip install torch torchvision
}

# Install remaining dependencies
Write-Host "`nInstalling dependencies..."
pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "`n[ERROR] Some packages failed to install." -ForegroundColor Red
    Write-Host ""
    Write-Host "If nes-py failed, you need Visual C++ Build Tools:" -ForegroundColor Yellow
    Write-Host "  1. Download from https://visualstudio.microsoft.com/visual-cpp-build-tools/"
    Write-Host "  2. Install 'Desktop development with C++'"
    Write-Host "  3. Re-run this script"
    exit 1
}

Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "   Setup Complete!" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# Run diagnostics
python -c "import torch; print(f'[OK] PyTorch {torch.__version__}'); print(f'[{chr(79)+chr(75) if torch.cuda.is_available() else chr(87)+chr(65)+chr(82)+chr(78)}] CUDA: {`"available`" if torch.cuda.is_available() else `"not available (CPU mode)`"}')"
python -c "import gym_super_mario_bros; print('[OK] gym-super-mario-bros')"
python -c "import stable_baselines3; print(f'[OK] stable-baselines3 {stable_baselines3.__version__}')"
python -c "import cv2; print(f'[OK] OpenCV {cv2.__version__}')"

Write-Host "`nTo start the application:"
Write-Host "  1. Activate the venv:  .\venv\Scripts\Activate.ps1"
Write-Host "  2. Run the server:     python server.py"
Write-Host "  3. Open browser:       http://localhost:8000"
Write-Host ""
