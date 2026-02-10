@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo    Mario AI Trainer - Setup Script (Windows)
echo ============================================================
echo.

REM Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Download from https://www.python.org/downloads/
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYVER=%%i
echo [OK] Python %PYVER% found

REM Create virtual environment
if not exist "venv" (
    echo.
    echo Creating virtual environment...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created
) else (
    echo [OK] Virtual environment already exists
)

REM Activate venv
call venv\Scripts\activate.bat

REM Upgrade pip
echo.
echo Upgrading pip...
python -m pip install --upgrade pip >nul 2>&1

REM Install PyTorch with CUDA
echo.
echo Checking for NVIDIA GPU...
nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] NVIDIA GPU detected
    echo Installing PyTorch with CUDA support...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
    if %errorlevel% neq 0 (
        echo [WARN] CUDA PyTorch install failed, falling back to CPU version
        pip install torch torchvision
    )
) else (
    echo [WARN] No NVIDIA GPU detected - installing CPU-only PyTorch
    pip install torch torchvision
)

REM Install remaining dependencies
echo.
echo Installing dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Some packages failed to install.
    echo.
    echo If nes-py failed, you need Visual C++ Build Tools:
    echo   1. Download from https://visualstudio.microsoft.com/visual-cpp-build-tools/
    echo   2. Install "Desktop development with C++"
    echo   3. Re-run this script
    echo.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo    Setup Complete!
echo ============================================================
echo.

REM Run diagnostics
python -c "import torch; print(f'[OK] PyTorch {torch.__version__}'); print(f'[{\"OK\" if torch.cuda.is_available() else \"WARN\"}] CUDA: {\"available\" if torch.cuda.is_available() else \"not available (CPU mode)\"}')" 2>&1
python -c "import gym_super_mario_bros; print('[OK] gym-super-mario-bros')" 2>&1
python -c "import stable_baselines3; print(f'[OK] stable-baselines3 {stable_baselines3.__version__}')" 2>&1
python -c "import cv2; print(f'[OK] OpenCV {cv2.__version__}')" 2>&1

echo.
echo To start the application:
echo   1. Activate the venv:  venv\Scripts\activate
echo   2. Run the server:     python server.py
echo   3. Open browser:       http://localhost:8000
echo.
pause
