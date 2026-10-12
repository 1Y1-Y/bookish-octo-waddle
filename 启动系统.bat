@echo off
title Tobacco Disease Diagnosis System
cd /d "%~dp0"

echo ============================================
echo   Tobacco Disease Diagnosis System
echo   Starting...
echo ============================================
echo.

python -c "import torch" 2>nul
if errorlevel 1 (
    echo [ERROR] Python or PyTorch not found. Please set up the environment first.
    pause
    exit /b
)

echo Starting, browser will open automatically...
echo If not, open this URL manually:  http://127.0.0.1:7860
echo.
echo Close this window to exit.
echo.

python web_demo.py

pause
