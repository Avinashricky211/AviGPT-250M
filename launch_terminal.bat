@echo off
chcp 65001 >nul 2>nul
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
title AviGPT-250M Terminal Reasoning Engine
echo ======================================================================
echo    AviGPT-250M: High-Efficiency Flagship SLM Terminal
echo    Architect ^& Creator: Yadlapalli Avinash Ricky
echo ======================================================================
echo.

REM Auto-download weights from Hugging Face if not present
if not exist "checkpoints\avigpt_250m_instruct.pt" (
    echo [*] Checkpoint not detected locally. Downloading from Hugging Face...
    py -3.12 download_weights.py 2>nul || python download_weights.py
)

if not exist "avigpt_ssd_memory.db" (
    echo [*] NVMe Memory Bus database not detected. Downloading from Hugging Face...
    py -3.12 download_weights.py 2>nul || python download_weights.py
)

echo.
where py >nul 2>nul
if %errorlevel% equ 0 (
    py -3.12 terminal_eval.py --interactive
    if %errorlevel% neq 0 (
        python terminal_eval.py --interactive
    )
) else (
    python terminal_eval.py --interactive
)

pause
