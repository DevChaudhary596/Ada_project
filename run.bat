@echo off
title MedVision-Agent Launcher
cd /d "%~dp0"

echo ====================================================
echo        Starting MedVision-Agent AI System
echo ====================================================
echo.

:: Add local bin to PATH for portable tools
set "PATH=%USERPROFILE%\.local\bin;%PATH%"

:: 1. If local virtual environment exists, run streamlit directly from it
if exist ".venv\Scripts\streamlit.exe" (
    echo [INFO] Launching MedVision-Agent from local environment...
    ".venv\Scripts\streamlit.exe" run app.py
    goto end
)

:: 2. If uv is available, create environment and run
where uv >nul 2>nul
if %ERRORLEVEL% equ 0 (
    echo [INFO] Setting up virtual environment with uv...
    uv venv
    uv pip install -r requirements.txt
    uv run streamlit run app.py
    goto end
)

:: 3. Fallback to standard python / py launcher
where py >nul 2>nul
if %ERRORLEVEL% equ 0 (
    echo [INFO] Using Python launcher...
    py -m pip install -r requirements.txt
    py -m streamlit run app.py
    goto end
)

where python >nul 2>nul
if %ERRORLEVEL% equ 0 (
    echo [INFO] Using Python...
    python -m pip install -r requirements.txt
    python -m streamlit run app.py
    goto end
)

echo [ERROR] Python environment not detected.
echo Please install Python from https://www.python.org/downloads/
echo and make sure to check "Add python.exe to PATH".
pause

:end
