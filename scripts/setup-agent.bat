@echo off
REM Navigate to the backend directory
cd /d "%~dp0\..\backend" || exit /b 1

REM Create virtual environment if it doesn't exist
if not exist ".venv" (
    python -m venv .venv
)

REM Activate the virtual environment
call .venv\Scripts\activate.bat

REM Install requirements using pip
pip install -r requirements.txt
