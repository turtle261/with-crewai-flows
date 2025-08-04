@echo off
REM Navigate to the backend directory
cd /d %~dp0\..\backend

REM Activate the virtual environment
call .venv\Scripts\activate.bat

REM Run the backend server
.venv\Scripts\python.exe server.py
