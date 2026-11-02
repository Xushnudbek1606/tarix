@echo off
cd /d "%~dp0.."
call .venv\Scripts\activate.bat
python scripts\run_server.py
pause
