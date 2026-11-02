@echo off
cd /d "%~dp0.."
if not exist .venv python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
if not exist .env copy .env.example .env
python scripts\download_models.py
echo.
echo Setup finished. Edit .env (secrets, database), then run scripts\start_server.bat on the server.
pause
