@echo off
cd /d "%~dp0.."
call .venv\Scripts\activate.bat
python -m training.extract_landmarks
python -m training.train --epochs 80
pause
