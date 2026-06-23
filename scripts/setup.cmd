@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0\.."

where py >nul 2>nul
if %errorlevel%==0 (
  py -3 run.py deps
  if %errorlevel% neq 0 exit /b %errorlevel%
  echo Setup complete. Run: py -3 run.py all
  exit /b 0
)

where python >nul 2>nul
if %errorlevel%==0 (
  python run.py deps
  if %errorlevel% neq 0 exit /b %errorlevel%
  echo Setup complete. Run: python run.py all
  exit /b 0
)

echo Python 3 was not found in PATH. Install Python 3.10+ first.
exit /b 1
