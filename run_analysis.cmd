@echo off
setlocal EnableDelayedExpansion
rem Delayed expansion keeps characters in a launcher path literal.
set "TRAILCAM_LAUNCH_COMMAND=!CMDCMDLINE!"
setlocal DisableDelayedExpansion
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" bootstrap.py %*
    goto :finished
)
py -3.12 -c "import sys; assert sys.maxsize > 2**32" >nul 2>&1
if not errorlevel 1 (
    py -3.12 bootstrap.py %*
    goto :finished
)
py -3.11 -c "import sys; assert sys.maxsize > 2**32" >nul 2>&1
if not errorlevel 1 (
    py -3.11 bootstrap.py %*
    goto :finished
)
python -c "import sys; assert sys.version_info[:2] in [(3,11),(3,12)] and sys.maxsize > 2**32" >nul 2>&1
if not errorlevel 1 (
    python bootstrap.py %*
    goto :finished
)
echo Install 64-bit Python 3.12, then run this file again.
echo Command: winget install -e --id Python.Python.3.12
echo Download: https://www.python.org/downloads/windows/
exit /b 2
:finished
set "trailcam_exit_code=%errorlevel%"
if not "%trailcam_exit_code%"=="0" echo Analysis stopped with an error. See the messages above.
exit /b %trailcam_exit_code%
