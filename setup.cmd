@echo off
call "%~dp0run_analysis.cmd" --setup-only %*
exit /b %errorlevel%
