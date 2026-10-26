@echo off
cd /d "%~dp0"
py -3 companiona.py
if errorlevel 1 python companiona.py
