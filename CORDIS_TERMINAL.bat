@echo off
title Cordis-OxCaml Dual Inference Terminal
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0terminal.ps1"
pause
