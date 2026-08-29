@echo off
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -Command "Start-Process powershell -ArgumentList '-NoProfile -ExecutionPolicy Bypass -Command `$host.UI.RawUI.WindowTitle = ''LlamamlDaemon''; & ''%~dp0server.ps1''' -WindowStyle Hidden"
start "" "http://localhost:8092"
