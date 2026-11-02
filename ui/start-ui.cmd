@echo off
setlocal
cd /d "%~dp0.."
echo MouseGPT UI: http://127.0.0.1:8878/ui/
echo Keep this window open. Press Ctrl+C to stop the preview.
if exist "%LOCALAPPDATA%\Programs\Python\Python313\python.exe" (
  "%LOCALAPPDATA%\Programs\Python\Python313\python.exe" ui\server.py --port 8878
) else (
  python ui\server.py --port 8878
)
if errorlevel 1 pause
