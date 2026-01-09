@echo off
title Vigilance AI - Local Test Server
cls
echo ========================================================
echo   🛡️  VIGILANCE AI - LOCAL TEST SERVER
echo ========================================================
echo.
echo Starting local web server...
echo.
echo [INSTRUCTIONS]
echo 1. The server will start below.
echo 2. Open your browser and go to: http://127.0.0.1:8000
echo 3. Keep this window open while testing.
echo.
echo ========================================================
echo.

python -m http.server 8000 --directory web

pause
