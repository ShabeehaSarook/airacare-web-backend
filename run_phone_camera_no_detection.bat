@echo off
setlocal

cd /d "%~dp0"

echo Airacare Animal Detector - Phone Camera Stream Test
echo.
echo This mode shows phone video only. YOLO detection is disabled.
echo Use it to check whether the phone/Wi-Fi stream itself is delayed.
echo.
set /p CAMERA_URL=Enter your phone camera video URL: 

if "%CAMERA_URL%"=="" (
    echo.
    echo Error: No camera URL entered.
    pause
    exit /b 1
)

echo.
echo Starting phone stream test...
echo Press Q in the camera window to quit.
echo.

venv\Scripts\python.exe main.py --camera-source "%CAMERA_URL%" --no-detection

echo.
echo Stream test closed.
pause
