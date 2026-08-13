@echo off
setlocal

cd /d "%~dp0"

echo Airacare Animal Detector - Phone Camera Tracking
echo.
echo Example phone camera URL:
echo http://192.168.1.10:8080/video
echo.
set /p CAMERA_URL=Enter your phone camera video URL: 

if "%CAMERA_URL%"=="" (
    echo.
    echo Error: No camera URL entered.
    echo Start the phone IP camera app, copy the /video URL, and try again.
    pause
    exit /b 1
)

echo.
echo Starting low-latency tracking...
echo Camera URL: %CAMERA_URL%
echo Confidence: 0.50
echo Image size: 640
echo.
echo Press Q in the camera window to quit.
echo.

venv\Scripts\python.exe main.py --camera-source "%CAMERA_URL%" --confidence 0.50 --imgsz 640 --iou 0.45 --tracking --tracker bytetrack.yaml

echo.
echo Detector closed.
pause



