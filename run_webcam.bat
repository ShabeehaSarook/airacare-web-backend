@echo off
setlocal

cd /d "%~dp0"

echo Airacare Animal Detector - Laptop Webcam
echo.
echo Starting tracking with default webcam...
echo Press Q in the camera window to quit.
echo.

venv\Scripts\python.exe main.py --camera-source 0 --confidence 0.50 --imgsz 640 --iou 0.45 --tracking --tracker bytetrack.yaml

echo.
echo Detector closed.
pause



