@echo off
title Test Camera 3D & 2D Intel RealSense (pyrealsense2)
echo ========================================================
echo   KHOI DONG TRINH XEM CAMERA 3D & 2D INTEL REALSENSE
echo   Hien thi dong thoi 2D RGB va 3D Ban do do sau
echo ========================================================
echo.

cd /d "%~dp0"

:: Kiem tra moi truong ao
if not exist ".venv\Scripts\activate.bat" (
    echo [LOI] Khong tim thay moi truong .venv!
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat

echo Dang ket noi toi camera Intel RealSense...
python test_realsense_3d_2d.py

pause
