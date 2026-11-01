@echo off
title eKYC Biometric System - Native Desktop GUI (PyQt6)
echo ===================================================
echo   KHOI DONG HE THONG eKYC NATIVE DESKTOP GUI
echo   Ung dung PyQt6 sieu nhe, khong giat lag
echo ===================================================
echo.

cd /d "%~dp0"

:: Kiem tra moi truong Python ao .venv
if not exist ".venv\Scripts\activate.bat" (
    echo [LOI] Khong tim thay thu muc .venv tai %~dp0
    echo Vui long kiem tra lai moi truong Python.
    pause
    exit /b 1
)

:: Kich hoat moi truong ao
call .venv\Scripts\activate.bat

echo Dang khoi chay giao dien Desktop eKYC...
python desktop_app\main.py

pause
