@echo off
title Smart-eKYC Biometric System
echo ===================================================
echo       KHOI DONG HE THONG SMART-eKYC
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

echo [1/2] Dang chuan bi mo trinh duyet tai http://127.0.0.1:8000 ...
:: Tu dong mo trinh duyet sau 2 giay de doi uvicorn khoi dong xong
start "" cmd /c "timeout /t 2 /nobreak >nul & start http://127.0.0.1:8000"

echo [2/2] Dang khoi chay FastAPI Server (Backend + Web)...
echo.
echo ===================================================
echo   He thong dang chay tai: http://127.0.0.1:8000
echo   De DUNG server, hay nhan to hop phim: Ctrl + C
echo ===================================================
echo.

python -m uvicorn app:app --app-dir backend --host 127.0.0.1 --port 8000 --reload

pause
