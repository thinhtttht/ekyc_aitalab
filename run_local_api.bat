@echo off
title eKYC Biometric API Server
color 0A
echo ======================================================================
echo           EKYC REST API SERVER - LOCAL WI-FI NETWORK
echo ======================================================================
echo.
echo Server dang khoi dong tren tat ca cac card mang (0.0.0.0:8000)...
echo.
echo [1] Truy cap tu chinh may ban:
echo     - Swagger Docs: http://localhost:8000/docs
echo     - API Base:     http://localhost:8000
echo.
echo [2] Cho may khac trong mang Wi-Fi goi API:
echo     - Swagger Docs: http://10.3.68.193:8000/docs
echo     - Dang ky mat:  POST http://10.3.68.193:8000/api/v1/register
echo     - Xac thuc mat: POST http://10.3.68.193:8000/api/v1/verify
echo     - So sanh mat:  POST http://10.3.68.193:8000/api/v1/compare
echo.
echo Nhan Ctrl + C de dung server.
echo ======================================================================
echo.

cd /d "%~dp0hf_space"
"..\.venv\Scripts\python.exe" -m uvicorn app:app --host 0.0.0.0 --port 8000
pause
