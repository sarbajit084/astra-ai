@echo off
cd /d "D:\rag_agent"
title RAG Assistant

netstat -ano | findstr :8000 | findstr LISTENING >nul
if %errorlevel% neq 0 (
    start /min "RAG Server" python app.py
    timeout /t 2 /nobreak >nul
)

if exist "C:\Program Files\Google\Chrome\Application\chrome.exe" (
    start "" "C:\Program Files\Google\Chrome\Application\chrome.exe" --app="http://127.0.0.1:8000" --window-size=1300,880
    exit /b
)

if exist "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe" (
    start "" "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe" --app="http://127.0.0.1:8000" --window-size=1300,880
    exit /b
)

start "" "http://127.0.0.1:8000"
exit /b
