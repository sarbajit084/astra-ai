@echo off
cd /d "%~dp0"
title Local RAG Agent - ChatGPT UI
echo ========================================================
echo Starting Local RAG Agent (ChatGPT-Style UI)
echo Port: 8000
echo URL:  http://127.0.0.1:8000
echo ========================================================
start "" http://127.0.0.1:8000
python app.py
pause
