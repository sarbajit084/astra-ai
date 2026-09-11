Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "Starting Local RAG Agent (ChatGPT-Style UI)" -ForegroundColor Green
Write-Host "Port: 8000" -ForegroundColor Yellow
Write-Host "URL:  http://127.0.0.1:8000" -ForegroundColor Yellow
Write-Host "========================================================" -ForegroundColor Cyan

Start-Process "http://127.0.0.1:8000"
python app.py
