@echo off
title VisionRAG Launcher
echo ================================================
echo  VisionRAG - Multimodal RAG with Gemini Vision
echo ================================================
echo.
echo Make sure the following are running before continuing:
echo   [1] Neo4j Community  ^(http://localhost:7474^)
echo   [2] Ollama           ^(ollama serve^)
echo.
pause

echo.
echo Starting backend on http://localhost:8081 ...
start "VisionRAG Backend" cmd /k "cd /d %~dp0backend && python -m uvicorn app.main:app --port 8081 --reload"

timeout /t 4 /nobreak >nul

echo Starting frontend on http://localhost:3001 ...
start "VisionRAG Frontend" cmd /k "cd /d %~dp0frontend && npm run dev -- --port 3001"

echo.
echo Both windows are opening.
echo   Backend API : http://localhost:8081
echo   Frontend    : http://localhost:3001
echo   Health      : http://localhost:8081/health
echo   Neo4j UI    : http://localhost:7474
echo.
pause
