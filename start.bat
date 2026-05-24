@echo off
title VisionRAG Launcher
echo ================================================
echo  VisionRAG - Multimodal RAG
echo ================================================
echo.
echo Make sure the following are running before continuing:
echo   [1] Neo4j Desktop  ^(start your DBMS at http://localhost:7474^)
echo   [2] Ollama         ^(ollama serve^)
echo.
echo Vision backend: set VISION_BACKEND in backend\.env
echo   ollama    ^(default^) - ollama pull qwen2.5vl:7b
echo   paligemma           - needs HF_TOKEN in .env
echo   gemini              - needs GEMINI_API_KEY in .env
echo   openai              - needs OPENAI_API_KEY in .env
echo.
pause

echo.
echo Starting backend on http://localhost:8081 ...
start "VisionRAG Backend" cmd /k "cd /d %~dp0backend && python -m uvicorn app.main:app --port 8081"

timeout /t 3 /nobreak >nul

echo Starting frontend on http://localhost:3001 ...
start "VisionRAG Frontend" cmd /k "cd /d %~dp0frontend && npm run dev -- --port 3001"

echo.
echo Both windows are opening.
echo   Frontend   : http://localhost:3001
echo   Backend    : http://localhost:8081
echo   Health     : http://localhost:8081/health
echo   Neo4j UI   : http://localhost:7474
echo.
echo NOTE: First ingest/query will be slow while models load.
echo       Subsequent requests will be fast.
echo.
pause
