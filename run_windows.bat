@echo off
cls
echo =====================================================================
echo   Vietnamese Fake News Detection System - Docker Runner (Windows)
echo =====================================================================
echo.

echo [1/2] Building Docker image 'viet-fake-news-detector'...
docker build -t viet-fake-news-detector .
if errorlevel 1 (
    echo.
    echo [ERROR] Docker build failed. Please check if Docker Desktop is running.
    pause
    exit /b 1
)
echo [SUCCESS] Docker image built successfully.
echo.

echo [2/2] Launching container in interactive mode...
echo.
echo Mounting local persistent volumes:
echo   - Checkpoints:   %CD%\src\checkpoints
echo   - KnowledgeBase: %CD%\KnowledgeBase
echo   - Cache:         %CD%\src\CACHES
echo.

docker run -it --gpus all ^
  -v "%CD%/src/checkpoints:/app/src/checkpoints" ^
  -v "%CD%/KnowledgeBase:/app/KnowledgeBase" ^
  -v "%CD%/src/CACHES:/app/src/CACHES" ^
  viet-fake-news-detector --interactive

if %ERRORLEVEL% EQU 125 goto run_cpu
goto end

:run_cpu
echo.
echo [WARNING] GPU acceleration is not supported or configured in Docker.
echo Retrying in CPU mode...
echo.
docker run -it ^
  -v "%CD%/src/checkpoints:/app/src/checkpoints" ^
  -v "%CD%/KnowledgeBase:/app/KnowledgeBase" ^
  -v "%CD%/src/CACHES:/app/src/CACHES" ^
  viet-fake-news-detector --interactive

:end
echo.
echo Container session closed.
pause
