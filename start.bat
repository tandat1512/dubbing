@echo off
echo ========================================
echo   AI Learning Platform - Starting...
echo ========================================

:: Start Python Backend Server
echo Starting Python Backend Server...
start "Backend Server" cmd /k "cd /d x:\youtube\learning-platform && python server.py"

:: Wait a moment for backend to initialize
timeout /t 3 /nobreak > nul

:: Start React Frontend
echo Starting React Frontend...
start "Frontend Dev Server" cmd /k "cd /d x:\youtube\learning-platform\react-app && npm run dev"

echo ========================================
echo   Servers are starting...
echo   Backend: http://localhost:3000
echo   Frontend: http://localhost:5173
echo ========================================

:: Open browser after a short delay
timeout /t 5 /nobreak > nul
start http://localhost:5173

echo Done! You can close this window.
pause
