@echo off
echo ========================================================
echo   Auto-Syncing SleepTwin AI Project to GitHub...
echo ========================================================
cd /d "%~dp0"

:: Check if there are any modified or untracked files
git status --porcelain > "%temp%\git_status_check.txt"
set /p GIT_CHANGES=<"%temp%\git_status_check.txt"
del "%temp%\git_status_check.txt" 2>nul

if "%GIT_CHANGES%"=="" (
    echo [INFO] No new changes detected. Your GitHub repository is already up to date!
    echo.
    timeout /t 3 >nul
    exit /b 0
)

echo [1/3] Staging all modified and new files...
git add .

:: Set custom commit message or default timestamped message
set COMMIT_MSG=%~1
if "%COMMIT_MSG%"=="" (
    set COMMIT_MSG=Auto-update: %date% %time%
)

echo [2/3] Committing changes: "%COMMIT_MSG%"...
git commit -m "%COMMIT_MSG%"

echo [3/3] Pushing to GitHub (origin main)...
git push origin main

echo.
echo ========================================================
echo   [SUCCESS] Successfully pushed all updates to GitHub!
echo ========================================================
timeout /t 3 >nul
