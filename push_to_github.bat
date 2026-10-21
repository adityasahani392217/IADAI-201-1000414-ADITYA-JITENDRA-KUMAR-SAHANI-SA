@echo off
TITLE ParkVision AI - Push to Summative Assessment GitHub Repository
echo =======================================================================
echo   ParkVision AI: Push to SA GitHub Repository
echo   Target: IADAI-201-1000414-ADITYA-JITENDRA-KUMAR-SAHANI-SA
echo =======================================================================
echo.

cd /d "%~dp0"

echo Checking git status...
git status
echo.

set /p confirm="Do you want to commit and push all changes to SA repository? (Y/N): "
if /i not "%confirm%"=="Y" (
    echo Operation cancelled by user.
    pause
    exit /b
)

echo Adding files to git...
git add -A

echo Committing files...
git commit -m "feat: complete build of ParkVision AI smart parking intelligence platform"

echo Pushing to origin main...
git push origin main

echo.
echo =======================================================================
echo   Sync Complete!
echo =======================================================================
pause
