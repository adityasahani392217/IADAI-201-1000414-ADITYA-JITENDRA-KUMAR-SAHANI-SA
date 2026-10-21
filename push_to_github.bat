@echo off
title StockSense Pro - GitHub Synchronization
echo ========================================================
echo  StockSense Pro - Pushing to Assignment GitHub Repository
echo  CRS Summative Assessment: Machine Learning & Deep Learning (FA 2)
echo ========================================================
git add -A
git commit -m "feat: complete StockSense Pro retail shelf intelligence system with fine-tuned YOLOv8, inventory rules, testing, and reports"
git push -u origin main
if %errorlevel% neq 0 (
    echo.
    echo If prompted by Git Credential Manager, click 'Sign in with your browser'.
    echo.
) else (
    echo.
    echo ========================================================
    echo  SUCCESS! StockSense Pro successfully pushed to:
    echo  https://github.com/adityasahani392217/IADAI-201-1000414-ADITYA-JITENDRA-KUMAR-SAHANI-FA
    echo ========================================================
)
pause
