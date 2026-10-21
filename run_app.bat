@echo off
TITLE ParkVision AI - Launch Dashboard
echo ========================================================
echo   ParkVision AI: Smart Parking Intelligence Platform
echo   CRS Artificial Intelligence Summative Assessment
echo   Candidate: Aditya Jitendra Kumar Sahani (1000414)
echo ========================================================
echo.
echo Activating virtual environment and launching Streamlit...
echo.

cd /d "%~dp0"
"..\ML & DL FA 2\.venv\Scripts\python.exe" -m streamlit run app.py

pause
