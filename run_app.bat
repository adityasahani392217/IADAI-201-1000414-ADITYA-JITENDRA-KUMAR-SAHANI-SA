@echo off
title StockSense Pro - Retail Shelf Intelligence
echo ========================================================
echo  Starting StockSense Pro Streamlit Application...
echo  AI-Powered Retail Shelf Intelligence & Product Detection
echo ========================================================
cd /d "%~dp0"

if exist ".venv\Scripts\streamlit.exe" (
    echo Using local virtual environment: .venv
    ".venv\Scripts\streamlit.exe" run app.py
) else if exist ".venv\Scripts\python.exe" (
    echo Using Python module runner: .venv
    ".venv\Scripts\python.exe" -m streamlit run app.py
) else (
    echo Using system Python runner...
    python -m streamlit run app.py
)
pause
