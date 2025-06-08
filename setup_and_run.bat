@echo off
REM Yolo PaddleOCR Setup and Run Script

REM 1. Check for Python
python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed or not in PATH. Please install Python 3.10 or 3.11 and try again.
    pause
    exit /b 1
)

REM 2. Create virtual environment if not exists
if not exist .venv (
    echo Creating virtual environment...
    python -m venv .venv
)

REM 3. Activate virtual environment
call .venv\Scripts\activate.bat

REM 4. Install uv if not present
pip show uv >nul 2>&1
if errorlevel 1 (
    echo Installing uv package manager...
    pip install uv
)

REM 5. Install all dependencies
if exist requirements.lock.txt (
    echo Installing dependencies from requirements.lock.txt...
    uv pip install -r requirements.lock.txt
) else if exist requirements.txt (
    echo Installing dependencies from requirements.txt...
    uv pip install -r requirements.txt
) else (
    echo No requirements file found!
    pause
    exit /b 1
)

REM 6. Reactivate environment (for new shells)
call .venv\Scripts\activate.bat

REM 7. Run the application
python src\main.py

pause
