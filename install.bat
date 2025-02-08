@echo off
echo Installing VBot and dependencies...

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed! Please install Python 3.10 or later.
    pause
    exit /b 1
)

:: Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

:: Install the package in development mode
echo Installing VBot...
pip install -e .

echo.
echo Installation complete! You can now:
echo 1. Run the animation directly using: run-animation
echo 2. Or use the run_animation.bat script
echo.
pause 