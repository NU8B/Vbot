@echo off
:: Set the PYTHONPATH to include the current directory
set PYTHONPATH=%~dp0;%PYTHONPATH%

:: Run the animation program
python tha4/app/autonomous_animation.py

:: Pause to show any error messages
pause 