@echo off
REM Set up the virtual environment
python -m venv env
REM Activate the virtual environment
call env\Scripts\activate
REM Install dependencies
pip install -r requirements.txt
