@echo off
cd /d "%~dp0"
echo Checking Python packages...
python -m pip install -r requirements.txt
if errorlevel 1 (
  echo Package installation failed. Check your internet connection and Python installation.
  pause
  exit /b 1
)
if not exist "models\colorization_release_v2.caffemodel" (
  echo Downloading required AI model. It is about 129 MB and needs internet.
  python download_model.py
  if errorlevel 1 (
    echo Model download failed. See README.md for manual download instructions.
    pause
    exit /b 1
  )
)
python GUI.py
if errorlevel 1 pause
