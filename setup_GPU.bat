@ECHO off
cd /D %~dp0
call set_env.bat

echo Setting up python...
python get-pip.py
if errorlevel 1 goto :error
python -m pip install --upgrade pip "setuptools<81" "packaging<25" wheel
if errorlevel 1 goto :error
echo Setting up torch
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
if errorlevel 1 goto :error
python -m pip install torchcodec
if errorlevel 1 goto :error
python -m pip install -r assets/requirements.txt
if errorlevel 1 goto :error

rem model install
python install_assets.py
if errorlevel 1 goto :error

pause
exit /b 0

:error
echo.
echo Setup failed.
pause
exit /b 1