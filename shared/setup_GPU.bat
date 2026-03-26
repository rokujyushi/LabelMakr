@ECHO off
SETLOCAL ENABLEEXTENSIONS
cd /D %~dp0

SET ROOT_DIR=%~dp0..
IF "%ROOT_DIR:~-1%"=="\" SET ROOT_DIR=%ROOT_DIR:~0,-1%
SET TORCH_GPU_PACKAGES=torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

call :install_runtime "gui" "%ROOT_DIR%\gui\python\python.exe" "%ROOT_DIR%\gui\requirements.txt" ""
if errorlevel 1 goto :error

call :install_runtime "runtime_a" "%ROOT_DIR%\runtime_a\python\python.exe" "%ROOT_DIR%\runtime_a\requirements.txt" "%TORCH_GPU_PACKAGES%"
if errorlevel 1 goto :error
"%ROOT_DIR%\runtime_a\python\python.exe" -m pip install torchcodec
if errorlevel 1 goto :error

call :install_runtime "runtime_b" "%ROOT_DIR%\runtime_b\python\python.exe" "%ROOT_DIR%\runtime_b\requirements.txt" "%TORCH_GPU_PACKAGES% torchcodec"
if errorlevel 1 goto :error

echo Installing shared assets...
"%ROOT_DIR%\gui\python\python.exe" "%~dp0install_assets.py"
if errorlevel 1 goto :error

echo.
echo Setup completed.
pause
exit /b 0

:install_runtime
SET RUNTIME_NAME=%~1
SET PYTHON_EXE=%~2
SET REQUIREMENTS_FILE=%~3
SET EXTRA_PACKAGES=%~4

if not exist "%PYTHON_EXE%" (
	echo Missing runtime python for %RUNTIME_NAME%: %PYTHON_EXE%
	exit /b 1
)

echo Setting up %RUNTIME_NAME%...
"%PYTHON_EXE%" "%~dp0get-pip.py"
if errorlevel 1 exit /b 1
"%PYTHON_EXE%" -m pip install --upgrade pip "setuptools<81" "packaging<25" wheel
if errorlevel 1 exit /b 1
if not "%EXTRA_PACKAGES%"=="" (
	"%PYTHON_EXE%" -m pip install %EXTRA_PACKAGES%
	if errorlevel 1 exit /b 1
)
"%PYTHON_EXE%" -m pip install -r "%REQUIREMENTS_FILE%"
if errorlevel 1 exit /b 1
exit /b 0

:error
echo.
echo Setup failed.
pause
exit /b 1