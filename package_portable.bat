@ECHO off
SETLOCAL ENABLEEXTENSIONS
cd /D %~dp0

set "RUN_PYTHON=python"
if exist "%~dp0.venv\Scripts\python.exe" set "RUN_PYTHON=%~dp0.venv\Scripts\python.exe"

set "BUILD_PYTHON="
if defined LABELMAKR_BUILD_PYTHON (
	set "BUILD_PYTHON=%LABELMAKR_BUILD_PYTHON%"
) else if exist "%~dp0.venv\Scripts\python.exe" (
	set "BUILD_PYTHON=%~dp0.venv\Scripts\python.exe"
)

set "BASE_ARGS=--zip --force"
if defined BUILD_PYTHON set "BASE_ARGS=%BASE_ARGS% --build-python "%BUILD_PYTHON%""

if /I "%~1"=="-h" goto :usage
if /I "%~1"=="--help" goto :usage
if /I "%~1"=="/?" goto :usage

if "%~1"=="" goto :run_base

set "CORPUS_DIR=%~1"
shift
call "%RUN_PYTHON%" package_portable.py --corpus-dir "%CORPUS_DIR%" %BASE_ARGS% %*
exit /b %ERRORLEVEL%

:run_base
call "%RUN_PYTHON%" package_portable.py %BASE_ARGS%
exit /b %ERRORLEVEL%

:usage
	echo Usage:
	echo   package_portable.bat [corpus_dir] [additional package_portable.py args]
	echo.
	echo Examples:
	echo   package_portable.bat
	echo   package_portable.bat "I:\LabelMakr\shared\corpus"
	echo   package_portable.bat "I:\LabelMakr\shared\corpus" --package-name LabelMakr+_test
	echo.
	echo Notes:
	echo   - Uses .venv\Scripts\python.exe automatically when available.
	echo   - You can override the build python with LABELMAKR_BUILD_PYTHON.
	echo.
	call "%RUN_PYTHON%" package_portable.py --help
	exit /b %ERRORLEVEL%