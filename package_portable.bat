@ECHO off
cd /D %~dp0

if "%~1"=="" goto :run_base
python package_portable.py --corpus-dir "%~1" --zip --force --build-python I:\\LabelMakr\\.venv\\Scripts\\python.exe
exit /b %ERRORLEVEL%

:run_base
python package_portable.py --zip --force --build-python I:\\LabelMakr\\.venv\\Scripts\\python.exe
exit /b %ERRORLEVEL%

:usage
	echo Usage:
	echo   package_portable.bat [corpus_dir]
	echo.
	echo Example:
	echo   package_portable.bat "G:\LabelMakr+_v040\corpus"
	exit /b 1