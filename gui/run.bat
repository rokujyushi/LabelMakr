@ECHO off
cd /D %~dp0
call ..\shared\set_env.bat "%~dp0"
python labelmakr.py
pause