@echo off
set PROJECT_PATH=%~dp0

set prefix=DECX_

cd %PROJECT_PATH%%prefix%%1
rmdir /s /q "build"
rmdir /s /q "MSVC"

cd %PROJECT_PATH%