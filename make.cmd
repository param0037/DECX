@echo off
set PROJECT_PATH=%~dp0

set prefix=DECX_

cd %PROJECT_PATH%%prefix%%1
cmake --build build -j 12 --config Release"

cd %PROJECT_PATH%