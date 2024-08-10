@echo off
set PROJECT_PATH=%~dp0

set prefix=DECX_

cd %PROJECT_PATH%%prefix%%1
cmake -B build -G"Visual Studio 16 2019"

cd %PROJECT_PATH%