@echo off

cd "E:\DECX_world\samples\Conv2D"

cmake -B build -G"Visual Studio 16 2019"

cmake --build build --config Release

pause