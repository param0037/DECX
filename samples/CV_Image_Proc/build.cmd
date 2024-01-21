@echo off

cd "E:\DECX_world\samples\CV_Image_Proc"

cmake -B build -G"Visual Studio 16 2019"

cmake --build build --config Release

pause