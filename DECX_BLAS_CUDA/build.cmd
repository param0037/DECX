@echo off

cd "E:\DECX_world\DECX_BLAS_CUDA"

cmake -B build -G"Visual Studio 16 2019"

cmake --build build --config Release

pause