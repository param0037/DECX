@echo off
set PROJECT_PATH_BUILD=%~dp0

cd %PROJECT_PATH_BUILD%/DECX_allocation
cmake --build build --config Release

cd %PROJECT_PATH_BUILD%/DECX_classes
cmake --build build --config Release

cd %PROJECT_PATH_BUILD%/DECX_Image
cmake --build build --config Release

cd %PROJECT_PATH_BUILD%/DECX_cpu
cmake --build build --config Release


cd %PROJECT_PATH_BUILD%/DECX_CUDA
cmake --build build --config Release"

cd %PROJECT_PATH_BUILD%

echo "All build success"
pause