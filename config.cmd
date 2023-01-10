@echo off
set PROJECT_PATH_CONFIG=%~dp0

cd %PROJECT_PATH_CONFIG%/DECX_allocation
cmake -B build -G"Visual Studio 16 2019"

cd %PROJECT_PATH_CONFIG%/DECX_classes
cmake -B build -G"Visual Studio 16 2019"

cd %PROJECT_PATH_CONFIG%/DECX_Image
cmake -B build -G"Visual Studio 16 2019"

cd %PROJECT_PATH_CONFIG%/DECX_cpu
cmake -B build -G"Visual Studio 16 2019"


cd %PROJECT_PATH_CONFIG%/DECX_CUDA
cmake -B build -G"Visual Studio 16 2019"

cd %PROJECT_PATH_CONFIG%

echo "Config done"
pause
