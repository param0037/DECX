@echo off
set PROJECT_PATH_CONFIG=%~dp0

cd %PROJECT_PATH_CONFIG%/DECX_core_CPU
cmake -B build -G"Visual Studio 16 2019"


cd %PROJECT_PATH_CONFIG%/DECX_core_CUDA
cmake -B build -G"Visual Studio 16 2019"


cd %PROJECT_PATH_CONFIG%/DECX_Async
cmake -B build -G"Visual Studio 16 2019"

cd %PROJECT_PATH_CONFIG%/DECX_BLAS_CPU
cmake -B build -G"Visual Studio 16 2019"

cd %PROJECT_PATH_CONFIG%/DECX_BLAS_CUDA
cmake -B build -G"Visual Studio 16 2019"


cd %PROJECT_PATH_CONFIG%/DECX_DSP_CPU
cmake -B build -G"Visual Studio 16 2019"

cd %PROJECT_PATH_CONFIG%/DECX_DSP_CUDA
cmake -B build -G"Visual Studio 16 2019"


cd %PROJECT_PATH_CONFIG%/DECX_NN_CPU
cmake -B build -G"Visual Studio 16 2019"


cd %PROJECT_PATH_CONFIG%/DECX_NN_CUDA
cmake -B build -G"Visual Studio 16 2019"

cd %PROJECT_PATH_CONFIG%

echo "Config done"
pause
