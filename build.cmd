@echo off
set PROJECT_PATH_BUILD=%~dp0

cd %PROJECT_PATH_BUILD%/DECX_core_CPU
cmake --build build --config Release

cd %PROJECT_PATH_BUILD%/DECX_core_CUDA
cmake --build build --config Release

cd %PROJECT_PATH_BUILD%/DECX_Async
cmake --build build --config Release"

cd %PROJECT_PATH_BUILD%/DECX_BLAS_CUDA
cmake --build build --config Release

cd %PROJECT_PATH_BUILD%/DECX_BLAS_CPU
cmake --build build --config Release


cd %PROJECT_PATH_BUILD%/DECX_DSP_CPU
cmake --build build --config Release"


cd %PROJECT_PATH_BUILD%/DECX_DSP_CUDA
cmake --build build --config Release"


cd %PROJECT_PATH_BUILD%/DECX_NN_CPU
cmake --build build --config Release"

cd %PROJECT_PATH_BUILD%/DECX_NN_CUDA
cmake --build build --config Release"

cd %PROJECT_PATH_BUILD%

echo "All build success"
pause