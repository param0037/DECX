@echo off

set PROJECT_PATH=%~dp0

set action=%1
set target_name=%2

if "%action%"=="" (
    echo "build.cmd [action] [project_name]"
    echo "actions : "
    echo "         c Clean the project(s), specify which project to be cleaned"
    echo "         i Configure the project(s), specify which project to be configured"
    echo "         m Build the project(s), specify which project to be built"
    echo "         all Clean, configure or build all the projects"
    echo "project_names : "
    echo "         Async, core_CPU(CUDA), BLAS_CPU(CUDA), DSP_CPU(CUDA), CV_CPU(CUDA), NN_CPU)CUDA)"
)

if %action%==c (
    if %target_name% neq all (
        call clear.cmd %target_name%
    ) else if %target_name%==all (
        call clear.cmd Async
        call clear.cmd core_CPU
        call clear.cmd core_CUDA
        call clear.cmd BLAS_CPU
        call clear.cmd BLAS_CUDA
        call clear.cmd DSP_CPU
        call clear.cmd DSP_CUDA
        call clear.cmd CV_CPU
        call clear.cmd CV_CUDA
        call clear.cmd NN_CPU
        call clear.cmd NN_CUDA
    ) else (
        echo Taget not exist!
    )
) else if %action%==i (
    if %target_name% neq all (
        call config.cmd %target_name%
    ) else if %target_name%==all (
        call config.cmd Async
        call config.cmd core_CPU
        call config.cmd core_CUDA
        call config.cmd BLAS_CPU
        call config.cmd BLAS_CUDA
        call config.cmd DSP_CPU
        call config.cmd DSP_CUDA
        call config.cmd CV_CPU
        call config.cmd CV_CUDA
        call config.cmd NN_CPU
        call config.cmd NN_CUDA
    ) else (
        echo Taget not exist!
    )
) else if %action%==m (
    if %target_name% neq all (
        call make.cmd %target_name%
    ) else if %target_name%==all (
        call make.cmd Async
        call make.cmd core_CPU
        call make.cmd core_CUDA
        call make.cmd BLAS_CPU
        call make.cmd BLAS_CUDA
        call make.cmd DSP_CPU
        call make.cmd DSP_CUDA
        call make.cmd CV_CPU
        call make.cmd CV_CUDA
        call make.cmd NN_CPU
        call make.cmd NN_CUDA
    ) else (
        echo Taget not exist!
    )
) else (
    echo Error action called!
)