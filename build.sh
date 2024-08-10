#!/bin/bash

# ----------------------------------------------------------------------------------
# Author : Wayne Anderson
# Date : 2021.04.16
# ----------------------------------------------------------------------------------
#
# This is a part of the open source project named "DECX", a high-performance scientific
# computational library. This project follows the MIT License. For more information
# please visit https:
#
# Copyright (c) 2021 Wayne Anderson
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this
# software and associated documentation files (the "Software"), to deal in the Software
# without restriction, including without limitation the rights to use, copy, modify,
# merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to the following
# conditions:
#
# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


full_path=$(realpath $0)
PROJECT_PATH_BUILD=$(dirname $full_path)

function printf_info()
{
    echo "build.sh -[action] [project_name]"
    echo "actions : "
    echo "         -c Clean the project(s), specify which project to be cleaned"
    echo "         -i Configure the project(s), specify which project to be configured"
    echo "         -m Build the project(s), specify which project to be built"
    echo "         -all Clean, configure or build all the projects"
    echo "project_names : "
    echo "         core_CPU(CUDA), BLAS_CPU(CUDA), DSP_CPU(CUDA), CV_CPU(CUDA), NN_CPU)CUDA)"
}


# clean
function clean_single()
{
    # Clean the generated asm sources
    #if [ "$1" = "DSP_CPU" ]; then
        #$PROJECT_PATH_BUILD/srcs/common/SIMD/x86_64/asm_preproc_SVML.sh -c NOP
    #fi
    #if [ "$1" = "core_CPU" ]; then
        #$PROJECT_PATH_BUILD/srcs/modules/core/configs/x86_64/asm_preproc_configs.sh -c NOP
    #fi

    if [ -d "$PROJECT_PATH_BUILD/DECX_"$1"/build/" ]; then
        rm -rf "$PROJECT_PATH_BUILD/DECX_"$1"/build/"
    else
        echo "Path not exist, skipped"
    fi

    if [ -d "$PROJECT_PATH_BUILD/DECX_"$1"/GCC/" ]; then
        rm -rf "$PROJECT_PATH_BUILD/DECX_"$1"/GCC/"
    else
        echo "Path not exist, skipped"
    fi
}



function clean_all()
{
    clean_single "core_CPU"
    clean_single "core_CUDA"
    clean_single "BLAS_CPU"
    clean_single "BLAS_CUDA"
    clean_single "DSP_CPU"
    clean_single "DSP_CUDA"
    clean_single "CV_CPU"
    clean_single "CV_CUDA"
    clean_single "NN_CPU"
    clean_single "NN_CUDA"

    cd $PROJECT_PATH_BUILD

    echo "Cleaned all built"
}


function clean_optional()
{
    if [ "$1" = "all" ]; then
        clean_all
    else
        clean_single $1
    fi
}

# config

function config_single()
{
    # If is DSP_CPU, configure asm sources for math intrinsics
    #if [ "$1" = "DSP_CPU" ]; then
        #$PROJECT_PATH_BUILD/srcs/common/SIMD/x86_64/asm_preproc_SVML.sh -i NOP
    #fi
    # If is core_CPU, configure asm sources for CPU configuration
    #if [ "$1" = "core_CPU" ]; then
        #$PROJECT_PATH_BUILD/srcs/modules/core/configs/x86_64/asm_preproc_configs.sh -i NOP
    #fi
    cd "$PROJECT_PATH_BUILD/DECX_$1"
    echo "$PROJECT_PATH_BUILD/DECX_$1"
    cmake -D_CPP_EXPORT_=true -D_C_EXPORT_=true -B build -G"Unix Makefiles"
}


function config_all()
{
    config_single "core_CPU"
    config_single "core_CUDA"
    config_single "BLAS_CPU"
    config_single "BLAS_CUDA"
    config_single "DSP_CPU"
    config_single "DSP_CUDA"
    config_single "CV_CPU"
    config_single "CV_CUDA"
    config_single "NN_CPU"
    config_single "NN_CUDA"

    cd $PROJECT_PATH_BUILD

    echo "All config success"
}


function config_optional()
{
    if [ "$1" = "all" ]; then
        config_all
    else
        config_single $1
    fi
}

# build

function build_single()
{
    cd "$PROJECT_PATH_BUILD/DECX_"$1
    cmake --build build -j 12 --config Release
    cp $PROJECT_PATH_BUILD/bin/x64/libDECX_DSP_CPU.so ~/DECX/libs/x64/
}


function build_all()
{
    build_single "core_CPU"
    build_single "core_CUDA"
    build_single "BLAS_CPU"
    build_single "BLAS_CUDA"
    build_single "DSP_CPU"
    build_single "DSP_CUDA"
    build_single "CV_CPU"
    build_single "CV_CUDA"
    build_single "NN_CPU"
    build_single "NN_CUDA"

    cd $PROJECT_PATH_BUILD

    echo "All build success"
}


function build_optional()
{
    if [ "$1" = "all" ]; then
        build_all
    else
        build_single $1
    fi
}


if [ "$#" -eq 0 ]; then
    echo "Require parameters"
    printf_info
    exit 1
fi


while getopts ":m:c:i:" opt; do
    case $opt in
        m)
            build_optional $OPTARG
            ;;
        c)
            clean_optional $OPTARG
            ;;
        i)
            config_optional $OPTARG
            ;;
    esac
done

