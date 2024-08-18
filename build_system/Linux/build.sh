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
PROJECT_PATH_BUILD=$(dirname $(dirname $(dirname $full_path)))

# clean
function clean_single()
{
    # Clean the generated asm sources
    if [ "$1" = "DSP_CPU" ]; then
        $PROJECT_PATH_BUILD/srcs/common/SIMD/x86_64/asm_preproc_SVML.sh -c NOP
    fi
    if [ "$1" = "core_CPU" ]; then
        $PROJECT_PATH_BUILD/srcs/modules/core/configs/x86_64/asm_preproc_configs.sh -c NOP
    fi

    if [ -d "$PROJECT_PATH_BUILD/DECX_"$1"/build/" ]; then
        rm -rf "$PROJECT_PATH_BUILD/DECX_"$1"/build/"
    else
        echo_status "Path not exist, skipped"
    fi

    if [ -d "$PROJECT_PATH_BUILD/DECX_"$1"/GNU/" ]; then
        rm -rf "$PROJECT_PATH_BUILD/DECX_"$1"/GNU/"
    else
        echo_status "Path not exist, skipped"
    fi

    if [ -d "$PROJECT_PATH_BUILD/DECX_"$1"/MSVC/" ]; then
        rm -rf "$PROJECT_PATH_BUILD/DECX_"$1"/MSVC/"
    else
        echo_status "Path not exist, skipped"
    fi

    if [ -d "$PROJECT_PATH_BUILD/DECX_"$1"/Clang/" ]; then
        rm -rf "$PROJECT_PATH_BUILD/DECX_"$1"/Clang/"
    else
        echo_status "Path not exist, skipped"
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

    echo_status "Cleaned all built"
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
    if [ "$1" = "DSP_CPU" ]; then
        $PROJECT_PATH_BUILD/srcs/common/SIMD/x86_64/asm_preproc_SVML.sh -i NOP
    fi
    # If is core_CPU, configure asm sources for CPU configuration
    if [ "$1" = "core_CPU" ]; then
        if [ "$DECX_HOST_ARCH" = "x86_64" ]; then
            $PROJECT_PATH_BUILD/srcs/modules/core/configs/x86_64/asm_preproc_configs.sh -i NOP
        fi
    fi

    cd "$PROJECT_PATH_BUILD/DECX_$1"
    echo_status "Current config project path is $PROJECT_PATH_BUILD/DECX_$1"

    cmake_config_cmd="cmake"

    is_aarch64 $DECX_HOST_ARCH
    if [ $? -eq 1 ]; then
        cmake_config_cmd="$cmake_config_cmd -DCMAKE_TOOLCHAIN_FILE=$DECX_CMAKE_TOOLCHAIN_PATH"
        cmake_config_cmd="$cmake_config_cmd -D_DECX_HOST_ARCH_=aarch64 \
            -DANDROID_ABI=arm64-v8a \
            -DANDROID_PLATFORM=android-24"
    else
        cmake_config_cmd="$cmake_config_cmd -D_DECX_HOST_ARCH_=x86_64"
    fi

    if [ $DECX_EXP_C ]; then
        cmake_config_cmd="$cmake_config_cmd -D_C_EXPORT_=true"
    fi
    if [ $DECX_EXP_CXX ]; then
        cmake_config_cmd="$cmake_config_cmd -D_CPP_EXPORT_=true"
    fi

    cmake_config_cmd="$cmake_config_cmd -B build -G\"Unix Makefiles\""
    eval $cmake_config_cmd
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

    echo_status "All config success"
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

    echo_status "All build success"
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
    echo_error "Require parameters"
    printf_info
    exit 1
fi


chmod u+x ./utils.sh
source ./utils.sh


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
