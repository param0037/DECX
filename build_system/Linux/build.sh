#!/bin/bash

# ----------------------------------------------------------------------------------
# Author : Wayne Anderson
# Date : 2021.04.16
# ----------------------------------------------------------------------------------
#
# This is a part of the open source project named "DECX", a high-performance scientific
# computational library. This project follows the MIT License. For more information
# please visit https://github.com/param0037/DECX.
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


function prebuilt_needed()
{
    prebuilt_cmd=""
    if [[ "$1" == *"CV"* ]]; then
        prebuilt_cmd="$prebuilt_cmd,SDL2,SDL2_Image"
    fi
    if [[ $DECX_EXP_PYTHON -eq 1 ]]; then
        prebuilt_cmd="$prebuilt_cmd,Python"
    fi
    echo -n "$prebuilt_cmd"
}


# clean
function clean_single()
{
    # Clean the generated asm sources
    if [[ "BLAS_CPU,DSP_CPU" == *"$1"* ]]; then
        echo_status "Deleted generated platform dependent asm files for SVML"
        # chmod u+x $PROJECT_PATH_BUILD/srcs/common/SIMD/x86_64/asm_preproc_SVML.sh
        run_script $PROJECT_PATH_BUILD/srcs/common/SIMD/x86_64/asm_preproc_SVML.sh -c NOP
    fi
    if [ "$1" = "core_CPU" ]; then
        # chmod u+x $PROJECT_PATH_BUILD/srcs/modules/core/configs/x86_64/asm_preproc_configs.sh
        run_script $PROJECT_PATH_BUILD/srcs/modules/core/configs/x86_64/asm_preproc_configs.sh -c NOP
    fi

    cmake_bin_dir="$PROJECT_PATH_BUILD/build/DECX_$1"

    if [ -d "$cmake_bin_dir" ]; then
        rm -rf "$cmake_bin_dir"
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
    # clean_single "NN_CPU"
    clean_single "NN_CUDA"

    cd $PROJECT_PATH_BUILD

    echo_success "Cleaned all built"
}


function clean_optional()
{
    if [ "$1" = "ALL" ]; then
        clean_all
    else
        clean_single $1
    fi
}


# config
function config_single()
{
    requirement_statement=$(prebuilt_needed $1)
    
    run_script $PROJECT_PATH_BUILD/build_system/Linux/prebuilts_manager.sh $requirement_statement

    if [ ! -e "$PROJECT_PATH_BUILD/build" ]; then
        mkdir $PROJECT_PATH_BUILD/build
    fi

    # If is DSP_CPU, configure asm sources for math intrinsics
    if [[ "BLAS_CPU,DSP_CPU" == *"$1"* ]]; then
        echo_status "Generated platform dependent asm files for SVML"
        run_script $PROJECT_PATH_BUILD/srcs/common/SIMD/x86_64/asm_preproc_SVML.sh -i NOP
    fi
    # If is core_CPU, configure asm sources for CPU configuration
    if [ "$1" = "core_CPU" ]; then
        if [ "$DECX_HOST_ARCH" = "x64" ]; then
            echo_status "generating x86-64 CPUID assembly"
            run_script $PROJECT_PATH_BUILD/srcs/modules/core/configs/x86_64/asm_preproc_configs.sh -i NOP
        fi
    fi

    module_name=$1
    # cd "$PROJECT_PATH_BUILD/DECX_$1"
    cd "$PROJECT_PATH_BUILD/srcs/modules/${module_name%%_*}/"
    echo_status "cd to $PROJECT_PATH_BUILD/srcs/modules/${module_name%%_*}/"

    cmake_config_cmd="cmake"

    is_CUDA_module $1
    is_CUDA=$?
    if [ $is_CUDA -eq 1 ]; then
        cmake_config_cmd="$cmake_config_cmd -D_DECX_CUDA_MODULE_=true"
        cmake_config_cmd="$cmake_config_cmd -D_DECX_CUDA_SM_=$DECX_CUDA_SM"
        cmake_config_cmd="$cmake_config_cmd -D_DECX_CUDA_COMPUTE_=$DECX_CUDA_COMPUTE"
    else
        cmake_config_cmd="$cmake_config_cmd -D_DECX_CUDA_MODULE_=false"
    fi

    is_aarch64 $DECX_HOST_ARCH
    is_arm64=$?
    if [ $is_arm64 -eq 1 ]; then
        cmake_config_cmd="$cmake_config_cmd -DCMAKE_TOOLCHAIN_FILE=$DECX_CMAKE_TOOLCHAIN_PATH"
        cmake_config_cmd="$cmake_config_cmd -D_DECX_HOST_ARCH_=aarch64 \
            -DANDROID_ABI=arm64-v8a \
            -DANDROID_PLATFORM=android-24"
    else
        cmake_config_cmd="$cmake_config_cmd -D_DECX_HOST_ARCH_=x64"
    fi

    if [ $DECX_EXP_C ]; then
        cmake_config_cmd="$cmake_config_cmd -D_C_EXPORT_=true"
    fi
    if [ $DECX_EXP_CXX ]; then
        cmake_config_cmd="$cmake_config_cmd -D_CPP_EXPORT_=true"
    fi
    cmake_bin_dir="$PROJECT_PATH_BUILD/build/DECX_$1/$DECX_HOST_ARCH"
    cmake_config_cmd="$cmake_config_cmd -B $cmake_bin_dir -G\"Unix Makefiles\""
    
    eval $cmake_config_cmd

    echo_success "--------------------------------------------------- Successfully configured $1 ---------------------------------------------------"
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
    # config_single "NN_CPU"
    config_single "NN_CUDA"
    
    cd $PROJECT_PATH_BUILD

    echo_success "All config success"
}


function config_optional()
{
    if [ "$1" = "ALL" ]; then
        config_all
    else
        config_single $1
    fi
}


# build
function build_single()
{
    cmake_bin_dir="$PROJECT_PATH_BUILD/build/DECX_$1/$DECX_HOST_ARCH"

    if [ $DECX_PARALLEL_BUILD -eq 1 ]; then
        # Get the maximum concurrency of CPU
        max_concurrency=$(nproc)
        cmake --build $cmake_bin_dir -j $max_concurrency --config Release
    else
        cmake --build $cmake_bin_dir --config Release
    fi
    cp /media/wayne/Disk/DECX_world/build/bin/x64/libDECX_BLAS_CPU.so ~/DECX/libs/x64/

    echo_success "--------------------------------------------------- Successfully built $1 ---------------------------------------------------"
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
    # build_single "NN_CPU"
    build_single "NN_CUDA"

    cd $PROJECT_PATH_BUILD

    echo_status "All build success"
}


function build_optional()
{
    if [ "$1" = "ALL" ]; then
        build_all
    else
        build_single $1
    fi
}



stat=($(ls -l ./utils.sh))
can_run=0
if [[ "$stat" == *"x"* ]]; then
    can_run=1
else
    can_run=0
fi


if [[ $can_run -ne 1 ]]; then
    chmod u+x ./utils.sh
fi
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
