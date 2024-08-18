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


function echo_status() 
{
    CYAN='\033[0;36m'
    WHITE='\033[0m'
    echo -e "${CYAN}[DECX] STATUS: $1${WHITE}"
}


function echo_error() 
{
    RED='\033[0;31m'
    WHITE='\033[0m'
    echo -e "${RED}[DECX] ERROR: $1${WHITE}"
}


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


function validate_module_name()
{
    lower_input=$(echo "$1" | tr '[:upper:]' '[:lower:]')
    MODULES_SET="core_cpu,core_cuda,blas_cpu,blas_cuda,dsp_cpu,dsp_cuda,cv_cpu,cv_cuda,nn_cpu,nn_cuda"

    IFS=',' read -ra array<<< "$MODULES_SET"
    for element in "${array[@]}"; do
        if [ "$element" = "$lower_input" ]; then
            return 1
        fi
    done

    return 0
}


function validate_arch_name()
{
    lower_input=$(echo "$1" | tr '[:upper:]' '[:lower:]')
    ARCH_SET="x86_64,x86-64,x86,aarch64,arm,arm64"

    IFS=',' read -ra array<<< "$ARCH_SET"
    for element in "${array[@]}"; do
        if [ "$element" = "$lower_input" ]; then
            return 1
        fi
    done

    return 0
}


function is_aarch64()
{
    lower_input=$(echo "$1" | tr '[:upper:]' '[:lower:]')
    ARCH_SET="aarch64,arm,arm64"
    
    IFS=',' read -ra array<<< "$ARCH_SET"
    for element in "${array[@]}"; do
        if [ "$element" = "$lower_input" ]; then
            return 1
        fi
    done

    return 0
}


