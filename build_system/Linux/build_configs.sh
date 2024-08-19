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


chmod u+x ./utils.sh
source ./utils.sh


function list_configs()
{
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    WHITE='\033[0m'
    YELLOW='\033[0;33m'

    echo "[DECX] Current build configurations : "
    echo "[DECX]   Target host architecture : $DECX_HOST_ARCH"
    if [ $DECX_EXP_C -eq 1 ]; then
        echo -e "[DECX]   C export : ${GREEN}Yes${WHITE}"
    else
        echo -e "[DECX]   C export : ${RED}No${WHITE}"
    fi
    if [ $DECX_EXP_CXX -eq 1 ]; then
        echo -e "[DECX]   CXX export : ${GREEN}Yes${WHITE}"
    else
        echo -e "[DECX]   CXX export : ${RED}No${WHITE}"
    fi
    if [ $DECX_EXP_PYTHON -eq 1 ]; then
        echo -e "[DECX]   Python export : ${GREEN}Yes${WHITE}"
    else
        echo -e "[DECX]   Python export : ${RED}No${WHITE}"
    fi

    echo "[DECX]   C++ version: C++$DECX_CXX_VER"
    # Print toolchain file path
    if [ -z "$DECX_CMAKE_TOOLCHAIN_PATH" ]; then
        echo -e "[DECX]   Cmake toolchain file path : ${YELLOW}Not Specified${WHITE}"
    else
        echo "[DECX]   Cmake toolchain file path : $DECX_CMAKE_TOOLCHAIN_PATH"
    fi
    # Print current module name
    if [ -z "$DECX_CURRENT_BUILD_MODULE" ]; then
        echo -e "[DECX]   Current module name build : ${YELLOW}Not Specified${WHITE}"
    else
        echo "[DECX]   Current module name build : $DECX_CURRENT_BUILD_MODULE"
    fi

    if [ $DECX_PARALLEL_BUILD -eq 1 ]; then
        echo -e "[DECX]   Parallel building : ${GREEN}Yes${WHITE}"
    else
        echo -e "[DECX]   Parallel building : ${RED}No${WHITE}"
    fi
}


# Set the host architecture
function host_arch()
{
    if [ -n "$1" ]; then
        validate_arch_name $1
        arch_name_correct=$?
        if [ $arch_name_correct -eq 1 ]; then
            export DECX_HOST_ARCH=$1
        else
            echo_error "Host architecture should either be X86-64 or ARM64"
        fi
        # Rectify arch name
        is_aarch64 $DECX_HOST_ARCH
        is_arm64=$?
        if [ $is_arm64 -eq 1 ]; then
            export DECX_HOST_ARCH="aarch64"
        else
            export DECX_HOST_ARCH="x86_64"
        fi
    else
        export DECX_HOST_ARCH=$DECX_HOST_ARCH
    fi
}


function exp_lang()
{
    if [ -n "$1" ]; then
        if [ -z $DECX_EXP_C ]; then
            export DECX_EXP_C=0
        else
            export DECX_EXP_C=$DECX_EXP_C
        fi
        if [ -z $DECX_EXP_CXX ]; then
            export DECX_EXP_CXX=0
        else
            export DECX_EXP_CXX=$DECX_EXP_CXX
        fi
        if [ -z $DECX_EXP_PYTHON ]; then
            export DECX_EXP_PYTHON=0
        else
            export DECX_EXP_PYTHON=$DECX_EXP_PYTHON
        fi
        
        IFS=',' read -ra array <<< "$1"
        for element in "${array[@]}"; do
            # Check CXX export
            lower_input=$(echo "$element" | tr '[:upper:]' '[:lower:]')

            if [ "$lower_input" = "cxx" ] || [ "$lower_input" = "cpp" ]; then
                export DECX_EXP_CXX=1
                echo_status "Enabled C++ export"
            fi
            # Check C export
            if [ "$lower_input" = "c" ]; then
                export DECX_EXP_C=1
                echo_status "Enabled C export"
            fi
            # Check Python export
            if [ "$lower_input" = "python" ] || [ "$lower_input" = "py" ]; then
                export DECX_EXP_PYTHON=1
                echo_status "Enabled Python export"
            fi
        done

        # Check if none of the language is exported
        if [ $DECX_EXP_C -eq 0 ] && [ $DECX_EXP_CXX -eq 0 ] && [ $DECX_EXP_PYTHON -eq 0 ]; then
            echo_error "At least one language should be exported"
        fi
    fi
}


# Set C++ version
function cxx_ver()
{
    if [ -n "$1" ]; then
        export DECX_CXX_VER=$1
    else
        export DECX_CXX_VER=$DECX_CXX_VER
    fi
    if [ -z "$DECX_CXX_VER" ]; then
        export DECX_CXX_VER=14
        echo_status "No C++ version is given, assigned to C++14 by default"
    fi
}


# Set the cmake toolchain file, if applicable
function toolchain()
{
    if [ -n "$1" ]; then
        export DECX_CMAKE_TOOLCHAIN_PATH=$1
    else
        export DECX_CMAKE_TOOLCHAIN_PATH=$DECX_CMAKE_TOOLCHAIN_PATH
    fi

    is_aarch64 $DECX_HOST_ARCH
    is_arm64=$?
    if [ $is_arm64 -eq 1 ]; then
        if [ -z "$DECX_CMAKE_TOOLCHAIN_PATH" ]; then
            echo_error "Cross compilation must specify the toolchain file"
        fi
    fi
}


function set_module()
{
    if [ -n "$1" ]; then
        validate_module_name $1
        if [ $? -eq 1 ]; then
            export DECX_CURRENT_BUILD_MODULE=$1
        else
            echo_error "The indicated module does not exist"
        fi
    else
        echo "[DECX] module options"
    fi
}


function conf()
{
    is_aarch64 $DECX_HOST_ARCH
    if [ $? -eq 1 ]; then
        if [ -z "$DECX_CMAKE_TOOLCHAIN_PATH" ]; then
            echo_error "Cross compilation must specify the toolchain file"
            exit -1
        fi
    fi

    if [ -z $1 ]; then
        ./build.sh -i $DECX_CURRENT_BUILD_MODULE
    else
        validate_module_name $1
        if [ $? -eq 0 ]; then
            echo_error "The indicated module does not exist"
        fi
        ./build.sh -i $1
    fi
}


function mk()
{
    if [ -z $1 ]; then
        ./build.sh -m $DECX_CURRENT_BUILD_MODULE
    else
        validate_module_name $1
        if [ $? -eq 0 ]; then
            echo_error "The indicated module does not exist"
        fi
        ./build.sh -m $1
    fi
}


function clean()
{
    if [ -z $1 ]; then
        ./build.sh -c $DECX_CURRENT_BUILD_MODULE
    else
        validate_module_name $1
        if [ $? -eq 0 ]; then
            echo_error "The indicated module does not exist"
        fi
        ./build.sh -c $1
    fi
}


function parallel_build()
{
    lower_input=$(echo "$1" | tr '[:upper:]' '[:lower:]')
    if [ "$lower_input" = "1" ] || [ "$lower_input" = 'true' ]; then
        export DECX_PARALLEL_BUILD=1
        echo_status "Parallel build enabled"
    elif [ "$lower_input" = "0" ] || [ "$lower_input" = 'false' ]; then
        export DECX_PARALLEL_BUILD=0
        echo_status "Parallel build enabled"
    else
        echo_error "Invalid input"
    fi
}


export DECX_PARALLEL_BUILD=1
echo_status "Enable parallel build by default"

host_arch $1
exp_lang $2
cxx_ver $3
toolchain $4
