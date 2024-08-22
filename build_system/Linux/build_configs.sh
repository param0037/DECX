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


function is_executable()
{
    stat=($(ls -l "$1"))
    if [[ "$stat" == *"x"* ]]; then
        return 1
    else
        return 0
    fi
}

function source_script()
{
    is_executable $1
    can_run=$?
    if [[ $can_run -ne 1 ]]; then
        chmod u+x $1
    fi
    source $1
}

source_script ./utils.sh
source_script ./params/param_lists.sh
source_script ./params/exp_lang.sh
source_script ./params/host_arch.sh
source_script ./params/cxx_ver.sh
source_script ./params/toolchain.sh
source_script ./params/set_module.sh


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


# function set_module()
# {
#     if [ -n "$1" ]; then
#         validate_module_name $1
#         if [ $? -eq 1 ]; then
#             export DECX_CURRENT_BUILD_MODULE=$1
#         else
#             echo_error "The indicated module does not exist"
#         fi
#     else
#         echo "[DECX] module options"
#     fi
# }


function set_module()
{
    __builtin_set_module $1
    export DECX_CURRENT_BUILD_MODULE=${stack[0]}
}


function toolchain()
{
    __builtin_toolchain $1
    export DECX_CMAKE_TOOLCHAIN_PATH=${stack[0]}
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


DECX_PARALLEL_BUILD=1
echo_status "Enable parallel build by default"

host_arch $1
DECX_HOST_ARCH="${stack[0]}"

exp_lang $2
DECX_EXP_C=${stack[0]}
DECX_EXP_CXX=${stack[1]}
DECX_EXP_PYTHON=${stack[2]}

cxx_ver $3
# DECX_CXX_VER=${stack[0]}

toolchain $4
# DECX_CMAKE_TOOLCHAIN_PATH=${stack[0]}