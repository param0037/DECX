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
export PROJECT_PATH_BUILD=$(dirname $(dirname $(dirname $full_path)))


stat=($(ls -l ./utils.sh))
if [[ "$stat" == *"x"* ]]; then
    source ./utils.sh
else
    chmod u+x ./utils.sh
    source ./utils.sh
fi

source_script ./params/param_lists.sh
source_script ./params/exp_lang.sh
source_script ./params/host_arch.sh
source_script ./params/cxx_ver.sh
source_script ./params/toolchain.sh
source_script ./params/set_module.sh
source_script ./params/cuda_sm.sh


function list_configs()
{
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

    echo "[DECX]   CUDA SM architecture : ${DECX_CUDA_SM}"
    echo "[DECX]   CUDA computability : ${DECX_CUDA_COMPUTE}"
}


function conf()
{
    source_script ./utils.sh

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
    source_script ./utils.sh

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
    source_script ./utils.sh

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
        echo_status "Parallel build disabled"
    else
        echo_error "Invalid input"
    fi
}


function build() {
    conf
    mk
}


function rebuild() {
    clean
    conf
    mk
}


function release() {
    set_module all
    clean
    conf
    mk
    install
}


function install()
{
    install_dir=$PROJECT_PATH_BUILD/install
    if [ ! -e "$install_dir" ]; then
        mkdir $install_dir
    fi

    cp -r $PROJECT_PATH_BUILD/srcs/includes $install_dir/includes
    echo_status "Install includes successfully"

    if [ ! -e "$install_dir/lib" ]; then
        mkdir $install_dir/lib
    fi

    cp -r $PROJECT_PATH_BUILD/build/bin/$DECX_HOST_ARCH \
          $install_dir/lib/
    echo_status "Install libs successfully"

    echo "Installing dependencies"
    
    SDL2_DIR=$PROJECT_PATH_BUILD/3rdparty/SDL2/$DECX_HOST_ARCH/Linux/lib
    if [ -d "$SDL2_DIR" ]; then
        cp -r $SDL2_DIR/* $install_dir/lib/$DECX_HOST_ARCH
        echo_status "Installed SDL2 libs"
    fi
    
    SDL2_Image_DIR=$PROJECT_PATH_BUILD/3rdparty/SDL2_Image/$DECX_HOST_ARCH/Linux/lib
    if [ -d "$SDL2_Image_DIR" ]; then
        cp -r $SDL2_Image_DIR/* $install_dir/lib/$DECX_HOST_ARCH
        echo_status "Install SDL2_Image libs"
    fi
}


DECX_PARALLEL_BUILD=1
echo_status "Enable parallel build by default"

host_arch $1

exp_lang $2

cxx_ver $3

toolchain $4
