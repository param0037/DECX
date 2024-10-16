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

# Entry from dir=../
source_script $BUILD_SYSTEM_DIR/Linux/utils.sh

# Set the cmake toolchain file, if applicable
function toolchain()
{
    DECX_CMAKE_TOOLCHAIN_PATH=""
    if [ -n "$1" ]; then
        DECX_CMAKE_TOOLCHAIN_PATH=$1
    else
        DECX_CMAKE_TOOLCHAIN_PATH=$DECX_CMAKE_TOOLCHAIN_PATH
    fi

    is_aarch64 $DECX_HOST_ARCH
    is_arm64=$?
    if [ $is_arm64 -eq 1 ]; then
        if [ -z "$DECX_CMAKE_TOOLCHAIN_PATH" ]; then
            echo_error "Cross compilation must specify the toolchain file"
        fi
    fi
}
