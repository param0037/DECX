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

# Set the host architecture
function host_arch()
{
    DECX_HOST_ARCH=""
    if [ -n "$1" ]; then
        validate_arch_name $1
        arch_name_correct=$?
        if [ $arch_name_correct -eq 1 ]; then
            DECX_HOST_ARCH=$1
        else
            echo_error "Host architecture should either be X86-64 or ARM64"
        fi
        # Rectify arch name
        is_aarch64 $DECX_HOST_ARCH
        is_arm64=$?
        if [ $is_arm64 -eq 1 ]; then
            echo_status "Set targeted architecture to aarch64"
            DECX_HOST_ARCH="aarch64"
        else
            echo_status "Set targeted architecture to x64"
            DECX_HOST_ARCH="x64"
        fi
    else
        if [ -n "$DECX_HOST_ARCH" ]; then
            DECX_HOST_ARCH=$DECX_HOST_ARCH
        else
            echo_status "Set targeted architecture to x64 by default"
            DECX_HOST_ARCH="x64"
        fi
    fi
}
