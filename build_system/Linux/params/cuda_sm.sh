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


function cuda_sm()
{
    # Convert the input to all lower case
    lower_input=$(echo "$1" | tr '[:upper:]' '[:lower:]')

    IFS=',' read -ra array <<< "$lower_input"
    for element in "${array[@]}"; do
        # Check if is about sm
        if [[ "$element" == *"sm" ]]; then
            val="${sm#*=}"
            is_numerical $val
            numer=$?
            if [[ $numer -eq 1 ]]; then
                DECX_CUDA_SM=$val
            else
                echo_error "CUDA SM architecture should be numerical"
            fi
        fi
        # Check if is about compute
        if [[ "$element" == *"compute" ]]; then
            val="${sm#*=}"
            is_numerical $val
            numer=$?
            if [[ $numer -eq 1 ]]; then
                DECX_CUDA_COMPUTE=$val
            else
                echo_error "CUDA computability should be numerical"
            fi
        fi
    done

    # If one of them is empty, make them equal
    if [ -z $DECX_CUDA_SM ]; then
        echo_status "CUDA SM architecture is empty, assign it with computability"
        DECX_CUDA_SM=$DECX_CUDA_COMPUTE
    fi
    if [ -z $DECX_CUDA_COMPUTE ]; then
        echo_status "CUDA computability is empty, assign it with SM architecture"
        DECX_CUDA_COMPUTE=$DECX_CUDA_SM
    fi
}