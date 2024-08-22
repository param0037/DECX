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
source_script ./utils.sh


function set_module()
{
    DECX_CURRENT_BUILD_MODULE=""
    if [ -n "$1" ]; then
        validate_module_name $1
        if [ $? -eq 1 ]; then
            DECX_CURRENT_BUILD_MODULE=$1
        else
            echo_error "The indicated module does not exist"
        fi
    else
        echo "[DECX] module options"
        echo "      module          | introduction"
        echo "     --------------------------------------------------------------------------------------"
        echo "      core_CPU        | Core of DECX, running on host"
        echo "      core_CUDA       | Core of DECX, interface of CUDA runtime and managing CUDA resources"
        echo "      BLAS_CPU        | Linear algebra module of DECX, running on CPU"
        echo "      BLAS_CUDA       | Linear algebra module of DECX, mostly running on CUDA"
        echo "      DSP_CPU         | Digital signal processing module, running on CPU"
        echo "      DSP_CUDA        | Digital signal processing module, mostly running on CUDA"
        echo "      CV_CPU          | Computer vision module, running on CPU"
        echo "      CV_CUDA         | Computer vision module, mostly running on CUDA"
        echo "      NN_CPU          | Neural network module, running on CPU"
        echo "      NN_CUDA         | Neural network module, mostly running on CUDA"
        echo "      all             | build all modules"
    fi

    stack[0]=$DECX_CURRENT_BUILD_MODULE
}
