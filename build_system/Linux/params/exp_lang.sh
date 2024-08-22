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


function exp_lang()
{
    DECX_EXP_C=0
    DECX_EXP_CXX=0
    DECX_EXP_PYTHON=0

    if [ -n "$1" ]; then
        if [ -n $DECX_EXP_C ]; then
            DECX_EXP_C=$DECX_EXP_C
        fi
        if [ -n $DECX_EXP_CXX ]; then
            DECX_EXP_CXX=$DECX_EXP_CXX
        fi
        if [ -n $DECX_EXP_PYTHON ]; then
            DECX_EXP_PYTHON=$DECX_EXP_PYTHON
        fi
        
        IFS=',' read -ra array <<< "$1"
        for element in "${array[@]}"; do
            # Check CXX export
            lower_input=$(echo "$element" | tr '[:upper:]' '[:lower:]')

            if [ "$lower_input" = "cxx" ] || [ "$lower_input" = "cpp" ]; then
                DECX_EXP_CXX=1
                echo_status "Enabled C++ export"
            fi
            # Check C export
            if [ "$lower_input" = "c" ]; then
                DECX_EXP_C=1
                echo_status "Enabled C export"
            fi
            # Check Python export
            if [ "$lower_input" = "python" ] || [ "$lower_input" = "py" ]; then
                DECX_EXP_PYTHON=1
                echo_status "Enabled Python export"
            fi
        done

        # Check if none of the language is exported
        if [ $DECX_EXP_C -eq 0 ] && [ $DECX_EXP_CXX -eq 0 ] && [ $DECX_EXP_PYTHON -eq 0 ]; then
            echo_error "At least one language should be exported"
        fi
    else
        DECX_EXP_C=1
        DECX_EXP_CXX=1
        DECX_EXP_PYTHON=0
        echo_status "No language export is given, enable c, cxx by default"
    fi
    # Ret
    # stack=($EXP_C $EXP_CXX $EXP_PYTHON)
}
