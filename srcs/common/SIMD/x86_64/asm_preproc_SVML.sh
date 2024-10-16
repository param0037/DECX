#/!bin/bash

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
SCRIPT_PATH=$(dirname $full_path)
cd $SCRIPT_PATH

asm_files=("./decx_trigonometric_fp32_x64.s"
           "./decx_trigonometric_fp64_x64.s")

while getopts ":m:c:i:" opt; do
    case $opt in
        c)
            for file in "${asm_files[@]}"
            do
                target_file_name="${file::-2}_nasm.asm"
                if [ -f "$target_file_name" ]; then
                    rm -f "$target_file_name"
                else
                    echo "File not exist, skipped"
                fi
            done
            ;;
        i)
            for file in "${asm_files[@]}"
            do
                target_file_name="${file::-2}_nasm.asm"
                echo "Generating assembly sources : $target_file_name"
                gcc -E -P -x c $file -D__NASM__ > $target_file_name
            done
            ;;
    esac
done