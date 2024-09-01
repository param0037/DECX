#   ----------------------------------------------------------------------------------
#   Author : Wayne Anderson
#   Date   : 2021.04.16
#   ----------------------------------------------------------------------------------
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

include_guard(GLOBAL)


set(BUILTIN_SVML_PATH "${DECX_WORLD_ABS_DIR}/srcs/common/SIMD/x86_64")


if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    enable_language(C ASM_MASM)
    message(STATUS "Windows platform detected, using MASM(ML64) to build")
    set(INTRIN_X86_64 "${BUILTIN_SVML_PATH}/decx_trigonometric_fp32_x64_masm.asm"
                      "${BUILTIN_SVML_PATH}/decx_trigonometric_fp64_x64_masm.asm")

elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    enable_language(C ASM_NASM)
    message(STATUS "Linux platform detected, using NASM to build")
    set(INTRIN_X86_64 "${BUILTIN_SVML_PATH}/decx_trigonometric_fp32_x64_nasm.asm"
                      "${BUILTIN_SVML_PATH}/decx_trigonometric_fp64_x64_nasm.asm")

endif()
