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

file(GLOB FFT "${DECX_WORLD_ABS_DIR}/srcs/modules/DSP/FFT/CPU/*.cxx")
file(GLOB FILTERS "${DECX_WORLD_ABS_DIR}/srcs/modules/DSP/filters/CPU/*.cxx")
file(GLOB CONV "${DECX_WORLD_ABS_DIR}/srcs/modules/DSP/convolution/CPU/*.cxx")

file(GLOB_RECURSE GATHER "${DECX_WORLD_ABS_DIR}/srcs/common/Basic_process/gather/CPU/*.cxx"
                 "${DECX_WORLD_ABS_DIR}/srcs/modules/DSP/resample/CPU/*.cxx")

include("${DECX_WORLD_ABS_DIR}/srcs/common/Element_wise/elementwise_com.cmake")

add_subdirectory("${DECX_WORLD_ABS_DIR}/srcs/modules/DSP/FFT/CPU/1D" "${DECX_SUBBUILD_BIN_DIR}/fft1d_cpu")
add_subdirectory("${DECX_WORLD_ABS_DIR}/srcs/modules/DSP/FFT/CPU/2D" "${DECX_SUBBUILD_BIN_DIR}/fft2d_cpu")
add_subdirectory("${DECX_WORLD_ABS_DIR}/srcs/modules/DSP/FFT/CPU/3D" "${DECX_SUBBUILD_BIN_DIR}/fft3d_cpu")
add_subdirectory("${DECX_WORLD_ABS_DIR}/srcs/modules/DSP/FFT/CPU/FFT_common" "${DECX_SUBBUILD_BIN_DIR}/fft_common_cpu")
add_subdirectory("${DECX_WORLD_ABS_DIR}/srcs/modules/DSP/convolution/CPU/2D" "${DECX_SUBBUILD_BIN_DIR}/conv2d_cpu")


add_library(DECX_DSP_CPU SHARED ${FFT} 
                                ${FILTERS}
                                ${CONV}
                                ${GATHER}
                                ${EW_CPU_COM_SRCS})

target_link_libraries(DECX_DSP_CPU PRIVATE fft1d_cpu
                                   PRIVATE fft2d_cpu
                                   PRIVATE fft3d_cpu
                                   PRIVATE fft_common_cpu
                                   PRIVATE conv2d_cpu)


if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    target_link_libraries(DECX_DSP_CPU PUBLIC DECX_core_CPU.lib)
    target_link_libraries(DECX_DSP_CPU PUBLIC DECX_BLAS_CPU.lib)

elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_libraries(DECX_DSP_CPU PUBLIC DECX_core_CPU.so)
    target_link_libraries(DECX_DSP_CPU PUBLIC DECX_BLAS_CPU.so)
    
else()
    message(STATUS "Building on unknown platform. Failed, not supported!")
endif()
