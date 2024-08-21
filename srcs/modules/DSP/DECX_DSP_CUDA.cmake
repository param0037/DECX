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


file(GLOB FFT_SRCS "${DECX_WORLD_ABS_DIR}/srcs/modules/DSP/FFT/CUDA/*.cu")
file(GLOB CONV_SRCS "${DECX_WORLD_ABS_DIR}/srcs/modules/DSP/convolution/CUDA/*.cu")
add_subdirectory("${DECX_WORLD_ABS_DIR}/srcs/modules/DSP/FFT/CUDA/1D" "${DECX_SUBBUILD_BIN_DIR}/fft1d_cuda")
add_subdirectory("${DECX_WORLD_ABS_DIR}/srcs/modules/DSP/FFT/CUDA/2D" "${DECX_SUBBUILD_BIN_DIR}/fft2d_cuda")
add_subdirectory("${DECX_WORLD_ABS_DIR}/srcs/modules/DSP/FFT/CUDA/3D" "${DECX_SUBBUILD_BIN_DIR}/fft3d_cuda")

add_subdirectory("${DECX_WORLD_ABS_DIR}/srcs/modules/DSP/convolution/CUDA/2D" "${DECX_SUBBUILD_BIN_DIR}/conv2d_cuda")


add_library(DECX_DSP_CUDA SHARED ${FFT_SRCS} ${CONV_SRCS} ${TRP_CUDA_COM_SRCS})


target_link_libraries(DECX_DSP_CUDA PRIVATE fft1d_cuda
                                    PRIVATE fft2d_cuda
                                    PRIVATE fft3d_cuda
                                    PRIVATE conv2d_cuda)


set_target_properties(DECX_DSP_CUDA PROPERTIES CUDA_ARCHITECTURES ${CUDA_TARGET_ARCH})
set_target_properties(DECX_DSP_CUDA PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)


if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    # Must tell NVCC and Host linker to consider multiple implementations of common CUDA source file
    # They will automatically generate symbols of __cudaRegisterLinkBinary with different names.
    target_link_libraries(DECX_DSP_CUDA PUBLIC DECX_BLAS_CUDA.lib)
    target_link_libraries(DECX_DSP_CUDA PUBLIC DECX_core_CUDA.lib)

elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # Must tell NVCC and Host linker to consider multiple implementations of common CUDA source file
    # They will automatically generate symbols of __cudaRegisterLinkBinary with different names.
    target_link_libraries(DECX_DSP_CUDA PUBLIC DECX_BLAS_CUDA.so)
    target_link_libraries(DECX_DSP_CUDA PUBLIC DECX_core_CUDA.so)
    
else()
endif()
