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


# include to add compile definitions to the target
include("${DECX_WORLD_ABS_DIR}/srcs/compile_defs.cmake")

file(GLOB_RECURSE BP_SRCS "${DECX_WORLD_ABS_DIR}/srcs/modules/BLAS/Basic_process/extension/*.cu"
                          "${DECX_WORLD_ABS_DIR}/srcs/modules/BLAS/Basic_process/type_cast/*.cu"
                          "${DECX_WORLD_ABS_DIR}/srcs/modules/BLAS/Basic_process/type_statistics/*.cu"
                          "${DECX_WORLD_ABS_DIR}/srcs/modules/BLAS/Basic_process/transpose/*.cu")

file(GLOB GEMM_SRCS "${DECX_WORLD_ABS_DIR}/srcs/modules/BLAS/GEMM/CUDA/*.cu")
file(GLOB DP "${DECX_WORLD_ABS_DIR}/srcs/modules/BLAS/Dot product/CUDA/*.cu")
file(GLOB ARITH "${DECX_WORLD_ABS_DIR}/srcs/modules/BLAS/Arithmetic/*.cu")

add_subdirectory("${DECX_WORLD_ABS_DIR}/srcs/modules/BLAS/GEMM/CUDA/extreme_shapes" "${DECX_SUBBUILD_BIN_DIR}/gemm_cuda_extreme_shape")
add_subdirectory("${DECX_WORLD_ABS_DIR}/srcs/modules/BLAS/GEMM/CUDA/large_squares" "${DECX_SUBBUILD_BIN_DIR}/gemm_cuda_large_squares")

add_subdirectory("${DECX_WORLD_ABS_DIR}/srcs/modules/BLAS/Dot product/CUDA/1D" "${DECX_SUBBUILD_BIN_DIR}/DP1D")
add_subdirectory("${DECX_WORLD_ABS_DIR}/srcs/modules/BLAS/Dot product/CUDA/2D" "${DECX_SUBBUILD_BIN_DIR}/DP2D")

# Common srcs
add_subdirectory("${DECX_WORLD_ABS_DIR}/srcs/common/Basic_process/transpose" "${DECX_SUBBUILD_BIN_DIR}/TRP_CUDA")
add_subdirectory("${DECX_WORLD_ABS_DIR}/srcs/common/Element_wise" "${DECX_SUBBUILD_BIN_DIR}/EW_CUDA")
add_subdirectory("${DECX_WORLD_ABS_DIR}/srcs/common/FMGR" "${DECX_SUBBUILD_BIN_DIR}/FMGR_CUDA")


include("${DECX_WORLD_ABS_DIR}/srcs/common/Basic_process/extension/extension_com.cmake")
include("${DECX_WORLD_ABS_DIR}/srcs/common/Basic_process/type_cast/typecast_com.cmake")


add_library(DECX_BLAS_CUDA SHARED ${GEMM_SRCS}   
                                  ${DP}        ${BP_SRCS}
                                  ${EXT_CUDA_COM_SRCS}
                                  ${FILL_CUDA_COM_SRCS}
                                  ${TYPECAST_CUDA_COM_SRCS}
                                  ${ARITH})

set_target_properties(DECX_BLAS_CUDA PROPERTIES CUDA_ARCHITECTURES ${CUDA_TARGET_ARCH})
set_target_properties(DECX_BLAS_CUDA PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_link_libraries(DECX_BLAS_CUDA PRIVATE gemm_cuda_extreme_shape
                                     PRIVATE gemm_cuda_large_squares
                                     PRIVATE DP1D_cuda
                                     PRIVATE TRP_CUDA
                                     PRIVATE DP2D_cuda
                                     PRIVATE EW_CUDA
                                     PRIVATE FMGR_CUDA)

target_link_libraries(DECX_BLAS_CUDA PUBLIC DECX_core_CPU
                                     PUBLIC DECX_core_CUDA)