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


file(GLOB GEMM "${DECX_WORLD_ABS_DIR}/srcs/modules/BLAS/GEMM/CPU/*.cxx")
if(${_DECX_HOST_ARCH_} STREQUAL "x64")
    file(GLOB_RECURSE BP "${DECX_WORLD_ABS_DIR}/srcs/modules/BLAS/Basic_process/*.cxx")
    file(GLOB EW "${DECX_WORLD_ABS_DIR}/srcs/modules/BLAS/Arithmetic/Matrix/*.cxx" 
                 "${DECX_WORLD_ABS_DIR}/srcs/common/element_wise/common/*.cxx")
else()
file(GLOB_RECURSE BP "${DECX_WORLD_ABS_DIR}/srcs/modules/BLAS/Basic_process/extension/*.cxx")
endif()

# Combine these kernel objects to GEMM_xxx_cpu.lib later
add_subdirectory("${DECX_WORLD_ABS_DIR}/srcs/modules/BLAS/GEMM/CPU/fp32" "${DECX_SUBBUILD_BIN_DIR}/gemm_fp32_cpu")
add_subdirectory("${DECX_WORLD_ABS_DIR}/srcs/modules/BLAS/GEMM/CPU/fp64" "${DECX_SUBBUILD_BIN_DIR}/gemm_64b_cpu")
if(${_DECX_HOST_ARCH_} STREQUAL "x64")
    add_subdirectory("${DECX_WORLD_ABS_DIR}/srcs/modules/BLAS/GEMM/CPU/cplxd" "${DECX_SUBBUILD_BIN_DIR}/gemm_cplxd_cpu")
endif()

# include common sources
include("${DECX_WORLD_ABS_DIR}/srcs/common/FMGR/FMGR_COM.cmake")
include("${DECX_WORLD_ABS_DIR}/srcs/common/Basic_process/extension/extension_com.cmake")

# Some parts of DECX_BLAS_CPU is not designed to support aarch64 yet, this will be done in the future
if(${_DECX_HOST_ARCH_} STREQUAL "x64")
include("${DECX_WORLD_ABS_DIR}/srcs/common/Basic_process/fill/fill_com.cmake")
include("${DECX_WORLD_ABS_DIR}/srcs/common/Basic_process/transpose/transpose_com.cmake")
include("${DECX_WORLD_ABS_DIR}/srcs/common/Basic_process/type_cast/typecast_com.cmake")


add_library(DECX_BLAS_CPU SHARED ${GEMM}                ${BP} 
                                 ${EXT_CPU_COM_SRCS}    ${FMGR_CPU_COM_SRCS}
                                 ${FILL_CPU_COM_SRCS}   ${TRP_CPU_COM_SRCS} 
                                 ${TYPECAST_CPU_COM_SRCS} ${EW})

target_link_libraries(DECX_BLAS_CPU PRIVATE gemm_fp32_cpu
                                    PRIVATE gemm_64b_cpu
                                    PRIVATE gemm_cplxd_cpu)
else()

add_library(DECX_BLAS_CPU SHARED ${GEMM}                ${BP} 
                                 ${EXT_CPU_COM_SRCS}    ${FMGR_CPU_COM_SRCS})

target_link_libraries(DECX_BLAS_CPU PRIVATE gemm_fp32_cpu
                                    PRIVATE gemm_64b_cpu)
                                    # PRIVATE gemm_cplxd_cpu)
endif()


if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_link_libraries(DECX_BLAS_CPU PUBLIC DECX_core_CPU.lib)
else()
    target_link_libraries(DECX_BLAS_CPU PUBLIC DECX_core_CPU.so)
endif()