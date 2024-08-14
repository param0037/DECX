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

cmake_minimum_required(VERSION 3.23)

set(_CPP_EXPORT_ CACHE BOOL "To control if this module export functions in CPP mode")
set(_C_EXPORT_ CACHE BOOL "To control if this module export functions in C mode")
set(_DECX_HOST_ARCH_ "x86_64" CACHE STRING "Specfiy the host architecture (either x86_64 or aarch64")

if(_CPP_EXPORT_ OR _C_EXPORT_)

project(DECX_BLAS_CPU LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# get upper dir
string(REGEX REPLACE "/$" "" CURRENT_FOLDER_ABSOLUTE ${CMAKE_CURRENT_SOURCE_DIR})
get_filename_component(DECX_WORLD_ABS_DIR ${CURRENT_FOLDER_ABSOLUTE} DIRECTORY)


# Define the path of the binaries built by subdirectories
set(DECX_SUBBUILD_BIN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/${CMAKE_CXX_COMPILER_ID})


# Define the path for the final built binary
if(CMAKE_CROSSCOMPILING)
    message("Host=aarch64")
    set(LIBRARY_OUTPUT_PATH ${DECX_WORLD_ABS_DIR}/bin/aarch64)
else()
    message("Host=x86_64")
    set(LIBRARY_OUTPUT_PATH ${DECX_WORLD_ABS_DIR}/bin/x64)
endif()

# include to add compile definitions to the target
include("${DECX_WORLD_ABS_DIR}/srcs/compile_defs.cmake")

file(GLOB GEMM "${DECX_WORLD_ABS_DIR}/srcs/modules/BLAS/GEMM/CPU/*.cxx")
# file(GLOB_RECURSE BP "${DECX_WORLD_ABS_DIR}/srcs/modules/BLAS/Basic_process/*.cxx")
file(GLOB_RECURSE BP "${DECX_WORLD_ABS_DIR}/srcs/modules/BLAS/Basic_process/extension/*.cxx")

# Combine these kernel objects to GEMM_xxx_cpu.lib later
add_subdirectory("${DECX_WORLD_ABS_DIR}/srcs/modules/BLAS/GEMM/CPU/fp32" "${DECX_SUBBUILD_BIN_DIR}/gemm_fp32_cpu")
add_subdirectory("${DECX_WORLD_ABS_DIR}/srcs/modules/BLAS/GEMM/CPU/fp64" "${DECX_SUBBUILD_BIN_DIR}/gemm_64b_cpu")
# add_subdirectory("${DECX_WORLD_ABS_DIR}/srcs/modules/BLAS/GEMM/CPU/cplxd" "${DECX_SUBBUILD_BIN_DIR}/gemm_cplxd_cpu")

# include common sources
include("${DECX_WORLD_ABS_DIR}/srcs/common/FMGR/FMGR_COM.cmake")
include("${DECX_WORLD_ABS_DIR}/srcs/common/Basic_process/extension/extension_com.cmake")
# include("${DECX_WORLD_ABS_DIR}/srcs/common/Basic_process/fill/fill_com.cmake")
# include("${DECX_WORLD_ABS_DIR}/srcs/common/Basic_process/transpose/transpose_com.cmake")
# include("${DECX_WORLD_ABS_DIR}/srcs/common/Basic_process/type_cast/typecast_com.cmake")

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    link_directories(${DECX_WORLD_ABS_DIR}/bin/x64/Release)
else()
    if ("${_DECX_HOST_ARCH_} " STREQUAL "x86_64 ")
        link_directories(${DECX_WORLD_ABS_DIR}/bin/x64)
    elseif ("${_DECX_HOST_ARCH_} " STREQUAL "aarch64 ")
        link_directories(${DECX_WORLD_ABS_DIR}/bin/aarch64)
    endif()
endif()

# add_library(DECX_BLAS_CPU SHARED ${GEMM}                ${BP} 
#                                  ${EXT_CPU_COM_SRCS}    ${FMGR_CPU_COM_SRCS}
#                                  ${FILL_CPU_COM_SRCS}   ${TRP_CPU_COM_SRCS} 
#                                  ${TYPECAST_CPU_COM_SRCS})



add_library(DECX_BLAS_CPU SHARED ${GEMM}                ${BP} 
                                 ${EXT_CPU_COM_SRCS}    ${FMGR_CPU_COM_SRCS})


target_link_libraries(DECX_BLAS_CPU PRIVATE gemm_fp32_cpu
                                    PRIVATE gemm_64b_cpu)
                                    # PRIVATE gemm_cplxd_cpu)

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_link_libraries(DECX_BLAS_CPU PUBLIC DECX_core_CPU.lib)

else()
    target_link_libraries(DECX_BLAS_CPU PUBLIC DECX_core_CPU.so)
    
endif()


else()
message("Either _CPP_EXPORT_ or _C_EXPORT_ should be true to build the library. Configuration terminated")
endif()         # if(_CPP_EXPORT_ OR _C_EXPORT_)