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


cmake_minimum_required(VERSION 3.10)

set(_DECX_CUDA_MODULE_          CACHE BOOL      "If the current built module is CUDA specific")
set(_CPP_EXPORT_                CACHE BOOL      "To control if this module export functions in CPP mode")
set(_C_EXPORT_                  CACHE BOOL      "To control if this module export functions in C mode")
set(_DECX_HOST_ARCH_ "x86_64"   CACHE STRING    "Specfiy the host architecture (either x86_64 or aarch64")
set(_DECX_CUDA_SM_              CACHE STRING    "Specfiy the CUDA SM architecture")
set(_DECX_CUDA_COMPUTE_         CACHE STRING    "Specfiy the CUDA computability")


if(_CPP_EXPORT_ OR _C_EXPORT_)

if(_DECX_CUDA_MODULE_)
    # Check if the two variables are set
    if(("${_DECX_CUDA_SM_} " STREQUAL " ") OR ("${_DECX_CUDA_COMPUTE_} " STREQUAL " "))
        message("Error! When building CUDA modules, _DECX_CUDA_SM_ and _DECX_CUDA_COMPUTE_ should be set")
    endif()

    project(DECX_DSP_CUDA LANGUAGES CXX CUDA)
    # Set CUDA arch name
    set(CUDA_TARGET_ARCH ${_DECX_CUDA_SM_})
    set(CUDA_ARCH_COMPILE_FLAGS "sm_${_DECX_CUDA_SM_};compute_${_DECX_CUDA_COMPUTE_}")

    # Enable seperate compilation mode
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
else()
    project(DECX_DSP_CPU LANGUAGES CXX)
    set(CMAKE_CXX_STANDARD 17)
endif()

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Configure the paths
include("../../config_paths.cmake")

# include to add compile definitions to the target
include("${DECX_WORLD_ABS_DIR}/srcs/compile_defs.cmake")


if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE release CACHE STRING "Build Type" FORCE)
endif()


# Set the link directories
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    link_directories(${DECX_WORLD_ABS_DIR}/build/bin/${_DECX_HOST_ARCH_}/Release)
else()
    link_directories(${DECX_WORLD_ABS_DIR}/build/bin/${_DECX_HOST_ARCH_})
endif()


include_directories("${DECX_WORLD_ABS_DIR}/srcs/modules/core/")
include_directories("${DECX_WORLD_ABS_DIR}/srcs/common/")

if(_DECX_CUDA_MODULE_)
    include(DECX_DSP_CUDA.cmake)
else()
    include(DECX_DSP_CPU.cmake)
endif()


else()
message("Either _CPP_EXPORT_ or _C_EXPORT_ should be true to build the library. Configuration terminated")
endif()         # if(_CPP_EXPORT_ OR _C_EXPORT_)