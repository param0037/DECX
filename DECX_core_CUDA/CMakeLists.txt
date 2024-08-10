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

cmake_minimum_required(VERSION 3.00)

set(_CPP_EXPORT_ CACHE BOOL "To control if this module export functions in CPP mode")
set(_C_EXPORT_ CACHE BOOL "To control if this module export functions in C mode")
set(_DECX_HOST_ARCH_ "x86_64" CACHE STRING "Specfiy the host architecture (either x86_64 or aarch64" FORCE)

if(_CPP_EXPORT_ OR _C_EXPORT_)

# get upper dir
string(REGEX REPLACE "/$" "" CURRENT_FOLDER_ABSOLUTE ${CMAKE_CURRENT_SOURCE_DIR})
get_filename_component(DECX_WORLD_ABS_DIR ${CURRENT_FOLDER_ABSOLUTE} DIRECTORY)
message("Project absolute directory : {DECX_WORLD_ABS_DIR}")

project(DECX_core_CUDA LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CUDA_TARGET_ARCH "75")
set(CUDA_ARCH_COMPILE_FLAGS "sm_75;compute_75")


# Define the path of the binaries built by subdirectories
set(DECX_SUBBUILD_BIN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/${CMAKE_CXX_COMPILER_ID})


SET(CMAKE_CONFIGURATION_TYPES ${CMAKE_BUILD_TYPE} CACHE STRING "" FORCE)

# Specify the binaries output path for all subdirectories
if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(DECX_SUBBUILD_BIN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/MSVC)

elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(DECX_SUBBUILD_BIN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/GCC)
    
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
else()
endif()

# include to add compile definitions to the target
include("${DECX_WORLD_ABS_DIR}/srcs/compile_defs.cmake")

file(GLOB_RECURSE CORE "${DECX_WORLD_ABS_DIR}/srcs/modules/core/global_vars.cu")
file(GLOB_RECURSE CLASSES "${DECX_WORLD_ABS_DIR}/srcs/modules/core/classes/*.cu")
file(GLOB_RECURSE CUSE "${DECX_WORLD_ABS_DIR}/srcs/modules/core/cudaStream_management/*.cu")
file(GLOB_RECURSE DTR "${DECX_WORLD_ABS_DIR}/srcs/modules/core/data transmission/*.cu")
file(GLOB_RECURSE CONFIGS "${DECX_WORLD_ABS_DIR}/srcs/modules/core/configs/*.cu")
file(GLOB FP16 "${DECX_WORLD_ABS_DIR}/srcs/common/FP16/*.cu")
add_subdirectory("${DECX_WORLD_ABS_DIR}/srcs/modules/core/allocators/CUDA" "${DECX_SUBBUILD_BIN_DIR}/allocators_CUDA")


if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    link_directories(${DECX_WORLD_ABS_DIR}/bin/x64/Release)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    link_directories(${DECX_WORLD_ABS_DIR}/bin/x64)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
else()
endif()

# Specify the final built binary output path
set(LIBRARY_OUTPUT_PATH "${DECX_WORLD_ABS_DIR}/bin/x64")

add_library(DECX_core_CUDA SHARED ${CORE} ${ALLOC} ${CLASSES} ${CUSE} ${DTR} ${CONFIGS} ${FP16})

set_target_properties(DECX_core_CUDA PROPERTIES CUDA_ARCHITECTURES ${CUDA_TARGET_ARCH})
set_target_properties(DECX_core_CUDA PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_link_libraries(DECX_core_CUDA PRIVATE allocators_CUDA)

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    target_link_libraries(DECX_core_CUDA PUBLIC DECX_core_CPU.lib)

elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_libraries(DECX_core_CUDA PUBLIC DECX_core_CPU.so)
    
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
else()
endif()

else()
message("Either _CPP_EXPORT_ or _C_EXPORT_ should be true to build the library. Configuration terminated")
endif()         # if(_CPP_EXPORT_ OR _C_EXPORT_)