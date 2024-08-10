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

project(DECX_core_CPU LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# get upper dir
string(REGEX REPLACE "/$" "" CURRENT_FOLDER_ABSOLUTE ${CMAKE_CURRENT_SOURCE_DIR})
get_filename_component(DECX_WORLD_ABS_DIR ${CURRENT_FOLDER_ABSOLUTE} DIRECTORY)

# Define the path of the binaries built by subdirectories
set(DECX_SUBBUILD_BIN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/${CMAKE_CXX_COMPILER_ID})

# include to add compile definitions to the target
include("${DECX_WORLD_ABS_DIR}/srcs/compile_defs.cmake")

file(GLOB CORE "${DECX_WORLD_ABS_DIR}/srcs/modules/core/*.cxx")
file(GLOB_RECURSE CLASSES "${DECX_WORLD_ABS_DIR}/srcs/modules/core/classes/*.cxx")
file(GLOB_RECURSE RESMGR "${DECX_WORLD_ABS_DIR}/srcs/modules/core/resources_manager/*.cxx")
file(GLOB_RECURSE CONFIGS "${DECX_WORLD_ABS_DIR}/srcs/modules/core/configs/*.cxx")
file(GLOB_RECURSE THREAD_POOL "${DECX_WORLD_ABS_DIR}/srcs/modules/core/thread_management/*.cxx")
add_subdirectory(${DECX_WORLD_ABS_DIR}/srcs/modules/core/allocators/CPU
                 ${DECX_SUBBUILD_BIN_DIR}/allocators_host)


if(CMAKE_CROSSCOMPILING)
    message("Host=aarch64")
    set(LIBRARY_OUTPUT_PATH ${DECX_WORLD_ABS_DIR}/bin/aarch64)
    add_subdirectory("${DECX_WORLD_ABS_DIR}/srcs/modules/core/configs/arm64"
                     "${DECX_SUBBUILD_BIN_DIR}/configs_aarch64_cpu")
else()
    message("Host=x86_64")
    set(LIBRARY_OUTPUT_PATH ${DECX_WORLD_ABS_DIR}/bin/x64)
    add_subdirectory("${DECX_WORLD_ABS_DIR}/srcs/modules/core/configs/x86_64"
                     "${DECX_SUBBUILD_BIN_DIR}/configs_x86_64_cpu")

endif()


if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE release CACHE STRING "Build Type" FORCE)
endif()


add_library(DECX_core_CPU SHARED ${CORE} ${CLASSES} ${RESMGR} ${THREAD_POOL} ${CONFIGS})


target_link_libraries(DECX_core_CPU PRIVATE allocators_host)


if(CMAKE_CROSSCOMPILING)
    target_link_libraries(DECX_core_CPU PRIVATE configs_aarch64_cpu)
else()
    target_link_libraries(DECX_core_CPU PRIVATE configs_x86_64_cpu)
endif()


else()
message("Either _CPP_EXPORT_ or _C_EXPORT_ should be true to build the library. Configuration terminated")
endif()         # if(_CPP_EXPORT_ OR _C_EXPORT_)
