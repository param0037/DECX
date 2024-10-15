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

# Set project root directory (absolute directory)
string(REGEX REPLACE "/$" "" CURRENT_FOLDER_ABSOLUTE ${CMAKE_CURRENT_SOURCE_DIR})
get_filename_component(DECX_WORLD_ABS_DIR__ ${CURRENT_FOLDER_ABSOLUTE} DIRECTORY)
get_filename_component(DECX_WORLD_ABS_DIR_ ${DECX_WORLD_ABS_DIR__} DIRECTORY)
get_filename_component(DECX_WORLD_ABS_DIR ${DECX_WORLD_ABS_DIR_} DIRECTORY)

# Set subdirectory build directory
set(DECX_SUBBUILD_BIN_DIR ${CMAKE_BINARY_DIR}/${CMAKE_CXX_COMPILER_ID}/${_DECX_HOST_ARCH_})


cmake_host_system_information(RESULT HOST_OS_NAME QUERY OS_NAME)
message(STATUS "=========================== Building on host, OS name = ${HOST_OS_NAME} ===========================")


# Set library output path
set(LIBRARY_OUTPUT_PATH ${DECX_WORLD_ABS_DIR}/build/bin/${_DECX_HOST_ARCH_})

if (${CMAKE_PROJECT_NAME} MATCHES "CV")
    set(SDL_PATH ${DECX_WORLD_ABS_DIR}/3rdparty/SDL2/${_DECX_HOST_ARCH_}/${HOST_OS_NAME})
    set(SDL_IMAGE_PATH ${DECX_WORLD_ABS_DIR}/3rdparty/SDL2_Image/${_DECX_HOST_ARCH_}/${HOST_OS_NAME})
endif()