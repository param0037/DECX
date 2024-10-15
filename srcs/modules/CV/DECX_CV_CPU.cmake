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

include_directories(${DECX_WORLD_ABS_DIR}/srcs/extern/SDL2)
include_directories(${DECX_WORLD_ABS_DIR}/srcs/extern/SDL2_image)


file(GLOB_RECURSE CV_CORE "${DECX_WORLD_ABS_DIR}/srcs/modules/CV/Image_IO/*.cxx"
                          "${DECX_WORLD_ABS_DIR}/srcs/modules/CV/edge_detection/*.cxx"
                          "${DECX_WORLD_ABS_DIR}/srcs/modules/CV/utils/*.cxx")
                          
file(GLOB_RECURSE UTILS "${DECX_WORLD_ABS_DIR}/srcs/common/Basic_process/type_statistics/CPU/cmp_max_exec.cxx"
                        "${DECX_WORLD_ABS_DIR}/srcs/common/Basic_process/type_statistics/CPU/cmp_min_exec.cxx"
                        "${DECX_WORLD_ABS_DIR}/srcs/modules/basic_calculations/operators/Maprange_exec.cxx")
message(STATUS "SDL is found in path : ${SDL_PATH}")

if(${HOST_OS_NAME} STREQUAL "Windows")
    link_directories(${SDL_PATH}/lib/x64
                    ${SDL_IMAGE_PATH}/lib/x64)

    add_library(DECX_CV_CPU SHARED ${CV_CORE} ${UTILS} ${FMGR_CPU_COM_SRCS})
    target_link_libraries(DECX_CV_CPU SDL2.lib SDL2main.lib SDL2test.lib SDL2_image.lib DECX_core_CPU.lib DECX_BLAS_CPU.lib)
    
elseif(${HOST_OS_NAME} STREQUAL "Linux")
    link_directories(${SDL_PATH}/lib
    		     ${SDL_IMAGE_PATH}/lib)

    add_library(DECX_CV_CPU SHARED ${CV_CORE} ${UTILS} ${FMGR_CPU_COM_SRCS})
    target_link_libraries(DECX_CV_CPU libSDL2-2.0.so.0 libSDL2_image-2.0.so.0 DECX_core_CPU.so DECX_BLAS_CPU.so)
endif()
