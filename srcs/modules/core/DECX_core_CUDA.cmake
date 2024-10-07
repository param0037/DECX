#   ----------------------------------------------------------------------------------
#   Author : Wayne Anderson
#   Date   : 2021.04.16
#   ----------------------------------------------------------------------------------
# 
# This is a part of the open source project named "DECX", a high-performance scientific
# computational library. This project follows the MIT License. For more information 
# please visit https://github.com/param0037/DECX.
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

file(GLOB_RECURSE   CORE    "${DECX_WORLD_ABS_DIR}/srcs/modules/core/global_vars.cu")
file(GLOB_RECURSE   CLASSES "${DECX_WORLD_ABS_DIR}/srcs/modules/core/classes/*.cu")
file(GLOB_RECURSE   CUSE    "${DECX_WORLD_ABS_DIR}/srcs/modules/core/cudaStream_management/*.cu")
file(GLOB_RECURSE   DTR     "${DECX_WORLD_ABS_DIR}/srcs/modules/core/data transmission/*.cu")
file(GLOB_RECURSE   CONFIGS "${DECX_WORLD_ABS_DIR}/srcs/modules/core/configs/*.cu")
file(GLOB           FP16    "${DECX_WORLD_ABS_DIR}/srcs/common/FP16/*.cu")

add_subdirectory("${DECX_WORLD_ABS_DIR}/srcs/modules/core/allocators/CUDA" 
                 "${DECX_SUBBUILD_BIN_DIR}/allocators_CUDA")

# Specify the link directory
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    link_directories(${LIBRARY_OUTPUT_PATH}/Release)
else()
    link_directories(${LIBRARY_OUTPUT_PATH})
    message(${LIBRARY_OUTPUT_PATH})
endif()


add_library(${PROJECT_NAME} SHARED ${CORE} ${ALLOC} ${CLASSES} ${CUSE} ${DTR} ${CONFIGS} ${FP16})


set_target_properties(DECX_core_CUDA PROPERTIES CUDA_ARCHITECTURES ${CUDA_TARGET_ARCH})
set_target_properties(DECX_core_CUDA PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)


target_link_libraries(DECX_core_CUDA PRIVATE allocators_CUDA)

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    target_link_libraries(DECX_core_CUDA PUBLIC DECX_core_CPU.lib)

elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_libraries(DECX_core_CUDA PUBLIC DECX_core_CPU.so)
else()
endif()