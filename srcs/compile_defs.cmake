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

# This file must not be included after add_library() or any argument that has created the target
# Since this file set the compile definitions and compiler command line flags, which need to be
# specified before generating the compile command line

include_guard(GLOBAL)

# Set the build type to release mode
if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE release CACHE STRING "Build Type" FORCE)
endif()


set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Determine if export CPP symbols (Can only be called in CPP environment)
if(_CPP_EXPORT_)
    add_compile_definitions(_CPP_EXPORT_ENABLED_=1)
else()
    add_compile_definitions(_CPP_EXPORT_ENABLED_=0)
endif()

# Determine if export C symbols (Can be called in both C and CPP environments)
if(_C_EXPORT_)
    add_compile_definitions(_C_EXPORT_ENABLED_=1)
else()
    add_compile_definitions(_C_EXPORT_ENABLED_=0)
endif()


if("${_DECX_HOST_ARCH_} " STREQUAL "x86_64 ")
    add_compile_definitions(__x86_64__)
elseif("${_DECX_HOST_ARCH_} " STREQUAL "aarch64 ")
    add_compile_definitions(__aarch64__)
endif()

# Set common compile definitions and settings for CUDA context
if (${CMAKE_PROJECT_NAME} MATCHES "CUDA")
    set(CUDA_ARCHS ${CUDA_NVCC_FLAGS} ${CUDA_ARCH_COMPILE_FLAGS})
    
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

    add_compile_definitions(_DECX_CUDA_PARTS_)

    set(CUDA_TARGET_ARCH "75")
    set(CUDA_ARCH_COMPILE_FLAGS "sm_75;compute_75")

    set(DECX_BUILD_WITH_CUDA "Y")
endif()

if (${CMAKE_PROJECT_NAME} MATCHES "CPU")
    add_compile_definitions(_DECX_CPU_PARTS_)
    message("add_compile_definitions(_DECX_CPU_PARTS_)")
    if ("${_DECX_HOST_ARCH_} " STREQUAL "x86_64 ")
        set(CXX_FLAGS -mfma -mavx2 -mavx)
    endif()

    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
    endif()
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    # This is necessary since this project contains lots of common sources,
    # this helps hide the common implementations in the binaries.
    add_compile_options($<$<OR:$<COMPILE_LANGUAGE:C>,$<COMPILE_LANGUAGE:CXX>,$<COMPILE_LANGUAGE:CUDA>>:-fvisibility=hidden>)
endif()

# Add compile definition according to each module's attribute
if("${CMAKE_PROJECT_NAME} " STREQUAL "DECX_core_CPU ")
    add_compile_definitions(_DECX_CORE_CPU_)

elseif("${CMAKE_PROJECT_NAME} " STREQUAL "DECX_core_CUDA ")
    add_compile_definitions(_DECX_CORE_CUDA_)
    
elseif("${CMAKE_PROJECT_NAME} " STREQUAL "DECX_BLAS_CPU ")
    add_compile_definitions(_DECX_BLAS_CPU_)

elseif("${CMAKE_PROJECT_NAME} " STREQUAL "DECX_BLAS_CUDA ")
    add_compile_definitions(_DECX_BLAS_CUDA_)

elseif("${CMAKE_PROJECT_NAME} " STREQUAL "DECX_DSP_CPU ")
    add_compile_definitions(_DECX_DSP_CPU_)

elseif("${CMAKE_PROJECT_NAME} " STREQUAL "DECX_DSP_CUDA ")
    add_compile_definitions(_DECX_DSP_CUDA_)

elseif("${CMAKE_PROJECT_NAME} " STREQUAL "DECX_NN_CPU ")
    add_compile_definitions(_DECX_NN_CPU_)

elseif("${CMAKE_PROJECT_NAME} " STREQUAL "DECX_NN_CUDA ")
    add_compile_definitions(_DECX_NN_CUDA_)

endif()
