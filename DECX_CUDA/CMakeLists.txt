cmake_minimum_required(VERSION 3.00)

# get upper dir
string(REGEX REPLACE "/$" "" CURRENT_FOLDER_ABSOLUTE ${CMAKE_CURRENT_SOURCE_DIR})
get_filename_component(DECX_WORLD_ABS_DIR ${CURRENT_FOLDER_ABSOLUTE} DIRECTORY)
message(${DECX_WORLD_ABS_DIR})

project(DECX_CUDA LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CUDA_ARCHS ${CUDA_NVCC_FLAGS} "sm_75;compute_75")

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

file(GLOB_RECURSE BASIC_CALCULATIONS "${DECX_WORLD_ABS_DIR}/srcs/modules/basic_calculations/*.cu")
file(GLOB CLASS "${DECX_WORLD_ABS_DIR}/srcs/modules/classes/classes_util.cu")
file(GLOB CONFIGS "${DECX_WORLD_ABS_DIR}/srcs/modules/core/configs/*.cu")
file(GLOB CUSTREAM "${DECX_WORLD_ABS_DIR}/srcs/modules/core/cudaStream_management/*.cu")
file(GLOB GLOBAL_VARS "${DECX_WORLD_ABS_DIR}/srcs/modules/core/global_vars.cu")
file(GLOB UTILS "${DECX_WORLD_ABS_DIR}/srcs/modules/core/utils/*.cu")
file(GLOB_RECURSE BASIC_PROCESS "${DECX_WORLD_ABS_DIR}/srcs/modules/basic_process/*.cu")
file(GLOB_RECURSE CONVOLUTION "${DECX_WORLD_ABS_DIR}/srcs/modules/convolution/CUDA/*.cu")
file(GLOB CV "${DECX_WORLD_ABS_DIR}/srcs/modules/cv/NLM/*.cu")
file(GLOB_RECURSE DOT_PRODUCT "${DECX_WORLD_ABS_DIR}/srcs/modules/Dot product/CUDA/*.cu")
file(GLOB_RECURSE GEMM "${DECX_WORLD_ABS_DIR}/srcs/modules/GEMM/*.cu")
file(GLOB_RECURSE REGIONAL "${DECX_WORLD_ABS_DIR}/srcs/modules/regional/regional_comparision/*.cu")
file(GLOB_RECURSE SIGNAL "${DECX_WORLD_ABS_DIR}/srcs/modules/signal/*.cu")

add_compile_definitions(_DECX_CUDA_CODES_)

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE release CACHE STRING "Build Type" FORCE)
endif()

set(LIBRARY_OUTPUT_PATH "${DECX_WORLD_ABS_DIR}/bin/x64")


if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    link_directories(${DECX_WORLD_ABS_DIR}/bin/x64/Release)

elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    link_directories(${DECX_WORLD_ABS_DIR}/bin/x64)
    
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    message(STATUS "Building on macOS")
else()
    message(STATUS "Building on unknown platform")
endif()

add_library(DECX_CUDA SHARED ${GLOBAL_VARS} ${CONFIGS} ${CUSTREAM} ${UTILS}
                            ${CLASS}
                            ${BASIC_CALCULATIONS} 
                            ${BASIC_PROCESS}
                            ${CONVOLUTION}
                            ${CV}
                            ${DOT_PRODUCT}
                            ${GEMM}
                            ${REGIONAL}
                            ${SIGNAL})

set_target_properties(DECX_CUDA PROPERTIES CUDA_ARCHITECTURES "75")

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    message(STATUS "Building on Windows")
    target_link_libraries(DECX_CUDA DECX_allocation.lib DECX_classes.lib)

elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(STATUS "Building on Linux")
    target_link_libraries(DECX_CUDA DECX_allocation.so DECX_classes.so)
    
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    message(STATUS "Building on macOS")
else()
    message(STATUS "Building on unknown platform")
endif()
