cmake_minimum_required(VERSION 3.23)

project(DECX_cpu LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# get upper dir
string(REGEX REPLACE "/$" "" CURRENT_FOLDER_ABSOLUTE ${CMAKE_CURRENT_SOURCE_DIR})
get_filename_component(DECX_WORLD_ABS_DIR ${CURRENT_FOLDER_ABSOLUTE} DIRECTORY)

file(GLOB_RECURSE BASIC_CALCULATIONS "${DECX_WORLD_ABS_DIR}/srcs/modules/basic_calculations/*.cxx")
file(GLOB CLASS "${DECX_WORLD_ABS_DIR}/srcs/modules/classes/classes_util.cxx")
file(GLOB CONFIGS "${DECX_WORLD_ABS_DIR}/srcs/modules/core/configs/*.cxx")
file(GLOB THREADING "${DECX_WORLD_ABS_DIR}/srcs/modules/core/thread_management/*.cxx")
file(GLOB GLOBAL_VARS "${DECX_WORLD_ABS_DIR}/srcs/modules/core/global_vars.cxx")
file(GLOB UTILS "${DECX_WORLD_ABS_DIR}/srcs/modules/core/utils/*.cxx")
file(GLOB_RECURSE BASIC_PROCESS "${DECX_WORLD_ABS_DIR}/srcs/modules/basic_process/*.cxx")
file(GLOB_RECURSE CONVOLUTION "${DECX_WORLD_ABS_DIR}/srcs/modules/convolution/CPU/*.cxx")
file(GLOB CV "${DECX_WORLD_ABS_DIR}/srcs/modules/cv/utils/*.cxx")
file(GLOB_RECURSE DOT_PRODUCT "${DECX_WORLD_ABS_DIR}/srcs/modules/Dot product/CPU/*.cxx")
file(GLOB_RECURSE GEMM "${DECX_WORLD_ABS_DIR}/srcs/modules/GEMM/*.cxx")
file(GLOB_RECURSE REGIONAL "${DECX_WORLD_ABS_DIR}/srcs/modules/regional/regional_comparision/*.cxx")
file(GLOB_RECURSE SIGNAL "${DECX_WORLD_ABS_DIR}/srcs/modules/signal/*.cxx")
file(GLOB_RECURSE VECOTRIAL "${DECX_WORLD_ABS_DIR}/srcs/modules/Vectorial/*.cxx")

#set(CXX_FLAGS -mFMA -mAVX2 -mAVX)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
add_compile_definitions(_DECX_CPU_CODES_)

include_directories(${SDL_PATH}/include ${SDL_IMAGE_PATH}/include)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE release CACHE STRING "Build Type" FORCE)
endif()

set(LIBRARY_OUTPUT_PATH ${DECX_WORLD_ABS_DIR}/bin/x64)

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    link_directories(${DECX_WORLD_ABS_DIR}/bin/x64/Release)

elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    link_directories(${DECX_WORLD_ABS_DIR}/bin/x64)
    
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    message(STATUS "Building on macOS")
else()
    message(STATUS "Building on unknown platform")
endif()

add_library(DECX_cpu SHARED ${GLOBAL_VARS} ${CONFIGS} ${THREADING} ${UTILS}
                            ${CLASS}
                            ${BASIC_CALCULATIONS} 
                            ${BASIC_PROCESS}
                            ${CONVOLUTION}
                            ${CV}
                            ${DOT_PRODUCT}
                            ${GEMM}
                            ${REGIONAL}
                            ${SIGNAL}
                            ${VECOTRIAL})

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    message(STATUS "Building on Windows")
    target_link_libraries(DECX_cpu DECX_allocation.lib DECX_classes.lib)

elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(STATUS "Building on Linux")
    target_link_libraries(DECX_cpu DECX_allocation.so DECX_classes.so)
    
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    message(STATUS "Building on macOS")
else()
    message(STATUS "Building on unknown platform")
endif()
