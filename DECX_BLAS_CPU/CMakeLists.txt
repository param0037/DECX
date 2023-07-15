cmake_minimum_required(VERSION 3.23)

project(DECX_BLAS_CPU LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# get upper dir
string(REGEX REPLACE "/$" "" CURRENT_FOLDER_ABSOLUTE ${CMAKE_CURRENT_SOURCE_DIR})
get_filename_component(DECX_WORLD_ABS_DIR ${CURRENT_FOLDER_ABSOLUTE} DIRECTORY)

file(GLOB_RECURSE BLAS_CORE "${DECX_WORLD_ABS_DIR}/srcs/modules/BLAS/*.cxx")
file(GLOB_RECURSE PARALLEL_CALCS "${DECX_WORLD_ABS_DIR}/srcs/modules/basic_calculations/*.cxx")
file(GLOB_RECURSE THREAD_ARRANGE "${DECX_WORLD_ABS_DIR}/srcs/modules/core/utils/fragment_arangement.cxx")

set(CXX_FLAGS -mFMA -mAVX2 -mAVX)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")

add_compile_definitions(_DECX_CPU_PARTS_)
add_compile_definitions(_DECX_BLAS_CPU_)

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

add_library(DECX_BLAS_CPU SHARED ${BLAS_CORE} ${PARALLEL_CALCS} ${THREAD_ARRANGE})

set_target_properties(DECX_BLAS_CPU PROPERTIES CUDA_ARCHITECTURES "75")

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    message(STATUS "Building on Windows")
    target_link_libraries(DECX_BLAS_CPU DECX_core_CPU.lib)
    target_link_libraries(DECX_BLAS_CPU DECX_Async_Engine.lib)

elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(STATUS "Building on Linux")
    target_link_libraries(DECX_BLAS_CPU DECX_core_CPU.so)
    target_link_libraries(DECX_BLAS_CPU DECX_Async_Engine.so)
    
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    message(STATUS "Building on macOS. Failed, not supported!")
else()
    message(STATUS "Building on unknown platform. Failed, not supported!")
endif()
