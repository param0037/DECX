cmake_minimum_required(VERSION 3.00)

# get upper dir
string(REGEX REPLACE "/$" "" CURRENT_FOLDER_ABSOLUTE ${CMAKE_CURRENT_SOURCE_DIR})
get_filename_component(DECX_WORLD_ABS_DIR ${CURRENT_FOLDER_ABSOLUTE} DIRECTORY)
message(${DECX_WORLD_ABS_DIR})

project(DECX_BLAS_CUDA LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CUDA_ARCHS ${CUDA_NVCC_FLAGS} "sm_75;compute_75")

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

file(GLOB_RECURSE PARALLEL_CALCS "${DECX_WORLD_ABS_DIR}/srcs/modules/basic_calculations/*.cu")
file(GLOB_RECURSE BLAS_CORE "${DECX_WORLD_ABS_DIR}/srcs/modules/BLAS/*.cu")


add_compile_definitions(_DECX_CUDA_PARTS_)
add_compile_definitions(_DECX_BLAS_CUDA_)
add_compile_definitions(_C_EXPORT_ENABLED_=1)
add_compile_definitions(_CPP_EXPORT_ENABLED_=1)


if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE release CACHE STRING "Build Type" FORCE)
endif()

set(LIBRARY_OUTPUT_PATH "${DECX_WORLD_ABS_DIR}/bin/x64")


if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    link_directories(${DECX_WORLD_ABS_DIR}/bin/x64/Release)

elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    link_directories(${DECX_WORLD_ABS_DIR}/bin/x64)
    
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    message(STATUS "Building on macOS. Failed, not supported!")
else()
    message(STATUS "Building on unknown platform. Failed, not supported!")
endif()

add_library(DECX_BLAS_CUDA SHARED ${PARALLEL_CALCS} ${BLAS_CORE})

set_target_properties(DECX_BLAS_CUDA PROPERTIES CUDA_ARCHITECTURES "75")

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    message(STATUS "Building on Windows")
    target_link_libraries(DECX_BLAS_CUDA DECX_core_CPU.lib)
    target_link_libraries(DECX_BLAS_CUDA DECX_core_CUDA.lib)
    target_link_libraries(DECX_BLAS_CUDA DECX_Async_Engine.lib)

elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(STATUS "Building on Linux")
    target_link_libraries(DECX_BLAS_CUDA DECX_core_CPU.so)
    target_link_libraries(DECX_BLAS_CUDA DECX_core_CUDA.so)
    target_link_libraries(DECX_BLAS_CUDA DECX_Async_Engine.so)
    
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    message(STATUS "Building on macOS. Failed, not supported!")
else()
    message(STATUS "Building on unknown platform. Failed, not supported!")
endif()
