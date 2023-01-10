cmake_minimum_required(VERSION 3.00)

# get upper dir
string(REGEX REPLACE "/$" "" CURRENT_FOLDER_ABSOLUTE ${CMAKE_CURRENT_SOURCE_DIR})
get_filename_component(DECX_WORLD_ABS_DIR ${CURRENT_FOLDER_ABSOLUTE} DIRECTORY)
message(${DECX_WORLD_ABS_DIR})

enable_language(CUDA)
project(DECX_allocation LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB CUDA_SOURCES "${DECX_WORLD_ABS_DIR}/srcs/modules/core/memory_management/*.cu")
#file(GLOB CXX_SOURCES "${DECX_WORLD_ABS_DIR}/srcs/modules/core/memory_management/*.cxx")
file(GLOB ALLOCATORS "${DECX_WORLD_ABS_DIR}/srcs/modules/core/allocators/*.cu")

add_compile_definitions(_DECX_ALLOC_CODES_)

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE release CACHE STRING "Build Type" FORCE)
endif()

add_library(DECX_allocation SHARED ${CUDA_SOURCES} ${CXX_SOURCES} ${ALLOCATORS})


set(LIBRARY_OUTPUT_PATH "${DECX_WORLD_ABS_DIR}/bin/x64")
