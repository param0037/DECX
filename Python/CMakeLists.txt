cmake_minimum_required(VERSION 3.23)

project(decx LANGUAGES C)

# get upper dir
string(REGEX REPLACE "/$" "" CURRENT_FOLDER_ABSOLUTE ${CMAKE_CURRENT_SOURCE_DIR})
get_filename_component(DECX_WORLD_ABS_DIR ${CURRENT_FOLDER_ABSOLUTE} DIRECTORY)


# Python
set(Python_ADDITIONAL_VERSIONS 3.7)
set(PYTHON_LIB_DIR "${DECX_WORLD_ABS_DIR}/3rdparty/Python/3.7")
include_directories(${PYTHON_LIB_DIR}/include)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE release CACHE STRING "Build Type" FORCE)
endif()

add_compile_definitions(_DECX_BUILD_FOR_PYTHON_)

set(LIBRARY_OUTPUT_PATH ${DECX_WORLD_ABS_DIR}/bin/x64)

file(GLOB_RECURSE PY_EXPORT_CORE "${DECX_WORLD_ABS_DIR}/srcs/modules/core/Python/*.c")

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    link_directories(${DECX_WORLD_ABS_DIR}/bin/x64/Release)
    link_directories(${PYTHON_LIB_DIR}/libs/Windows)

elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    message(STATUS "Building on macOS")
else()
    message(STATUS "Building on unknown platform")
endif()


add_library(decx SHARED ${PY_EXPORT_CORE})

target_link_libraries(decx DECX_core_CPU.lib DECX_core_CUDA.lib)
