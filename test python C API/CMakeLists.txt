cmake_minimum_required(VERSION 3.23)

project(TestCPy LANGUAGES C)

set(Python_ADDITIONAL_VERSIONS 3.7)

# get upper dir
string(REGEX REPLACE "/$" "" CURRENT_FOLDER_ABSOLUTE ${CMAKE_CURRENT_SOURCE_DIR})
get_filename_component(DECX_WORLD_ABS_DIR ${CURRENT_FOLDER_ABSOLUTE} DIRECTORY)

file(GLOB_RECURSE CORE "${DECX_WORLD_ABS_DIR}/test python C API/test.c")

set(PYTHON_LIB_DIR "${DECX_WORLD_ABS_DIR}/3rdparty/Python/3.7")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")

add_compile_definitions(_DECX_CPU_PARTS_)
add_compile_definitions(_DECX_CORE_CPU_)
add_compile_definitions(_C_EXPORT_ENABLED_=1)
add_compile_definitions(_CPP_EXPORT_ENABLED_=1)

include_directories(${PYTHON_LIB_DIR}/include)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE release CACHE STRING "Build Type" FORCE)
endif()

set(LIBRARY_OUTPUT_PATH ${DECX_WORLD_ABS_DIR}/bin/x64)

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    link_directories(${DECX_WORLD_ABS_DIR}/bin/x64/Release)
    link_directories(${PYTHON_LIB_DIR}/libs/Windows)

elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    link_directories(${DECX_WORLD_ABS_DIR}/bin/x64)
    
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    message(STATUS "Building on macOS")
else()
    message(STATUS "Building on unknown platform")
endif()

add_library(TestCPy SHARED ${CORE})

target_link_libraries(TestCPy python3.lib python37.lib)