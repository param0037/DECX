cmake_minimum_required(VERSION 3.23)

project(DECX_core_CPU LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# get upper dir
string(REGEX REPLACE "/$" "" CURRENT_FOLDER_ABSOLUTE ${CMAKE_CURRENT_SOURCE_DIR})
get_filename_component(DECX_WORLD_ABS_DIR ${CURRENT_FOLDER_ABSOLUTE} DIRECTORY)

file(GLOB_RECURSE CLASS "${DECX_WORLD_ABS_DIR}/srcs/modules/classes/*.cxx")
file(GLOB_RECURSE CORE "${DECX_WORLD_ABS_DIR}/srcs/modules/core/*.cxx")

#set(CXX_FLAGS -mFMA -mAVX2 -mAVX)
if(CMAKE_CROSSCOMPILING)
    add_compile_definitions(__aarch64__)
    message("Host=aarch64")
    set(LIBRARY_OUTPUT_PATH ${DECX_WORLD_ABS_DIR}/bin/aarch64)
else()
    add_compile_definitions(__x86_64__)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
    message("Host=x86_64")
    set(LIBRARY_OUTPUT_PATH ${DECX_WORLD_ABS_DIR}/bin/x64)
endif()
# add_compile_definitions(_DECX_CPU_CODES_)
add_compile_definitions(_DECX_CPU_PARTS_)
add_compile_definitions(_DECX_CORE_CPU_)
add_compile_definitions(_C_EXPORT_ENABLED_=1)
add_compile_definitions(_CPP_EXPORT_ENABLED_=1)


#include_directories(${SDL_PATH}/include ${SDL_IMAGE_PATH}/include)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE release CACHE STRING "Build Type" FORCE)
endif()


if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    link_directories(${DECX_WORLD_ABS_DIR}/bin/x64/Release)

elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    link_directories(${DECX_WORLD_ABS_DIR}/bin/x64)
    
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    message(STATUS "Building on macOS")
else()
    message(STATUS "Building on unknown platform")
endif()

add_library(DECX_core_CPU SHARED ${CORE} ${CLASS})