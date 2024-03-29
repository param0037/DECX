cmake_minimum_required(VERSION 3.23)

project(DECX_Async_Engine LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# get upper dir
string(REGEX REPLACE "/$" "" CURRENT_FOLDER_ABSOLUTE ${CMAKE_CURRENT_SOURCE_DIR})
get_filename_component(DECX_WORLD_ABS_DIR ${CURRENT_FOLDER_ABSOLUTE} DIRECTORY)

file(GLOB_RECURSE SRCS "${DECX_WORLD_ABS_DIR}/srcs/Async Engine/*.cxx")
file(GLOB GLOBAL_VARS "${DECX_WORLD_ABS_DIR}/srcs/modules/core/global_vars.cxx")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
add_compile_definitions(_DECX_ASYNC_CORE_)
add_compile_definitions(_C_EXPORT_ENABLED_=1)
add_compile_definitions(_CPP_EXPORT_ENABLED_=1)


if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE release CACHE STRING "Build Type" FORCE)
endif()

set(LIBRARY_OUTPUT_PATH ${DECX_WORLD_ABS_DIR}/bin/x64)


add_library(DECX_Async_Engine SHARED ${SRCS} ${GLOBAL_VARS})
