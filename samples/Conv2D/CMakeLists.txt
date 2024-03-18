cmake_minimum_required(VERSION 3.23)

project(Conv2D LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# get upper dir
string(REGEX REPLACE "/$" "" CURRENT_FOLDER_ABSOLUTE ${CMAKE_CURRENT_SOURCE_DIR})
get_filename_component(DECX_WORLD_ABS_DIR ${CURRENT_FOLDER_ABSOLUTE} DIRECTORY)

include_directories("../../APIs")

file(GLOB_RECURSE CXX_SOURCE "Conv2D.cpp")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")


if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE release CACHE STRING "Build Type" FORCE)
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    link_directories("../../bin/x64/Release")

elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    link_directories("../../bin/x64")
    
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    message(STATUS "Building on macOS")
else()
    message(STATUS "Building on unknown platform")
endif()

set(EXECUTABLE_OUTPUT_PATH "../../x64")
add_executable(Conv2D ${CXX_SOURCE})

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    message(STATUS "Building on Windows")
    target_link_libraries(Conv2D DECX_core_CPU.lib DECX_core_CUDA.lib DECX_CV_CPU.lib DECX_NN_CUDA.lib
    DECX_NN_CPU.lib)

elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(STATUS "Building on Linux")
    target_link_libraries(Conv2D DECX_core_CPU.so DECX_core_CUDA.so DECX_CV_CPU.so DECX_NN_CUDA.so
    DECX_NN_CPU.so)
    
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    message(STATUS "Building on macOS. Failed, not supported!")
else()
    message(STATUS "Building on unknown platform. Failed, not supported!")
endif()
