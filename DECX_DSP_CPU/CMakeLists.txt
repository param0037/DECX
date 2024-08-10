cmake_minimum_required(VERSION 3.23)

set(_CPP_EXPORT_ CACHE BOOL "To control if this module export functions in CPP mode")
set(_C_EXPORT_ CACHE BOOL "To control if this module export functions in C mode")
set(_DECX_HOST_ARCH_ "x86_64" CACHE STRING "Specfiy the host architecture (either x86_64 or aarch64")

if(_CPP_EXPORT_ OR _C_EXPORT_)

project(DECX_DSP_CPU LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# get upper dir
string(REGEX REPLACE "/$" "" CURRENT_FOLDER_ABSOLUTE ${CMAKE_CURRENT_SOURCE_DIR})
get_filename_component(DECX_WORLD_ABS_DIR ${CURRENT_FOLDER_ABSOLUTE} DIRECTORY)


# Define the path of the binaries built by subdirectories
set(DECX_SUBBUILD_BIN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/${CMAKE_CXX_COMPILER_ID})


# Define the path for the final built binary
if(CMAKE_CROSSCOMPILING)
    message("Host=aarch64")
    set(LIBRARY_OUTPUT_PATH ${DECX_WORLD_ABS_DIR}/bin/aarch64)
else()
    message("Host=x86_64")
    set(LIBRARY_OUTPUT_PATH ${DECX_WORLD_ABS_DIR}/bin/x64)
endif()

# include to add compile definitions to the target
include("${DECX_WORLD_ABS_DIR}/srcs/compile_defs.cmake")

file(GLOB FFT "${DECX_WORLD_ABS_DIR}/srcs/modules/DSP/FFT/CPU/*.cxx")
file(GLOB FILTERS "${DECX_WORLD_ABS_DIR}/srcs/modules/DSP/filters/CPU/*.cxx")
file(GLOB CONV "${DECX_WORLD_ABS_DIR}/srcs/modules/DSP/convolution/CPU/*.cxx")

add_subdirectory("${DECX_WORLD_ABS_DIR}/srcs/modules/DSP/FFT/CPU/1D" "${DECX_SUBBUILD_BIN_DIR}/fft1d_cpu")
add_subdirectory("${DECX_WORLD_ABS_DIR}/srcs/modules/DSP/FFT/CPU/2D" "${DECX_SUBBUILD_BIN_DIR}/fft2d_cpu")
add_subdirectory("${DECX_WORLD_ABS_DIR}/srcs/modules/DSP/FFT/CPU/3D" "${DECX_SUBBUILD_BIN_DIR}/fft3d_cpu")
add_subdirectory("${DECX_WORLD_ABS_DIR}/srcs/modules/DSP/FFT/CPU/FFT_common" "${DECX_SUBBUILD_BIN_DIR}/fft_common_cpu")
add_subdirectory("${DECX_WORLD_ABS_DIR}/srcs/modules/DSP/convolution/CPU/2D" "${DECX_SUBBUILD_BIN_DIR}/conv2d_cpu")


if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE release CACHE STRING "Build Type" FORCE)
endif()


if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    link_directories(${DECX_WORLD_ABS_DIR}/bin/x64/Release)
else()
    link_directories(${DECX_WORLD_ABS_DIR}/bin/x64)
endif()

add_library(DECX_DSP_CPU SHARED ${FFT} 
                                ${FILTERS}
                                ${CONV})

target_link_libraries(DECX_DSP_CPU PRIVATE fft1d_cpu
                                   PRIVATE fft2d_cpu
                                   PRIVATE fft3d_cpu
                                   PRIVATE fft_common_cpu
                                   PRIVATE conv2d_cpu)

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    message(STATUS "Building on Windows")
    target_link_libraries(DECX_DSP_CPU PUBLIC DECX_core_CPU.lib)
    target_link_libraries(DECX_DSP_CPU PUBLIC DECX_BLAS_CPU.lib)

elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(STATUS "Building on Linux")
    target_link_libraries(DECX_DSP_CPU PUBLIC DECX_core_CPU.so)
    target_link_libraries(DECX_DSP_CPU PUBLIC DECX_BLAS_CPU.so)
    
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    message(STATUS "Building on macOS. Failed, not supported!")
else()
    message(STATUS "Building on unknown platform. Failed, not supported!")
endif()


else()
message("Either _CPP_EXPORT_ or _C_EXPORT_ should be true to build the library. Configuration terminated")
endif()         # if(_CPP_EXPORT_ OR _C_EXPORT_)