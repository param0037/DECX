cmake_minimum_required(VERSION 3.00)


set(_CPP_EXPORT_ CACHE BOOL "To control if this module export functions in CPP mode")
set(_C_EXPORT_ CACHE BOOL "To control if this module export functions in C mode")
set(_DECX_HOST_ARCH_ "x86_64" CACHE STRING "Specfiy the host architecture (either x86_64 or aarch64" FORCE)

if(_CPP_EXPORT_ OR _C_EXPORT_)

# get upper dir
string(REGEX REPLACE "/$" "" CURRENT_FOLDER_ABSOLUTE ${CMAKE_CURRENT_SOURCE_DIR})
get_filename_component(DECX_WORLD_ABS_DIR ${CURRENT_FOLDER_ABSOLUTE} DIRECTORY)
message(${DECX_WORLD_ABS_DIR})

project(DECX_DSP_CUDA LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CUDA_TARGET_ARCH "75")
set(CUDA_ARCH_COMPILE_FLAGS "sm_75;compute_75")


# Define the path of the binaries built by subdirectories
set(DECX_SUBBUILD_BIN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/${CMAKE_CXX_COMPILER_ID})


if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    link_directories(${DECX_WORLD_ABS_DIR}/bin/x64/Release)
else()
    link_directories(${DECX_WORLD_ABS_DIR}/bin/x64)
endif()

# include common definitions
include("${DECX_WORLD_ABS_DIR}/srcs/compile_defs.cmake")

file(GLOB FFT_SRCS "${DECX_WORLD_ABS_DIR}/srcs/modules/DSP/FFT/CUDA/*.cu")
file(GLOB CONV_SRCS "${DECX_WORLD_ABS_DIR}/srcs/modules/DSP/convolution/CUDA/*.cu")
add_subdirectory("${DECX_WORLD_ABS_DIR}/srcs/modules/DSP/FFT/CUDA/1D" "${DECX_SUBBUILD_BIN_DIR}/fft1d_cuda")
add_subdirectory("${DECX_WORLD_ABS_DIR}/srcs/modules/DSP/FFT/CUDA/2D" "${DECX_SUBBUILD_BIN_DIR}/fft2d_cuda")
add_subdirectory("${DECX_WORLD_ABS_DIR}/srcs/modules/DSP/FFT/CUDA/3D" "${DECX_SUBBUILD_BIN_DIR}/fft3d_cuda")

add_subdirectory("${DECX_WORLD_ABS_DIR}/srcs/modules/DSP/convolution/CUDA/2D" "${DECX_SUBBUILD_BIN_DIR}/conv2d_cuda")

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE release CACHE STRING "Build Type" FORCE)
endif()

set(LIBRARY_OUTPUT_PATH "${DECX_WORLD_ABS_DIR}/bin/x64")

add_library(DECX_DSP_CUDA SHARED ${FFT_SRCS} ${CONV_SRCS} ${TRP_CUDA_COM_SRCS})


target_link_libraries(DECX_DSP_CUDA PRIVATE fft1d_cuda
                                    PRIVATE fft2d_cuda
                                    PRIVATE fft3d_cuda
                                    PRIVATE conv2d_cuda)


set_target_properties(DECX_DSP_CUDA PROPERTIES CUDA_ARCHITECTURES ${CUDA_TARGET_ARCH})
set_target_properties(DECX_DSP_CUDA PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    message(STATUS "Building on Windows")
    # Must tell NVCC and Host linker to consider multiple implementations of common CUDA source file
    # They will automatically generate symbols of __cudaRegisterLinkBinary with different names.
    target_link_libraries(DECX_DSP_CUDA PUBLIC DECX_BLAS_CUDA.lib)
    target_link_libraries(DECX_DSP_CUDA PUBLIC DECX_core_CUDA.lib)

elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(STATUS "Building on Linux")
    # Must tell NVCC and Host linker to consider multiple implementations of common CUDA source file
    # They will automatically generate symbols of __cudaRegisterLinkBinary with different names.
    target_link_libraries(DECX_DSP_CUDA PUBLIC DECX_BLAS_CUDA.so)
    target_link_libraries(DECX_DSP_CUDA PUBLIC DECX_core_CUDA.so)
    
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
else()
endif()


else()
message("Either _CPP_EXPORT_ or _C_EXPORT_ should be true to build the library. Configuration terminated")
endif()         # if(_CPP_EXPORT_ OR _C_EXPORT_)