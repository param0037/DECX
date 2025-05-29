cmake_minimum_required(VERSION 3.10)

project(test)

enable_language(ASM_NASM)

include_directories("/home/wayne/Documents/DECX/srcs/includes")
include_directories("/usr/local/include/opencv4")
link_directories("/home/wayne/Documents/DECX/build/bin/x64")
link_directories("/usr/local/lib")

add_executable(test "test_func.asm" "./test.cpp")

set(POSITION_INDEPENDENT_CODE true)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")

target_link_libraries(test DECX_core_CPU DECX_core_CUDA 
DECX_BLAS_CPU 
DECX_BLAS_CUDA 
DECX_DSP_CPU 
# DECX_DSP_CUDA
# SDL2-2.0 
# SDL2_image-2.0
#libopencv_imgproc.so 
#libopencv_core.so
#libopencv_highgui.so
#libopencv_imgcodecs.so)
DECX_CV_CPU)

