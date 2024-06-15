cmake_minimum_required(VERSION 3.23)

project(DECX_DSP_CPU LANGUAGES CXX)
if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
enable_language(C ASM_MASM)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")

endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# get upper dir
string(REGEX REPLACE "/$" "" CURRENT_FOLDER_ABSOLUTE ${CMAKE_CURRENT_SOURCE_DIR})
get_filename_component(DECX_WORLD_ABS_DIR ${CURRENT_FOLDER_ABSOLUTE} DIRECTORY)

file(GLOB_RECURSE DSP_CORE "${DECX_WORLD_ABS_DIR}/srcs/modules/DSP/*.cxx" 
                           "${DECX_WORLD_ABS_DIR}/srcs/modules/BLAS/basic_process/transpose/CPU/transpose_exec.cxx")
file(GLOB_RECURSE COMPLEX_OPERATORS "${DECX_WORLD_ABS_DIR}/srcs/modules/basic_calculations/operators/cp_ops_exec.cxx")
file(GLOB_RECURSE SIMD_FAST_MATH "${DECX_WORLD_ABS_DIR}/srcs/modules/core/utils/SSE_fast_math_masm.asm")

set(CXX_FLAGS -mFMA -mAVX2 -mAVX)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")

add_compile_definitions(_DECX_CPU_PARTS_)
add_compile_definitions(_DECX_DSP_CPU_)
add_compile_definitions(_C_EXPORT_ENABLED_=1)
add_compile_definitions(_CPP_EXPORT_ENABLED_=1)
add_compile_definitions(_SIMD_VER_=AVX256)


include_directories(${SDL_PATH}/include ${SDL_IMAGE_PATH}/include)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE release CACHE STRING "Build Type" FORCE)
endif()

set(LIBRARY_OUTPUT_PATH ${DECX_WORLD_ABS_DIR}/bin/x64)

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    link_directories(${DECX_WORLD_ABS_DIR}/bin/x64/Release)

elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    link_directories(${DECX_WORLD_ABS_DIR}/bin/x64)
    
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    message(STATUS "Building on macOS")
else()
    message(STATUS "Building on unknown platform")
endif()

add_library(DECX_DSP_CPU SHARED ${DSP_CORE} ${COMPLEX_OPERATORS} ${SIMD_FAST_MATH})

set_target_properties(DECX_DSP_CPU PROPERTIES CUDA_ARCHITECTURES "75")

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    message(STATUS "Building on Windows")
    target_link_libraries(DECX_DSP_CPU DECX_core_CPU.lib)
    target_link_libraries(DECX_DSP_CPU DECX_BLAS_CPU.lib)

elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(STATUS "Building on Linux")
    target_link_libraries(DECX_DSP_CPU DECX_core_CPU.so)
    target_link_libraries(DECX_DSP_CPU DECX_BLAS_CPU.so)
    
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    message(STATUS "Building on macOS. Failed, not supported!")
else()
    message(STATUS "Building on unknown platform. Failed, not supported!")
endif()
