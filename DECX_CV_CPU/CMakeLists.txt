cmake_minimum_required(VERSION 3.23)


set(_CPP_EXPORT_ CACHE BOOL "To control if this module export functions in CPP mode")
set(_C_EXPORT_ CACHE BOOL "To control if this module export functions in C mode")
set(_DECX_HOST_ARCH_ "x86_64" CACHE STRING "Specfiy the host architecture (either x86_64 or aarch64")

if(_CPP_EXPORT_ OR _C_EXPORT_)

project(DECX_CV_CPU LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# get upper dir
string(REGEX REPLACE "/$" "" CURRENT_FOLDER_ABSOLUTE ${CMAKE_CURRENT_SOURCE_DIR})
get_filename_component(DECX_WORLD_ABS_DIR ${CURRENT_FOLDER_ABSOLUTE} DIRECTORY)

# Define the path of the binaries built by subdirectories
if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(DECX_SUBBUILD_BIN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/MSVC)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(DECX_SUBBUILD_BIN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/GCC)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
else()
endif()

# Define the path for the final built binary
if(CMAKE_CROSSCOMPILING)
    message("Host=aarch64")
    set(LIBRARY_OUTPUT_PATH ${DECX_WORLD_ABS_DIR}/bin/aarch64)
else()
    message("Host=x86_64")
    set(LIBRARY_OUTPUT_PATH ${DECX_WORLD_ABS_DIR}/bin/x64)
endif()


include_directories(${DECX_WORLD_ABS_DIR}/srcs/extern/SDL2)
include_directories(${DECX_WORLD_ABS_DIR}/srcs/extern/SDL2_image)

# include to add compile definitions to the target
include("${DECX_WORLD_ABS_DIR}/srcs/compile_defs.cmake")

# file(GLOB_RECURSE CV_CORE "${DECX_WORLD_ABS_DIR}/srcs/modules/CV/*.cxx")
file(GLOB_RECURSE CV_CORE "${DECX_WORLD_ABS_DIR}/srcs/modules/CV/Image_IO/*.cxx"
                          "${DECX_WORLD_ABS_DIR}/srcs/modules/CV/edge_detection/*.cxx"
                          "${DECX_WORLD_ABS_DIR}/srcs/modules/CV/utils/*.cxx")
                          
file(GLOB_RECURSE UTILS "${DECX_WORLD_ABS_DIR}/srcs/common/Basic_process/type_statistics/CPU/cmp_max_exec.cxx"
                        "${DECX_WORLD_ABS_DIR}/srcs/common/Basic_process/type_statistics/CPU/cmp_min_exec.cxx"
                        "${DECX_WORLD_ABS_DIR}/srcs/modules/basic_calculations/operators/Maprange_exec.cxx")


if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE release CACHE STRING "Build Type" FORCE)
endif()

set(LIBRARY_OUTPUT_PATH ${DECX_WORLD_ABS_DIR}/bin/x64)


if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    link_directories(${SDL_PATH}/lib/x64
                    ${SDL_IMAGE_PATH}/lib/x64
                    ${DECX_WORLD_ABS_DIR}/bin/x64/Release)

    add_library(DECX_CV_CPU SHARED ${CV_CORE})
    target_link_libraries(DECX_CV_CPU SDL2.lib SDL2main.lib SDL2test.lib SDL2_image.lib DECX_core_CPU.lib DECX_BLAS_CPU.lib)
    
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    link_directories(${SDL_PATH}/lib
    		     ${SDL_IMAGE_PATH}/lib
                 ${DECX_WORLD_ABS_DIR}/bin/x64)

    add_library(DECX_CV_CPU SHARED ${CV_CORE})
    target_link_libraries(DECX_CV_CPU SDL2.so SDL2_image-2.0.so.0.2.3 DECX_core_CPU.so DECX_BLAS_CPU.so)
    
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
else()
endif()


else()
message("Either _CPP_EXPORT_ or _C_EXPORT_ should be true to build the library. Configuration terminated")
endif()         # if(_CPP_EXPORT_ OR _C_EXPORT_)