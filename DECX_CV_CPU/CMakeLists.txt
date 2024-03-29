cmake_minimum_required(VERSION 3.23)

project(DECX_CV_CPU LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# get upper dir
string(REGEX REPLACE "/$" "" CURRENT_FOLDER_ABSOLUTE ${CMAKE_CURRENT_SOURCE_DIR})
get_filename_component(DECX_WORLD_ABS_DIR ${CURRENT_FOLDER_ABSOLUTE} DIRECTORY)

file(GLOB_RECURSE CV_CORE "${DECX_WORLD_ABS_DIR}/srcs/modules/CV/*.cxx")
file(GLOB_RECURSE UTILS "${DECX_WORLD_ABS_DIR}/srcs/modules/BLAS/basic_process/type_statistics/CPU/cmp_max_exec.cxx"
                        "${DECX_WORLD_ABS_DIR}/srcs/modules/BLAS/basic_process/type_statistics/CPU/cmp_min_exec.cxx"
                        "${DECX_WORLD_ABS_DIR}/srcs/modules/basic_calculations/operators/Maprange_exec.cxx")

set(CXX_FLAGS -mFMA -mAVX2 -mAVX)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")

add_compile_definitions(_DECX_CPU_PARTS_)
add_compile_definitions(_DECX_CV_CPU_)
add_compile_definitions(_C_EXPORT_ENABLED_=1)
add_compile_definitions(_CPP_EXPORT_ENABLED_=1)


include_directories(${SDL_PATH}/include ${SDL_IMAGE_PATH}/include)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE release CACHE STRING "Build Type" FORCE)
endif()

set(LIBRARY_OUTPUT_PATH ${DECX_WORLD_ABS_DIR}/bin/x64)

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    message(STATUS "Building on Windows")
    set(SDL_PATH "${DECX_WORLD_ABS_DIR}/3rdparty/SDL2/Windows")
    set(SDL_IMAGE_PATH "${DECX_WORLD_ABS_DIR}/3rdparty/SDL2_Image/Windows")
    include_directories(${SDL_PATH}/include ${SDL_IMAGE_PATH}/include)

elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(STATUS "Building on Linux")
    set(SDL_PATH "${DECX_WORLD_ABS_DIR}/3rdparty/SDL2/Linux")
    set(SDL_IMAGE_PATH "${DECX_WORLD_ABS_DIR}/3rdparty/SDL2_Image/Linux")
    include_directories(${SDL_PATH}/include)
    
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    message(STATUS "Building on macOS")
else()
    message(STATUS "Building on unknown platform")
endif()


if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    link_directories(${SDL_PATH}/lib/x64
                    ${SDL_IMAGE_PATH}/lib/x64
                    ${DECX_WORLD_ABS_DIR}/bin/x64/Release
                    ${DECX_WORLD_ABS_DIR}/bin/x64)

    add_library(DECX_CV_CPU SHARED ${CV_CORE} ${UTILS})
    target_link_libraries(DECX_CV_CPU SDL2.lib SDL2main.lib SDL2test.lib SDL2_image.lib DECX_core_CPU.lib DECX_BLAS_CPU.lib)
    
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    link_directories(${SDL_PATH}/lib
    		     ${SDL_IMAGE_PATH}/lib
                 ${DECX_WORLD_ABS_DIR}/bin/x64)

    add_library(DECX_CV_CPU SHARED ${CV_CORE})
    target_link_libraries(DECX_CV_CPU SDL2.so SDL2_image-2.0.so.0.2.3 DECX_core_CPU.so DECX_BLAS_CPU.so)
    
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    message(STATUS "Building on macOS. Failed, not supported!")
else()
    message(STATUS "Building on unknown platform. Failed, not supported!")
endif()