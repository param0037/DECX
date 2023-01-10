cmake_minimum_required(VERSION 3.00)

# get upper dir
string(REGEX REPLACE "/$" "" CURRENT_FOLDER_ABSOLUTE ${CMAKE_CURRENT_SOURCE_DIR})
get_filename_component(DECX_WORLD_ABS_DIR ${CURRENT_FOLDER_ABSOLUTE} DIRECTORY)
message(${DECX_WORLD_ABS_DIR})

project(DECX_classes LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_definitions(_DECX_CLASSES_CODES_)

aux_source_directory(${DECX_WORLD_ABS_DIR}/srcs/modules/classes CLASSES_SRC_FILES)
list(REMOVE_ITEM CLASSES_SRC_FILES "${DECX_WORLD_ABS_DIR}/srcs/modules/classes/classes_util.cxx")
list(REMOVE_ITEM CLASSES_SRC_FILES "${DECX_WORLD_ABS_DIR}/srcs/modules/classes/classes_util.cu")
aux_source_directory(${DECX_WORLD_ABS_DIR}/srcs/modules/core/cudaStream_management CUSTREAM_SRC_FILES)

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE release CACHE STRING "Build Type" FORCE)
endif()

set(LIBRARY_OUTPUT_PATH "${DECX_WORLD_ABS_DIR}/bin/x64")

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    link_directories(${DECX_WORLD_ABS_DIR}/bin/x64/Release)

elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    link_directories(${DECX_WORLD_ABS_DIR}/bin/x64)
    
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
else()
endif()

add_library(DECX_classes SHARED ${CUSTREAM_SRC_FILES} ${CLASSES_SRC_FILES})

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    message(STATUS "Building on Windows")
    target_link_libraries(DECX_classes DECX_allocation.lib)

elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(STATUS "Building on Linux")
    target_link_libraries(DECX_classes DECX_allocation.so)
    
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    message(STATUS "Building on macOS")
else()
    message(STATUS "Building on unknown platform")
endif()
