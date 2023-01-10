cmake_minimum_required(VERSION 3.00)

project(DECX_Image LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# get upper dir
string(REGEX REPLACE "/$" "" CURRENT_FOLDER_ABSOLUTE ${CMAKE_CURRENT_SOURCE_DIR})
get_filename_component(DECX_WORLD_ABS_DIR ${CURRENT_FOLDER_ABSOLUTE} DIRECTORY)


file(GLOB CXX_SOURCES "${DECX_WORLD_ABS_DIR}/srcs/modules/cv/Image_IO/*.cxx")


if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    message(STATUS "Building on Windows")
    set(SDL_PATH "${DECX_WORLD_ABS_DIR}/3rdparty/Windows/SDL/SDL2-devel-2.6.2-VC")
    set(SDL_IMAGE_PATH "${DECX_WORLD_ABS_DIR}/3rdparty/Windows/SDL/SDL2_image-devel-2.6.2-VC")
    include_directories(${SDL_PATH}/include ${SDL_IMAGE_PATH}/include)

elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(STATUS "Building on Linux")
    set(SDL_PATH "${DECX_WORLD_ABS_DIR}/3rdparty/Linux/SDL2")
    set(SDL_IMAGE_PATH "${DECX_WORLD_ABS_DIR}/3rdparty/Linux/SDL2_Image")
    include_directories(${SDL_PATH}/include)
    
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    message(STATUS "Building on macOS")
else()
    message(STATUS "Building on unknown platform")
endif()

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE release CACHE STRING "Build Type" FORCE)
endif()

set(LIBRARY_OUTPUT_PATH ${DECX_WORLD_ABS_DIR}/bin/x64)


if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    link_directories(${SDL_PATH}/lib/x64
                    ${SDL_IMAGE_PATH}/lib/x64
                    ${DECX_WORLD_ABS_DIR}/bin/x64/Release
                    ${DECX_WORLD_ABS_DIR}/bin/x64)

    add_library(DECX_Image SHARED ${CXX_SOURCES})
    target_link_libraries(DECX_Image SDL2.lib SDL2main.lib SDL2test.lib SDL2_image.lib DECX_classes.lib)
    
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    link_directories(${SDL_PATH}/lib
    		     ${SDL_IMAGE_PATH}/lib
                    ${DECX_WORLD_ABS_DIR}/bin/x64)

    add_library(DECX_Image SHARED ${CXX_SOURCES})
    target_link_libraries(DECX_Image SDL2.so SDL2_image-2.0.so.0.2.3 DECX_classes.so)
    
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    message(STATUS "Building on macOS")
else()
    message(STATUS "Building on unknown platform")
endif()


