#!/bin/bash

echo $0

full_path=$(realpath $0)

PROJECT_PATH_BUILD=$(dirname $full_path)

cd $PROJECT_PATH_BUILD/DECX_allocation
cmake --build build --config Release

cd $PROJECT_PATH_BUILD/DECX_classes
cmake --build build --config Release

cd $PROJECT_PATH_BUILD/DECX_Image
cmake --build build --config Release

cd $PROJECT_PATH_BUILD/DECX_cpu
cmake --build build --config Release


cd $PROJECT_PATH_BUILD/DECX_CUDA
cmake --build build --config Release

cd $PROJECT_PATH_BUILD

echo "All build success"
