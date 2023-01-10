#!/bin/bash

echo $0

full_path=$(realpath $0)

PROJECT_PATH_CONFIG=$(dirname $full_path)

cd $PROJECT_PATH_CONFIG/DECX_allocation
cmake -B build -G"Unix Makefiles"

cd $PROJECT_PATH_CONFIG/DECX_classes
cmake -B build -G"Unix Makefiles"

cd $PROJECT_PATH_CONFIG/DECX_Image
cmake -B build -G"Unix Makefiles"


cd $PROJECT_PATH_CONFIG/DECX_cpu
cmake -B build -G"Unix Makefiles"

cd $PROJECT_PATH_CONFIG/DECX_CUDA
cmake -B build -G"Unix Makefiles"

echo "Config done"

cd $PROJECT_PATH_CONFIG
