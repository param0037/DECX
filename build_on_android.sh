#!/bin/bash

full_path=$(realpath $0)
PROJECT_PATH_BUILD=$(dirname $full_path)

cd "$PROJECT_PATH_BUILD/DECX_BLAS_CPU"
cmake --build build --config Release

cp $PROJECT_PATH_BUILD/bin/aarch64/libDECX_BLAS_CPU.so ~/AndroidStudioProjects/MyApplication/app/src/main/jniLibs/arm64-v8a
