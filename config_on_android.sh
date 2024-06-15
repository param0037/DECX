#!/bin/bash

full_path=$(realpath $0)
PROJECT_PATH_BUILD=$(dirname $full_path)

cd "$PROJECT_PATH_BUILD/DECX_core_CPU"
cmake 	-DCMAKE_TOOLCHAIN_FILE=/usr/Android/Sdk/ndk/26.1.10909125/build/cmake/android.toolchain.cmake \
	-DANDROID_ABI=arm64-v8a	\
	-DANDROID_PLATFORM=android-24	\
	-B build -G"Unix Makefiles"
