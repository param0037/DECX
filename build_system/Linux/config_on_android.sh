#!/bin/bash

full_path=$(realpath $0)
PROJECT_PATH_BUILD=$(dirname $(dirname $(dirname $full_path)))


cd "$PROJECT_PATH_BUILD/DECX_BLAS_CPU"
cmake 	-DCMAKE_TOOLCHAIN_FILE=/usr/Android/Sdk/ndk/26.1.10909125/build/cmake/android.toolchain.cmake \
	-DANDROID_ABI=arm64-v8a	\
	-DANDROID_PLATFORM=android-24	\
	-D_CPP_EXPORT_=true \
	-D_C_EXPORT_=true \
	-D_DECX_HOST_ARCH_=aarch64 \
	-B build -G"Unix Makefiles"
