#!/bin/bash

full_path=$(realpath $0)
PROJECT_PATH_BUILD=$(dirname $full_path)

cd "$PROJECT_PATH_BUILD/DECX_core_CPU"
cmake --build build --config Release
