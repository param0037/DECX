full_path=$(realpath $0)

PROJECT_PATH_CONFIG=$(dirname $full_path)

cd $PROJECT_PATH_CONFIG

cmake -B build -G"Unix Makefiles"

cmake --build build --config Release

rm -f /home/wayne/DECX/libs/x64/libDECX_core_CUDA.so
cp ../bin/x64/libDECX_core_CUDA.so /home/wayne/DECX/libs/x64/
