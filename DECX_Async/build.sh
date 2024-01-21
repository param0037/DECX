full_path=$(realpath $0)

PROJECT_PATH_CONFIG=$(dirname $full_path)

cd $PROJECT_PATH_CONFIG

cmake -B build -G"Unix Makefiles"

cmake --build build --config Release
