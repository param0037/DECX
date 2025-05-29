rm -rf ./build

cmake -B build -G "Unix Makefiles"

cmake --build build

cp ./build/test ./
