#!/bin/bash


full_path=$(realpath $0)
PROJECT_PATH_BUILD=$(dirname $(dirname $(dirname $full_path)))

echo $PROJECT_PATH_BUILD
