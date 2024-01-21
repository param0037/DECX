#!/bin/bash

full_path=$(realpath $0)
PROJECT_PATH_BUILD=$(dirname $full_path)

function printf_info()
{
    echo "build.sh -[action] [project_name]"
    echo "actions : "
    echo "         -c Clean the project(s), specify which project to be cleaned"
    echo "         -i Configure the project(s), specify which project to be configured"
    echo "         -m Build the project(s), specify which project to be built"
    echo "         -all Clean, configure or build all the projects"
    echo "project_names : "
    echo "         Async, core_CPU(CUDA), BLAS_CPU(CUDA), DSP_CPU(CUDA), CV_CPU(CUDA), NN_CPU)CUDA)"
}


# clean
function clean_single()
{
    if [ -d "$PROJECT_PATH_BUILD/DECX_"$1"/build/" ]; then
        rm -rf "$PROJECT_PATH_BUILD/DECX_"$1"/build/"
    else
        echo "File not exist"
    fi

    if [ -d "$PROJECT_PATH_BUILD/DECX_"$1"/x64/" ]; then
        rm -rf "$PROJECT_PATH_BUILD/DECX_"$1"/x64/"
    else
        echo "File not exist"
    fi
}



function clean_all()
{
    clean_single "core_CPU"
    clean_single "core_CUDA"
    clean_single "BLAS_CPU"
    clean_single "BLAS_CUDA"
    clean_single "DSP_CPU"
    clean_single "DSP_CUDA"
    clean_single "CV_CPU"
    clean_single "CV_CUDA"
    clean_single "NN_CPU"
    clean_single "NN_CUDA"
    clean_single "Async"

    cd $PROJECT_PATH_BUILD

    echo "Cleaned all built"
}


function clean_optional()
{
    if [ "$1" = "all" ]; then
        clean_all
    else
        clean_single $1
    fi
}

# config

function config_single()
{
    cd "$PROJECT_PATH_BUILD/DECX_$1"
    echo "$PROJECT_PATH_BUILD/DECX_$1"
    cmake -B build -G"Unix Makefiles"
}


function config_all()
{
    config_single "Async"
    config_single "core_CPU"
    config_single "core_CUDA"
    config_single "BLAS_CPU"
    config_single "BLAS_CUDA"
    config_single "DSP_CPU"
    config_single "DSP_CUDA"
    config_single "CV_CPU"
    config_single "CV_CUDA"
    config_single "NN_CPU"
    config_single "NN_CUDA"

    cd $PROJECT_PATH_BUILD

    echo "All config success"
}


function config_optional()
{
    if [ "$1" = "all" ]; then
        config_all
    else
        config_single $1
    fi
}

# build

function build_single()
{
    cd "$PROJECT_PATH_BUILD/DECX_"$1
    cmake --build build --config Release
}


function build_all()
{
    build_single "core_CPU"
    build_single "core_CUDA"
    build_single "BLAS_CPU"
    build_single "BLAS_CUDA"
    build_single "DSP_CPU"
    build_single "DSP_CUDA"
    build_single "CV_CPU"
    build_single "CV_CUDA"
    build_single "NN_CPU"
    build_single "NN_CUDA"
    build_single "Async"

    cd $PROJECT_PATH_BUILD

    echo "All build success"
}


function build_optional()
{
    if [ "$1" = "all" ]; then
        build_all
    else
        build_single $1
    fi
}


if [ "$#" -eq 0 ]; then
    echo "Require parameters"
    printf_info
    exit 1
fi


while getopts ":m:c:i:" opt; do
    case $opt in
        m)
            build_optional $OPTARG
            ;;
        c)
            clean_optional $OPTARG
            ;;
        i)
            config_optional $OPTARG
            ;;
    esac
done

