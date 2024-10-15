#!/bin/bash

# ----------------------------------------------------------------------------------
# Author : Wayne Anderson
# Date : 2021.04.16
# ----------------------------------------------------------------------------------
#
# This is a part of the open source project named "DECX", a high-performance scientific
# computational library. This project follows the MIT License. For more information
# please visit https:
#
# Copyright (c) 2021 Wayne Anderson
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this
# software and associated documentation files (the "Software"), to deal in the Software
# without restriction, including without limitation the rights to use, copy, modify,
# merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to the following
# conditions:
#
# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


# The folowing graph illustartes how prebuits are installed
# and managed in Linux context.
# 
# ├── 3rdparty
# │  ├── ${DEP_NAME}
# │  │   ├── ${HOST_ARCH}
# │  │   |   ├── ${OS}
# │  │   |   |   ├── tmp    // Tmp dir
# │  │   |   |   |   ├── DIR_${DOWNLOAD_PACKAGE_NAME}   // Package extract dir
# │  │   |   |   ├── lib    // The path where the prebuilts are installed
# |  |   |   |
# │  └── other dependencies ...


source $PROJECT_PATH_BUILD/build_system/Linux/utils.sh


function is_prebuilt_exist()
{
    prebuilt_name="$1"
    if [ -d "$PROJECT_PATH_BUILD/3rdparty/$1/$DECX_HOST_ARCH/Linux" ]; then
        echo_status "Found prebuilt package $1 ($DECX_HOST_ARCH)"
        return 1;
    else
        echo_warning "Prebuilt package $1 ($DECX_HOST_ARCH) not exist"
        return 0
    fi
}


# Should have entered Tmp dir
function extract_downloads()
{
    filename=$1
    extract_folder_name="DIR_$filename"

    if [ ! -e "./$extract_folder_name" ]; then
        mkdir ./$extract_folder_name > /dev/null
    fi

    if [[ "$filename" == *".deb"* ]]; then
        dpkg-deb -X ./$filename ./$extract_folder_name > /dev/null
    fi
    if [[ "$filename" == *".tar.xz"* ]]; then
        tar -xf ./$filename -C ./$extract_folder_name > /dev/null
    fi

    echo -n "$extract_folder_name"
}


# Should have entered Tmp dir
function download_n_install_SDL2()
{
    prebuilt_name="SDL2"
    echo_status "Envoke wget to download $prebuilt_name"

    is_aarch64 $DECX_HOST_ARCH
    aarch64_case=$?
    if [ $aarch64_case -eq 0 ]; then
        url=http://ftp.us.debian.org/debian/pool/main/libs/libsdl2/libsdl2-2.0-0_2.26.5+dfsg-1_i386.deb
    else
        url=http://mirror.archlinuxarm.org/aarch64/extra/sdl2-2.30.7-1-aarch64.pkg.tar.xz
    fi

    package_name_download=${url##*/}

    wget $url
    echo_status "Installing $prebuilt_name"

    extracted_folder=$(extract_downloads $package_name_download)

    if [ $aarch64_case -eq 0 ]; then    # Execute x86-SLD2 installation procedures
        cp -r ./$extracted_folder/usr/lib/i386-linux-gnu ../
        mv ../i386-linux-gnu ../lib
    else                                # Execute aarch64-SLD2 installation procedures
        cp -r ./$extracted_folder/usr/lib ../
    fi
}

# Should have entered Tmp dir
function download_n_install_SDL2_Image()
{
    prebuilt_name="SDL2_Image"
    echo_status "Envoke wget to download $prebuilt_name"

    is_aarch64 $DECX_HOST_ARCH
    aarch64_case=$?
    if [ $aarch64_case -eq 0 ]; then
        url=http://ftp.us.debian.org/debian/pool/main/libs/libsdl2-image/libsdl2-image-2.0-0_2.6.3+dfsg-1_i386.deb
    else
        url=http://mirror.archlinuxarm.org/aarch64/extra/sdl2_image-2.8.2-5-aarch64.pkg.tar.xz
    fi

    package_name_download=${url##*/}

    wget $url
    echo_status "Installing $prebuilt_name"

    extracted_folder=$(extract_downloads $package_name_download)

    if [ $aarch64_case -eq 0 ]; then    # Execute x86-SLD2 installation procedures
        cp -r ./$extracted_folder/usr/lib/i386-linux-gnu ../
        mv ../i386-linux-gnu ../lib
    else                                # Execute aarch64-SLD2 installation procedures
        cp -r ./$extracted_folder/usr/lib ../
    fi
}


# Should have entered Tmp dir
function download_n_install_Python3()
{
    prebuilt_name="Python"
    echo_status "Envoke wget to download $prebuilt_name"

    url=http://mirror.archlinuxarm.org/aarch64/core/python-3.12.6-1-aarch64.pkg.tar.xz

    package_name_download=${url##*/}

    wget $url
    echo_status "Installing $prebuilt_name"

    extracted_folder=$(extract_downloads $package_name_download)

    cp -r ./$extracted_folder/usr/lib ../
}


function download_prebuilts()
{
    prebuilt_name=$1        # Get prebuilt name from the parameter
    
    # IF not exist, call wget to download it
    is_prebuilt_exist $prebuilt_name        # Check if this dependency exists
    
    if [ $? -eq 0 ]; then
        prebuilt_tmp_dir=$PROJECT_PATH_BUILD/3rdparty/$prebuilt_name/$DECX_HOST_ARCH/Linux
        mkdir $prebuilt_tmp_dir
        prebuilt_tmp_dir=$prebuilt_tmp_dir/tmp
        mkdir $prebuilt_tmp_dir

        # Enter Tmp dir
        echo_status "Entering directory $prebuilt_tmp_dir"
        cd $prebuilt_tmp_dir

        # Unpack & install it
        install_cmd=download_n_install_$prebuilt_name
        eval $install_cmd       # Install the prebuilts

        cd ../../
        echo_status "Leaving directory $prebuilt_tmp_dir"

        # Clean the tmp
        echo_status "Download $prebuilt_name done, cleaning the caches"
        rm -rf ./Linux/tmp
    fi
}


function manage_prebuilt()
{
    if [ ! -e "$PROJECT_PATH_BUILD/3rdparty" ]; then
        mkdir $PROJECT_PATH_BUILD/3rdparty
    fi

    prebuilt_name=$1

    lib_path_dir=$PROJECT_PATH_BUILD/3rdparty/$prebuilt_name

    if [ ! -e "$lib_path_dir" ]; then
        echo_status "Path not exist, creating $lib_path_dir"
        mkdir $lib_path_dir
    fi

    lib_path_dir="$lib_path_dir/$DECX_HOST_ARCH"

    if [ ! -e "$lib_path_dir" ]; then
        echo_status "Path not exist, creating $lib_path_dir"
        mkdir $lib_path_dir
    fi

    echo_status "Entering directory: $lib_path_dir"
    cd $lib_path_dir

    download_prebuilts $prebuilt_name

    path_n_filename=$(realpath "$0")
    current_path=$(dirname "$path_n_filename")
    cd $current_path
    echo_status "Leaving directory: $lib_path_dir"

    echo_success "--------------------------------------------------- Successfully installed $prebuilt_name ---------------------------------------------------"
}


function detect_Python3()
{
    if command -v python3 &> /dev/null
    then
        echo_success "Python3 detected"
        return 1
    else
        echo_error "Python3 is not installed, please install it first !"
        return 0
    fi
}


function install_libpython_NoCross()
{
    if [ ! -e "$libpython_install_dir" ]; then
        echo_status "No python prebuilt found, now install it"

        libpython_install_dir=$PROJECT_PATH_BUILD/3rdparty/Python3
        if [ ! -e "$libpython_install_dir" ]; then
            mkdir $libpython_install_dir
        fi
        libpython_install_dir=$libpython_install_dir/$DECX_HOST_ARCH
        if [ ! -e "$libpython_install_dir" ]; then
            mkdir $libpython_install_dir
        fi
        libpython_install_dir=$libpython_install_dir/Linux
        if [ ! -e "$libpython_install_dir" ]; then
            mkdir $libpython_install_dir
        fi

        detect_Python3

        if [ $? -eq 1 ]; then
            libpython_dir=$(python3 -c "import sysconfig;   \
                                        print(sysconfig.get_config_var('LIBDIR'))")

            echo_status "Found python library at $libpython_dir"

            if [ -d "$libpython_dir" ]; then
                lsout=$(ls $libpython_dir | grep libpython)
            
                echo "$lsout" | while IFS= read -r line
                do
                    echo_status "Copied $line to $libpython_install_dir/"
                    cp $libpython_dir/$line $libpython_install_dir
                done

                echo_success "--------------------------------------------------- Successfully installed Python3 ---------------------------------------------------"
            else
                echo_error "The path provided by Python not exist, please check Python installation or reinstall it"
            fi
        fi
    else
        echo_status "Found python prebuilt, skip"
    fi
}


IFS=',' read -ra array <<< "$1"
for element in "${array[@]}"; do
    case $element in
    "SDL2")
        manage_prebuilt SDL2
        ;;
    "SDL2_Image")
        manage_prebuilt SDL2_Image
        ;;
    "Python")
        if [ "$DECX_HOST_ARCH" = "aarch64" ]; then
            manage_prebuilt Python3
        else
            install_libpython_NoCross
        fi
        ;;
    esac
done
