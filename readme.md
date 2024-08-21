# DECX
A fast linear algebra, computer vision, and mathematic library, based on CUDA and CPU multi-threading

# Components
1. Linear Algebra (Including matrix multiplication, transposing, scanning(i.e. prefix sum), dot product, and so on)
2. Digital signal processing (Including fast Fourier transformation applied on 1D and 2D array, frequency domain filters, and so on)
3. Computer vision (Including Edge detectors, color space transformation, filters, and so on)
4. Neural Network components (Including convolution on the multi-channel array, executing functions including sigmoid(developing))

# Supported Platforms
1. Windows 7 or later, x86, 64bit.
2. Ubuntu, x86, 64bit.
3. Android that supports ARM64 CPUs.


# Make
## Dependencies
1. Nvidia CUDA toolkit with proper version.
2. CMake v3.23 or later if using CMake.
3. SDL2 and SDL_image.
4. NDK, if building for Android ARM64 platform.

## make options

## 1. General x86 64bit Windows
### (a) Using Visual Studio (VS)
Clone this repository, open DECX_world.sln with VS. The version of VS should be 2019 or higher. In the window of VS project, right click on the solution, click build solution.

### (b) Using CMake on Windows
Temporary unavailable due to the chnage of buid system recently, will soon be online.

## 2. General x86 Ubuntu
Clone the project, enter the root path of the project. 
1. cd to ${project\_path}/build_system/Linux/
2. Make the necessary scripts executable.
    ```bash 
    sudo chmod u+x ./build_configs.sh
    sudo chmod u+x ./build.sh
3. ```bash
    source ./build_configs.sh
4. Configure the necessary variables.
5. Set the targeted host architecture to x86
    ```bash
    host_arch x86[x86_64][x86-64]
6. Set module to build.
    ```bash 
    set_module core_CPU
7. Check the configured items
    ```bash
    list_configs
8. Clean the configurations of the selected module.
    ```bash
    clean
9. Configure the selected module.
    ```bash
    conf
10. Call function to make
    ```bash
    mk

(Note: If terminal exports "/bin/bash^M: bad interpreter: No such file or directory", please run ./fix_CRLF_cases_for_linux_script.sh or copy the command in this script if it's also unavailable. This is caused editing these scripts on Windows. The line endings LF are replaced by CRLF. So, use sed to fix it in Linux)

## 3. Android for ARM64 (aarch64)
The setps are similar to thoes described in section 2. However, to cross compile aarch64 targets, you need to specify the architecture and toolcahin file first. 

To set the architecture, run 
```bash
    host_arch aarch64
```
To set the toolchain file, run
```bash
    toolchain /path/to/your/toolchain_file.cmake
```

More information about the new building system can be found [here](https://github.com/param0037/DECX/tree/dev_DECX/build_system/readme.md).

(Note: Now all the modules are migrated to the new building system, except for DECX_NN_CPU, because it has no source temporarily.)