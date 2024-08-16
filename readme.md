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

## Toolchain combination

## 1. General x86 64bit Windows
### (a) Using Visual Studio (VS)
Clone this repository, open DECX_world.sln with VS. The version of VS should be 2019 or higher. In the window of VS project, right click on the solution, click build solution.

### (b) Using CMake on Windows
Follow the same steps described in section (a) until opening DECX_world.sln. Run cmd or powershell under the root path of the project. Run the build script build.cmd. Simply enter "build" and the script will print the instructions. Follow these instructions to build the project.

## 2. General x86 Ubuntu
Clone the project, enter the root path of the project. Then run the build script for Linux, which is build.sh. Simply enter "./build.sh" and follow the printed instructions to build the project. 
(Note: If terminal exports "/bin/bash^M: bad interpreter: No such file or directory", please run ./fix_CRLF_cases_for_linux_script.sh or copy the command in this script if it's also unavailable. This is caused editing these scripts on Windows. The line endings LF are replaced by CRLF. So, use sed to fix it in Linux)

## 3. Android for ARM64 (aarch64)
Follow the same steps described in section 2 before running the script. When performing cross compilation for Android ARM64, replace the directory where your NDK is installed for -DCMAKE_TOOLCHAIN_FILE at line 10 of config_on_android.sh. Run "config_on_android.sh" and then run "build_on_android.sh". Please notice that only DECX_core_CPU supports aarch64. Additionally, its building system is just a prototype. Supports for aarch64 will be updated later.

Now buiding for android is only supported by DECX_core_CPU and DECX_BLAS_CPU. By changing the built module, change the cd to the path of the built module (i.e. \${project_path}/DECX_core_CPU) on file ${project_dir}/config_on_android.sh. According to the dependency, you need to first build DECX_core_CPU and then DECX_BLAS_CPU, manually. Such process aren't included in the main building script yet, but will be soon.
