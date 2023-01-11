# DECX
A fast linear algebra, computer vision and mathematic library, based on CUDA and CPU multi-thread

# MODULES
1. Basic calculations such as addition, subtraction, etc.
2. Basic operations such as matrix transpose, type casting, etc.
3. General matrix multiplication(GEMM) and convolution, most frequently used functions in varieties of fields.
4. Signal processing APIs. Such as fast Fourier transform(FFT), inverse fast Fourier transform(IFFT) and many kinds of filters for 1D array and 2D matrix. 
5. APIs for computer vision. (developing)
6. APIs for vector4 (can represent 3D vector and point). Such as linear transform，divergence and rotation, etc.

# To developers and viewers
You can download and modify my source to compile your own project if you want. And you can also tell me what functionality you want me to integrate. If you have 
better ideas, please contact me via enloquecer0037@gmail.com. I hope my library can help you with your research or coursework.
`
# How to compile
Please notice MacOS is not supported! On Windows or Linux, you should first download CMake and make sure its version index at least 3.00 and not exceed 3.40. Since the CMakeLists.txt in this project might not suitable for the latest version of CMake.
Please notice, all the third libraries(except for CUDA) are ready in the folder called "3rdparty". Even in Linux, you are not required to install them.
## 1. Windows
1. Install CUDA SDK and make sure nvcc(Nvidia compiler) is correctly configured. 
2. Install MSVC compiler (Visual Studio 2019 or above is recommended)
3. Open cmd.exe or powershell, and enter the project directory.
4. Enter "config", hit "enter" key. If configurations are successful, then you can move to next step.
5. Enter "build", hit "enter" key. After successfully built, you can see the binary files(five .dll files and five .lib files) under "${project_directory}/bin"
## 2. Linux
1. Install CUDA SDK and make sure nvcc(Nvidia compiler) is correctly configured. 
2. Install Mingw.
3. Open the terminal, and enter the project directory.
4. Enter "./config.sh", hit "enter" key. If configurations are successful, then you can move to next step.
5. Enter "./build.sh", hit "enter" key. After successfully built, you can see the binary files(five .so files) under "${project_directory}/bin"
