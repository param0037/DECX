# DECX
A fast linear algebra, computer vision, and mathematic library, based on CUDA and CPU multi-threading

# Components
1. Linear Algebra (Including matrix multiplication, transposing, scanning(i.e. prefix sum), dot product, and so on)
2. Digital signal processing (Including fast Fourier transformation applied on 1D and 2D array, frequency domain filters, and so on)
3. Computer vision (Including Edge detectors, color space transformation, filters, and so on)
4. Neural Network components (Including convolution on the multi-channel array, executing functions including sigmoid(developing))

# Warning
## The project is recommended to be compiled using C++ 17 standard or above.

# Changelog
## Update 29/02/2024
1. Added 3D FFT on both CPU and CUDA.
2. The convolution on DECX_NN_CPU and DECX_NN_CUDA are temporarily unavailable.

# Upcoming optimizations
1. Full reduction mode of comparisons on vector and matrix.
2. Adapt titling technique to make maximum utilization of L1 and L2 cache, to speed up memory access in CPU-based FFT.
3. Add warp-level primitives to CUDA-based FFT kernels to speed up short-length FFT transforms.
4. More asynchronous APIs are under development.
5. CmakeLists for building on Linux.

# How to build
## 1. Windows
   Open Cmd or PowerShell right in the folder where the project is located. Input "config" first and hit enter, waiting for the process
   to complete. Then input "build" if there is no error in the configuration, waiting until the project is fully built may
   take some time.
## 2. Linux
   Coming soon.

# References
[1] Duane Merrill, Michael Garland "Single-pass Parallel Prefix Scan with Decoupled Look-back" Tuesday, March 1, 2016
   [Online] Available: https://research.nvidia.com/publication/2016-03_single-pass-parallel-prefix-scan-decoupled-look-back
