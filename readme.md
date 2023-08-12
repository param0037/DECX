# DECX
A fast linear algebra, computer vision, and mathematic library, based on CUDA and CPU multi-threading

# Components
1. Linear Algebra (Including matrix multiplication, transposing, scanning(i.e. prefix sum), dot product, and so on)
2. Digital signal processing (Including fast Fourier transformation applied on 1D and 2D array, frequency domain filters, and so on)
3. Computer vision (Including Edge detectors, color space transformation, filters, and so on)
4. Neural Network components (Including convolution on the multi-channel array, executing functions including sigmoid(developing))

# Changelog
## Update 16/7/2023
1. Divide the project according to components according to their functionalities.
2. Added CUDA scan algorithms based on decoupled lookback method [1], which gives perfect performance.

## Update 28/7/2023
1. Added reduction algorithms will be further integrated into summation, min/max, dot product, etc.
2. Added asynchronous part of memory copying (between host and device)

## Update 12/8/2023
1. Added full, horizontal, and vertical reduction for the 2D array.
2. Added support of fp64(double) and int32(32-bit signed integer) in full reduction mode of de::Matrix.
3. Sorted the APIs.

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
