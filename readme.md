# DECX
A fast linear algebra, computer vision, and mathematic library, based on CUDA and CPU multi-threading

# Components
1. Linear Algebra (Including matrix multiplication, transposing, scanning(i.e. prefix sum), dot product, and so on)
2. Digital signal processing (Including fast Fourier transformation applied on 1D and 2D array, frequency domain filters, and so on)
3. Computer vision (Including Edge detectors, color space transformation, filters, and so on)
4. Neural Network components (Including convolution on the multi-channel array, executing functions including sigmoid(developing))

# Warning
## The project is recommended to be compiled using C++ 17 standard or above.
## The samples are out-of-dated, please ignore them. The latest version is coming.

# Changelog
## Update 29/02/2024
1. Added dpitch=4 and 8 case for conv2D im2col. More depths will be adjusted later.
2. Added strides parameter for conv2D im2col.
3. The memcpy series functions are changed, de::MemcpyLinear is cancelled due to the data structure alignment changes.

# Upcoming optimizations
1. Adjust the pitch of CUDA FFTs kernels (1D, 2D and 3D) to be aligned to 128 bytes for coalesced memory access.
2. Linear algebra mathematical functions, especially iterations (Jacob and GS method) for solving linear equations.


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
