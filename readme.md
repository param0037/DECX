# DECX
A fast linear algebra, computer vision, and mathematic library, based on CUDA and CPU multi-threading

# Components
1. Linear Algebra (Including matrix multiplication, transposing, scanning(i.e. prefix sum), dot product, and so on)
2. Digital signal processing (Including fast Fourier transformation applied on 1D and 2D array, frequency domain filters, and so on)
3. Computer vision (Including Edge detectors, color space transformation, filters, and so on)
4. Neural Network components (Including convolution on multi-channel array, executing functions including sigmoid(developing))

# Update 16/7/2023
1. Divide the project according into components according to their functionalities.
2. Add CUDA scan algorithm based on decoupled lookback method [1], which gives perfect performance.

# How to build
## 1. Windows
   Open cmd or powershell right in the folder where the project is located. Input "config" first and hit enter, waiting for the process
   to complete. Then input "build" if there is no error occurs in the configuration, waiting until the project is fully built, which may
   take some time.

# References
[1] Duane Merrill, Michael Garland "Single-pass Parallel Prefix Scan with Decoupled Look-back" Tuesday, March 1, 2016
   [Online] Available: https://research.nvidia.com/publication/2016-03_single-pass-parallel-prefix-scan-decoupled-look-back
