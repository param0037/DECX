/**
*   ----------------------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ----------------------------------------------------------------------------------
* 
* This is a part of the open source project named "DECX", a high-performance scientific
* computational library. This project follows the MIT License. For more information 
* please visit https://github.com/param0037/DECX.
* 
* Copyright (c) 2021 Wayne Anderson
* 
* Permission is hereby granted, free of charge, to any person obtaining a copy of this 
* software and associated documentation files (the "Software"), to deal in the Software 
* without restriction, including without limitation the rights to use, copy, modify, 
* merge, publish, distribute, sublicense, and/or sell copies of the Software, and to 
* permit persons to whom the Software is furnished to do so, subject to the following 
* conditions:
* 
* The above copyright notice and this permission notice shall be included in all copies 
* or substantial portions of the Software.
* 
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
* INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR 
* PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE 
* FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR 
* OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
* DEALINGS IN THE SOFTWARE.
*/


#include "transpose_kernels.cuh"


__global__ void 
decx::blas::GPUK::cu_transpose2D_b4_dense(const float* __restrict src, 
                                      float* __restrict dst,
                                      const uint32_t pitchsrc_v1, 
                                      const uint32_t pitchdst_v1, 
                                      const uint2 proc_dim_dst)
{
    uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;

    uint64_t dex = (tidy << 2) * pitchsrc_v1 + tidx;

    __shared__ float4 _buffer[32][8 + 1];

    decx::utils::_cuda_vec128 _regs;
    for (uint8_t i = 0; i < 4; ++i) {
        _regs._vf = decx::utils::vec4_set1_fp32(0.0);
    }
    // Load source data to registers
    if (tidx < pitchsrc_v1) {
        for (uint8_t i = 0; i < 4; ++i) {
            if ((tidy << 2) + i < proc_dim_dst.x) { _regs._arrf[i] = src[dex + pitchsrc_v1 * i]; }
        }
    }
    
    // Store to shared memory, also in transposed form (lane 0)
    _buffer[threadIdx.x][threadIdx.y] = _regs._vf;

    __syncthreads();

    for (uint8_t i = 0; i < 4; ++i) {
        _regs._arrf[i] = ((float*)_buffer[threadIdx.y * 4 + i])[threadIdx.x];
    }

    tidx = threadIdx.x + blockIdx.y * blockDim.x;
    tidy = threadIdx.y + blockIdx.x * blockDim.y;

    dex = tidy * 4 * pitchdst_v1 + tidx;

    if (tidx < pitchdst_v1) {
        for (uint8_t i = 0; i < 4; ++i) {
            if (tidy * 4 + i < proc_dim_dst.y) { dst[dex + pitchdst_v1 * i] = _regs._arrf[i]; }
        }
    }
}
