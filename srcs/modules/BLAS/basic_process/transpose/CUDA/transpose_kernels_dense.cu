/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#include "transpose_kernels.cuh"


__global__ void 
decx::bp::GPUK::cu_transpose2D_b4_dense(const float* __restrict src, 
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