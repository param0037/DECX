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


#include "im2col_GEMM_fp32.cuh"
#include "../../../../../common/CUSV/decx_cuda_vectypes_ops.cuh"


// block[32, 8]
// load_src[128, 1]
// load_kernel[1, 4]
// store_dst[4, 128]
__global__
void decx::nn::GPUK::cu_im2col_GEMM_fp32(const float4* __restrict   im2col_buf, 
                                         const float4* __restrict   kernel,
                                         float4* __restrict         dst, 
                                         const uint32_t             dpitch_dst_v1, 
                                         const uint32_t             wpitch_i2c_v1, 
                                         const uint32_t             wpitch_dst_v1, 
                                         const uint32_t             _L_proc_v1, 
                                         const uint2                conv2D_area)
{
    const int32_t tidx_dist = threadIdx.x + blockIdx.x * blockDim.x;
    const int32_t tidy_dist = threadIdx.y + blockIdx.y * blockDim.y;

    const uint64_t _logical_pitch_i2c_v1 = (uint64_t)wpitch_i2c_v1 * (uint64_t)conv2D_area.y;

    uint32_t _LDG_I2C_X = tidx_dist + tidy_dist * wpitch_i2c_v1 / 4;
    uint32_t _STG_dst_X = threadIdx.x + blockIdx.x * 128 + tidy_dist * wpitch_dst_v1 * dpitch_dst_v1 / 4;

    decx::utils::_cuda_vec128 _recv_i2c, _recv_kernel, _res[4];
    _res[0]._vf = decx::utils::vec4_set1_fp32(0.f);
    _res[1]._vf = decx::utils::vec4_set1_fp32(0.f);
    _res[2]._vf = decx::utils::vec4_set1_fp32(0.f);
    _res[3]._vf = decx::utils::vec4_set1_fp32(0.f);

    __shared__ float4 _shmem[_IM2COL_GEMM_FP32_BLOCK_Y_][32 * 4 + 1];

    for (uint32_t i = 0; i < _L_proc_v1; ++i)
    {
        _recv_i2c._vf = decx::utils::vec4_set1_fp32(0);
        _recv_kernel._vf = decx::utils::vec4_set1_fp32(0);

        //uint32_t _physical_i = decx::nn::GPUK::_Lproc_params_i2c_fp32[0].get_phyaddr_L1(i);
        
        if (tidx_dist < decx::utils::ceil<uint32_t>(conv2D_area.x, 4) && tidy_dist < conv2D_area.y)
        {
            _recv_i2c._vf = im2col_buf[_LDG_I2C_X + /*_physical_i*/i * _logical_pitch_i2c_v1 / 4];

            _recv_kernel._vf = kernel[blockIdx.z + i * 8];  // 8 stands for 128bytes = 8float4

#pragma unroll 4
            for (int32_t j = 0; j < 4; ++j) {
                _res[j]._vf.x = __fmaf_rn(_recv_i2c._arrf[j], _recv_kernel._vf.x, _res[j]._vf.x);
                _res[j]._vf.y = __fmaf_rn(_recv_i2c._arrf[j], _recv_kernel._vf.y, _res[j]._vf.y);
                _res[j]._vf.z = __fmaf_rn(_recv_i2c._arrf[j], _recv_kernel._vf.z, _res[j]._vf.z);
                _res[j]._vf.w = __fmaf_rn(_recv_i2c._arrf[j], _recv_kernel._vf.w, _res[j]._vf.w);
            }
        }
    }

    _shmem[threadIdx.y][threadIdx.x * 4] = _res[0]._vf;
    _shmem[threadIdx.y][threadIdx.x * 4 + 1] = _res[1]._vf;
    _shmem[threadIdx.y][threadIdx.x * 4 + 2] = _res[2]._vf;
    _shmem[threadIdx.y][threadIdx.x * 4 + 3] = _res[3]._vf;

    __syncthreads();

#pragma unroll 4
    for (int32_t j = 0; j < 4; ++j) 
    {
        _res[j]._vf = _shmem[threadIdx.y][threadIdx.x + j * 32];
        
        if (threadIdx.x + blockIdx.x * 128 + _CUDA_WARP_SIZE_ * j < conv2D_area.x && tidy_dist < conv2D_area.y) {
            dst[_STG_dst_X] = _res[j]._vf;
            _STG_dst_X += _CUDA_WARP_SIZE_ * dpitch_dst_v1 / 4;
        }
    }
}
