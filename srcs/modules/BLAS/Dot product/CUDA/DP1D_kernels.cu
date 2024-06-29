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


#include "DP_kernels.cuh"
#include "../../../DSP/CUDA_cpf32.cuh"


__global__ void 
decx::blas::GPUK::cu_block_dot1D_fp32(const float4* __restrict       A, 
                                     const float4* __restrict       B, 
                                     float* __restrict              dst,
                                     const uint64_t                 proc_len_v4, 
                                     const uint64_t                 proc_len_v1)
{
    uint64_t LDG_dex = (uint64_t)threadIdx.x + (uint64_t)blockIdx.x * (uint64_t)blockDim.x;
    uint16_t warp_lane_id = threadIdx.x & 0x1f;
    uint16_t local_warp_id = threadIdx.x / 32;

    // 8 effective values but extended to 32 for warp-level loading
    /*
    * Don't need to initialize the shared memory to all zero because the rest of the
      array (with length of 32 - 8 = 24) is not invovled in the computation if the template
      variable of the warp-level reduce fuction is set to 8
    */
    __shared__ float warp_reduce_results[32];

    decx::utils::_cuda_vec128 _recv1, _recv2;
    _recv1._vf = decx::utils::vec4_set1_fp32(0);
    _recv2._vf = decx::utils::vec4_set1_fp32(0);
    float tmp1, tmp2;

    if (LDG_dex < proc_len_v4) {
        _recv1._vf = A[LDG_dex];
        _recv2._vf = B[LDG_dex];
        if (LDG_dex == proc_len_v4 - 1) {
            for (int i = 4 - (proc_len_v4 * 4 - proc_len_v1); i < 4; ++i) {
                _recv2._arrf[i] = 0.f;
            }
        }
    }

    tmp1 = __fmul_rn(_recv1._vf.x, _recv2._vf.x);
    tmp1 = fmaf(_recv1._vf.y, _recv2._vf.y, tmp1);
    tmp1 = fmaf(_recv1._vf.z, _recv2._vf.z, tmp1);
    tmp1 = fmaf(_recv1._vf.w, _recv2._vf.w, tmp1);
    
    // warp reducing
    decx::reduce::GPUK::cu_warp_reduce<float, 32>(__fadd_rn, &tmp1, &tmp2);

    if (warp_lane_id == 0) {
        warp_reduce_results[local_warp_id] = tmp2;
    }

    __syncthreads();

    // let the 0th warp execute the warp-reducing process
    if (local_warp_id == 0) {
        tmp1 = warp_reduce_results[warp_lane_id];
        // synchronize this warp
        __syncwarp(0xffffffff);

        decx::reduce::GPUK::cu_warp_reduce<float, 8>(__fadd_rn, &tmp1, &tmp2);

        if (warp_lane_id == 0) {
            dst[blockIdx.x] = tmp2;
        }
    }
}





__global__ void 
decx::blas::GPUK::cu_block_dot1D_fp16_L1(const float4* __restrict       A,
                                        const float4* __restrict       B, 
                                        float* __restrict              dst,
                                        const uint64_t                 proc_len_v8, 
                                        const uint64_t                 proc_len_v1)
{
#if __ABOVE_SM_53
    uint64_t LDG_dex = (uint64_t)threadIdx.x + (uint64_t)blockIdx.x * (uint64_t)blockDim.x;
    uint16_t warp_lane_id = threadIdx.x & 0x1f;
    uint16_t local_warp_id = threadIdx.x / 32;

    // 8 effective values but extended to 32 for warp-level loading
    /*
    * Don't need to initialize the shared memory to all zero because the rest of the
      array (with length of 32 - 8 = 24) is not invovled in the computation if the template
      variable of the warp-level reduce fuction is set to 8
    */
    __shared__ float warp_reduce_results[32];

    decx::utils::_cuda_vec128 _recv1, _recv2, reg1, reg2;

    _recv1._vf = decx::utils::vec4_set1_fp32(0);
    _recv2._vf = decx::utils::vec4_set1_fp32(0);
    float tmp1, tmp2;

    if (LDG_dex < proc_len_v8) {
        _recv1._vf = A[LDG_dex];
        _recv2._vf = B[LDG_dex];
        if (LDG_dex == proc_len_v8 - 1) {
            for (int i = 8 - (proc_len_v8 * 8 - proc_len_v1); i < 8; ++i) {
                _recv2._arrs[i] = 0;
            }
        }
    }
    // convert and calculate
    reg1._arrf2[0] = __half22float2(_recv1._arrh2[0]);      reg1._arrf2[1] = __half22float2(_recv1._arrh2[1]);
    reg2._arrf2[0] = __half22float2(_recv2._arrh2[0]);      reg2._arrf2[1] = __half22float2(_recv2._arrh2[1]);
    tmp1 = __fmul_rn(reg1._vf.x, reg2._vf.x);               tmp1 = fmaf(reg1._vf.y, reg2._vf.y, tmp1);
    tmp1 = fmaf(reg1._vf.z, reg2._vf.z, tmp1);              tmp1 = fmaf(reg1._vf.w, reg2._vf.w, tmp1);

    // convert and calculate
    reg1._arrf2[0] = __half22float2(_recv1._arrh2[2]);      reg1._arrf2[1] = __half22float2(_recv1._arrh2[3]);
    reg2._arrf2[0] = __half22float2(_recv2._arrh2[2]);      reg2._arrf2[1] = __half22float2(_recv2._arrh2[3]);
    tmp1 = fmaf(reg1._vf.x, reg2._vf.x, tmp1);              tmp1 = fmaf(reg1._vf.y, reg2._vf.y, tmp1);
    tmp1 = fmaf(reg1._vf.z, reg2._vf.z, tmp1);              tmp1 = fmaf(reg1._vf.w, reg2._vf.w, tmp1);
    
    // warp reducing
    decx::reduce::GPUK::cu_warp_reduce<float, 32>(__fadd_rn, &tmp1, &tmp2);

    if (warp_lane_id == 0) {
        warp_reduce_results[local_warp_id] = tmp2;
    }

    __syncthreads();

    // let the 0th warp execute the warp-reducing process
    if (local_warp_id == 0) {
        tmp1 = warp_reduce_results[warp_lane_id];
        // synchronize this warp
        __syncwarp(0xffffffff);

        decx::reduce::GPUK::cu_warp_reduce<float, 8>(__fadd_rn, &tmp1, &tmp2);

        if (warp_lane_id == 0) {
            dst[blockIdx.x] = tmp2;
        }
    }
#endif
}




__global__ void 
decx::blas::GPUK::cu_block_dot1D_fp16_L2(const float4* __restrict       A,
                                        const float4* __restrict       B, 
                                        __half* __restrict             dst,
                                        const uint64_t                 proc_len_v8, 
                                        const uint64_t                 proc_len_v1)
{
#if __ABOVE_SM_53
    uint64_t LDG_dex = (uint64_t)threadIdx.x + (uint64_t)blockIdx.x * (uint64_t)blockDim.x;
    uint16_t warp_lane_id = threadIdx.x & 0x1f;
    uint16_t local_warp_id = threadIdx.x / 32;

    // 8 effective values but extended to 32 for warp-level loading
    /*
    * Don't need to initialize the shared memory to all zero because the rest of the
      array (with length of 32 - 8 = 24) is not invovled in the computation if the template
      variable of the warp-level reduce fuction is set to 8
    */
    __shared__ float warp_reduce_results[32];

    decx::utils::_cuda_vec128 _recv1, _recv2, reg1, reg2;

    _recv1._vf = decx::utils::vec4_set1_fp32(0);
    _recv2._vf = decx::utils::vec4_set1_fp32(0);
    float tmp1, tmp2;

    if (LDG_dex < proc_len_v8) {
        _recv1._vf = A[LDG_dex];
        _recv2._vf = B[LDG_dex];
        if (LDG_dex == proc_len_v8 - 1) {
            for (int i = 8 - (proc_len_v8 * 8 - proc_len_v1); i < 8; ++i) {
                _recv2._arrs[i] = 0;
            }
        }
    }
    // convert and calculate
    reg1._arrf2[0] = __half22float2(_recv1._arrh2[0]);      reg1._arrf2[1] = __half22float2(_recv1._arrh2[1]);
    reg2._arrf2[0] = __half22float2(_recv2._arrh2[0]);      reg2._arrf2[1] = __half22float2(_recv2._arrh2[1]);
    tmp1 = __fmul_rn(reg1._vf.x, reg2._vf.x);               tmp1 = fmaf(reg1._vf.y, reg2._vf.y, tmp1);
    tmp1 = fmaf(reg1._vf.z, reg2._vf.z, tmp1);              tmp1 = fmaf(reg1._vf.w, reg2._vf.w, tmp1);

    // convert and calculate
    reg1._arrf2[0] = __half22float2(_recv1._arrh2[2]);      reg1._arrf2[1] = __half22float2(_recv1._arrh2[3]);
    reg2._arrf2[0] = __half22float2(_recv2._arrh2[2]);      reg2._arrf2[1] = __half22float2(_recv2._arrh2[3]);
    tmp1 = fmaf(reg1._vf.x, reg2._vf.x, tmp1);              tmp1 = fmaf(reg1._vf.y, reg2._vf.y, tmp1);
    tmp1 = fmaf(reg1._vf.z, reg2._vf.z, tmp1);              tmp1 = fmaf(reg1._vf.w, reg2._vf.w, tmp1);
    
    // warp reducing
    decx::reduce::GPUK::cu_warp_reduce<float, 32>(__fadd_rn, &tmp1, &tmp2);

    if (warp_lane_id == 0) {
        warp_reduce_results[local_warp_id] = tmp2;
    }

    __syncthreads();

    // let the 0th warp execute the warp-reducing process
    if (local_warp_id == 0) {
        tmp1 = warp_reduce_results[warp_lane_id];
        // synchronize this warp
        __syncwarp(0xffffffff);

        decx::reduce::GPUK::cu_warp_reduce<float, 8>(__fadd_rn, &tmp1, &tmp2);

        if (warp_lane_id == 0) {
            dst[blockIdx.x] = __float2half(tmp2);
        }
    }
#endif
}




__global__ void 
decx::blas::GPUK::cu_block_dot1D_fp16_L3(const float4* __restrict       A,
                                        const float4* __restrict       B, 
                                        __half* __restrict             dst,
                                        const uint64_t                 proc_len_v8, 
                                        const uint64_t                 proc_len_v1)
{
#if __ABOVE_SM_53
    uint64_t LDG_dex = (uint64_t)threadIdx.x + (uint64_t)blockIdx.x * (uint64_t)blockDim.x;
    uint16_t warp_lane_id = threadIdx.x & 0x1f;
    uint16_t local_warp_id = threadIdx.x / 32;

    // 8 effective values but extended to 32 for warp-level loading
    /*
    * Don't need to initialize the shared memory to all zero because the rest of the
      array (with length of 32 - 8 = 24) is not invovled in the computation if the template
      variable of the warp-level reduce fuction is set to 8
    */
    __shared__ __half warp_reduce_results[32];

    decx::utils::_cuda_vec128 _recv1, _recv2;

    _recv1._vf = decx::utils::vec4_set1_fp32(0);
    _recv2._vf = decx::utils::vec4_set1_fp32(0);
    __half2 tmp;

    if (LDG_dex < proc_len_v8) {
        _recv1._vf = A[LDG_dex];
        _recv2._vf = B[LDG_dex];
        if (LDG_dex == proc_len_v8 - 1) {
            for (int i = 8 - (proc_len_v8 * 8 - proc_len_v1); i < 8; ++i) {
                _recv2._arrs[i] = 0;
            }
        }
    }
    // on-thread register calculation
    tmp = __hmul2(_recv1._arrh2[0], _recv2._arrh2[0]);
    tmp = __hfma2(_recv1._arrh2[1], _recv2._arrh2[1], tmp);
    tmp = __hfma2(_recv1._arrh2[2], _recv2._arrh2[2], tmp);
    tmp = __hfma2(_recv1._arrh2[3], _recv2._arrh2[3], tmp);
    
    tmp.x = __hadd(tmp.x, tmp.y);

    // warp reducing
    decx::reduce::GPUK::cu_warp_reduce<__half, 32>(__hadd, &tmp.x, &tmp.y);

    if (warp_lane_id == 0) {
        warp_reduce_results[local_warp_id] = tmp.y;
    }

    __syncthreads();

    // let the 0th warp execute the warp-reducing process
    if (local_warp_id == 0) {
        tmp.x = warp_reduce_results[warp_lane_id];
        // synchronize this warp
        __syncwarp(0xffffffff);

        decx::reduce::GPUK::cu_warp_reduce<__half, 8>(__hadd, &tmp.x, &tmp.y);

        if (warp_lane_id == 0) {
            dst[blockIdx.x] = tmp.y;
        }
    }
#endif
}




__global__ void 
decx::blas::GPUK::cu_block_dot1D_fp64(const double2* __restrict       A, 
                                     const double2* __restrict       B, 
                                     double* __restrict              dst,
                                     const uint64_t                  proc_len_v2, 
                                     const uint64_t                  proc_len_v1)
{
    uint64_t LDG_dex = (uint64_t)threadIdx.x + (uint64_t)blockIdx.x * (uint64_t)blockDim.x;
    uint16_t warp_lane_id = threadIdx.x & 0x1f;
    uint16_t local_warp_id = threadIdx.x / 32;

    // 8 effective values but extended to 32 for warp-level loading
    /*
    * Don't need to initialize the shared memory to all zero because the rest of the
      array (with length of 32 - 8 = 24) is not invovled in the computation if the template
      variable of the warp-level reduce fuction is set to 8
    */
    __shared__ double warp_reduce_results[32];

    decx::utils::_cuda_vec128 _recv1, _recv2;
    _recv1._vd = decx::utils::vec2_set1_fp64(0);
    _recv2._vd = decx::utils::vec2_set1_fp64(0);
    double tmp1, tmp2;

    if (LDG_dex < proc_len_v2) {
        _recv1._vd = A[LDG_dex];
        _recv2._vd = B[LDG_dex];
        if (LDG_dex == proc_len_v2 - 1) {
            for (int i = 2 - (proc_len_v2 * 2 - proc_len_v1); i < 2; ++i) {
                _recv2._arrd[i] = 0.f;
            }
        }
    }
    
    tmp1 = __dmul_rn(_recv1._vd.x, _recv2._vd.x);
    tmp1 = __fma_rn(_recv1._vd.y, _recv2._vd.y, tmp1);
    
    // warp reducing
    decx::reduce::GPUK::cu_warp_reduce<double, 32>(__dadd_rn, &tmp1, &tmp2);

    if (warp_lane_id == 0) {
        warp_reduce_results[local_warp_id] = tmp2;
    }

    __syncthreads();

    // let the 0th warp execute the warp-reducing process
    if (local_warp_id == 0) {
        tmp1 = warp_reduce_results[warp_lane_id];
        // synchronize this warp
        __syncwarp(0xffffffff);

        decx::reduce::GPUK::cu_warp_reduce<double, 8>(__dadd_rn, &tmp1, &tmp2);

        if (warp_lane_id == 0) {
            dst[blockIdx.x] = tmp2;
        }
    }
}




__global__ void 
decx::blas::GPUK::cu_block_dot1D_cplxf(const float4* __restrict         A, 
                                      const float4* __restrict         B, 
                                      de::CPf* __restrict              dst,
                                      const uint64_t                   proc_len_v2, 
                                      const uint64_t                   proc_len_v1)
{
    uint64_t LDG_dex = (uint64_t)threadIdx.x + (uint64_t)blockIdx.x * (uint64_t)blockDim.x;
    uint16_t warp_lane_id = threadIdx.x & 0x1f;
    uint16_t local_warp_id = threadIdx.x / 32;

    // 8 effective values but extended to 32 for warp-level loading
    /*
    * Don't need to initialize the shared memory to all zero because the rest of the
      array (with length of 32 - 8 = 24) is not invovled in the computation if the template
      variable of the warp-level reduce fuction is set to 8
    */
    __shared__ de::CPf warp_reduce_results[32];

    decx::utils::_cuda_vec128 _recv1, _recv2;
    _recv1._vd = decx::utils::vec2_set1_fp64(0);
    _recv2._vd = decx::utils::vec2_set1_fp64(0);
    de::CPf tmp1, tmp2;

    if (LDG_dex < proc_len_v2) {
        _recv1._vf = A[LDG_dex];
        _recv2._vf = B[LDG_dex];
        if (LDG_dex == proc_len_v2 - 1) {
            for (int i = 2 - (proc_len_v2 * 2 - proc_len_v1); i < 2; ++i) {
                _recv2._arrd[i] = 0.f;
            }
        }
    }
    
    tmp1 = decx::dsp::fft::GPUK::_complex_mul_fp32(_recv1._arrcplxf2[0], _recv2._arrcplxf2[0]);
    tmp1 = decx::dsp::fft::GPUK::_complex_fma_fp32(_recv1._arrcplxf2[1], _recv2._arrcplxf2[1], tmp1);
    
    // warp reducing
    decx::reduce::GPUK::cu_warp_reduce<double, 32>(decx::dsp::fft::GPUK::_complex_add_warp_call, ((double*)&tmp1), ((double*)&tmp2));

    if (warp_lane_id == 0) {
        warp_reduce_results[local_warp_id] = tmp2;
    }

    __syncthreads();

    // let the 0th warp execute the warp-reducing process
    if (local_warp_id == 0) {
        tmp1 = warp_reduce_results[warp_lane_id];
        // synchronize this warp
        __syncwarp(0xffffffff);

        decx::reduce::GPUK::cu_warp_reduce<double, 8>(decx::dsp::fft::GPUK::_complex_add_warp_call, 
            ((double*)&tmp1), ((double*)&tmp2));

        if (warp_lane_id == 0) {
            dst[blockIdx.x] = tmp2;
        }
    }
}