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


#include "reduce_sum.cuh"
#include <CUSV/CUDA_cpf32.cuh>


__global__ void
decx::reduce::GPUK::cu_block_reduce_sum1D_fp32(const float4* __restrict     src, 
                                               float* __restrict            dst, 
                                               const uint64_t               proc_len_v4,
                                               const uint64_t               proc_len_v1)
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
    
    decx::utils::_cuda_vec128 _recv;
    _recv._vf = decx::utils::vec4_set1_fp32(0);
    float tmp1, tmp2;

    if (LDG_dex < proc_len_v4) 
    {
        _recv._vf = src[LDG_dex];
        if (LDG_dex == proc_len_v4 - 1) {
            for (int i = 4 - (proc_len_v4 * 4 - proc_len_v1); i < 4; ++i) {
                _recv._arrf[i] = 0.f;
            }
        }
    }

    tmp1 = decx::reduce::GPUK::float4_sum(_recv._vf);
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
decx::reduce::GPUK::cu_block_reduce_sum1D_fp64(const double2* __restrict    src, 
                                               double* __restrict           dst, 
                                               const uint64_t               proc_len_v2,
                                               const uint64_t               proc_len_v1)
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

    decx::utils::_cuda_vec128 _recv;
    _recv._vd = decx::utils::vec2_set1_fp64(0);
    double tmp1, tmp2;

    if (LDG_dex < proc_len_v2) 
    {
        _recv._vd = src[LDG_dex];
        if (LDG_dex == proc_len_v2 - 1) {
            for (int i = 2 - (proc_len_v2 * 2 - proc_len_v1); i < 2; ++i) {
                _recv._arrd[i] = 0.0;
            }
        }
    }

    tmp1 = __dadd_rn(_recv._vd.x, _recv._vd.y);
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
decx::reduce::GPUK::cu_block_reduce_sum1D_cplxf(const float4* __restrict    src, 
                                               de::CPf* __restrict           dst, 
                                               const uint64_t               proc_len_v2,
                                               const uint64_t               proc_len_v1)
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

    decx::utils::_cuda_vec128 _recv;
    _recv._vd = decx::utils::vec2_set1_fp64(0);
    de::CPf tmp1, tmp2;

    if (LDG_dex < proc_len_v2) 
    {
        _recv._vf = src[LDG_dex];
        if (LDG_dex == proc_len_v2 - 1) {
            for (int i = 2 - (proc_len_v2 * 2 - proc_len_v1); i < 2; ++i) {
                _recv._arrd[i] = 0.0;
            }
        }
    }

    tmp1 = decx::dsp::fft::GPUK::_complex_add_fp32(_recv._arrcplxf2[0], _recv._arrcplxf2[1]);
    // warp reducing
    decx::reduce::GPUK::cu_warp_reduce<double, 32>(decx::dsp::fft::GPUK::_complex_add_warp_call, 
        ((double*)&tmp1), ((double*)&tmp2));

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




__global__ void 
decx::reduce::GPUK::cu_warp_reduce_sum2D_flatten_fp32(const float4 * __restrict   src, 
                                                   float* __restrict           dst,
                                                   const uint32_t              Wsrc_v4, 
                                                   const uint2                 proc_dims)
{
    uint32_t tidx = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t tidy = threadIdx.y + blockDim.y * blockIdx.y;

    uint64_t LDG_dex = Wsrc_v4 * tidy + tidx;
    uint64_t STG_dex = blockIdx.y * gridDim.x + blockIdx.x;
    
    uint32_t proc_W_v4 = decx::utils::ceil<uint32_t>(proc_dims.x, 4);

    /*
    * Shared memory for the reduced results of 8 warps in a block
    * Extended to 32 for warp-level loading
    * No need to set the remaining values to zero since for the warp reducing
    * all the redundancy will not invovled with correct parameters set
    */
    __shared__ float _sh_warp_reduce_res[32];

    decx::utils::_cuda_vec128 _recv;
    _recv._vf = decx::utils::vec4_set1_fp32(0);

    float _thread_sum = 0, _warp_reduce_res = 0;

    if (tidx < proc_W_v4 && tidy < proc_dims.y) {
        _recv._vf = src[LDG_dex];
        if (tidx == proc_W_v4 - 1) {
            for (int i = 4 - (proc_W_v4 * 4 - proc_dims.x); i < 4; ++i) {
                _recv._arrf[i] = 0.f;
            }
        }
    }

    _thread_sum = decx::reduce::GPUK::float4_sum(_recv._vf);

    decx::reduce::GPUK::cu_warp_reduce<float, 32>(__fadd_rn, &_thread_sum, &_warp_reduce_res);

    if (threadIdx.x == 0) {
        _sh_warp_reduce_res[threadIdx.y] = _warp_reduce_res;
    }

    __syncthreads();

    if (threadIdx.y == 0) {
        _thread_sum = _sh_warp_reduce_res[threadIdx.x];
        __syncwarp(0xffffffff);

        decx::reduce::GPUK::cu_warp_reduce<float, 8>(__fadd_rn, &_thread_sum, &_warp_reduce_res);

        if (threadIdx.x == 0) {
            dst[STG_dex] = _warp_reduce_res;
        }
    }
}




__global__ void 
decx::reduce::GPUK::cu_warp_reduce_sum2D_flatten_i32(const int4 * __restrict   src, 
                                                   int32_t* __restrict          dst,
                                                   const uint32_t              Wsrc_v4, 
                                                   const uint2                 proc_dims)
{
    uint32_t tidx = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t tidy = threadIdx.y + blockDim.y * blockIdx.y;

    uint64_t LDG_dex = Wsrc_v4 * tidy + tidx;
    uint64_t STG_dex = blockIdx.y * gridDim.x + blockIdx.x;
    
    uint32_t proc_W_v4 = decx::utils::ceil<uint32_t>(proc_dims.x, 4);

    /*
    * Shared memory for the reduced results of 8 warps in a block
    * Extended to 32 for warp-level loading
    * No need to set the remaining values to zero since for the warp reducing
    * all the redundancy will not invovled with correct parameters set
    */
    __shared__ int32_t _sh_warp_reduce_res[32];

    decx::utils::_cuda_vec128 _recv;
    _recv._vi = decx::utils::vec4_set1_int32(0);

    int32_t _thread_sum = 0, _warp_reduce_res = 0;

    if (tidx < proc_W_v4 && tidy < proc_dims.y) {
        _recv._vi = src[LDG_dex];
        if (tidx == proc_W_v4 - 1) {
            for (int i = 4 - (proc_W_v4 * 4 - proc_dims.x); i < 4; ++i) {
                _recv._arri[i] = 0;
            }
        }
    }

    _thread_sum = decx::reduce::GPUK::int4_sum(_recv._vi);

    decx::reduce::GPUK::cu_warp_reduce<int32_t, 32>(decx::utils::cuda::__i32_add, &_thread_sum, &_warp_reduce_res);

    if (threadIdx.x == 0) {
        _sh_warp_reduce_res[threadIdx.y] = _warp_reduce_res;
    }

    __syncthreads();

    if (threadIdx.y == 0) {
        _thread_sum = _sh_warp_reduce_res[threadIdx.x];
        __syncwarp(0xffffffff);

        decx::reduce::GPUK::cu_warp_reduce<int32_t, 8>(decx::utils::cuda::__i32_add, &_thread_sum, &_warp_reduce_res);

        if (threadIdx.x == 0) {
            dst[STG_dex] = _warp_reduce_res;
        }
    }
}




__global__ void 
decx::reduce::GPUK::cu_warp_reduce_sum2D_flatten_fp64(const double2 * __restrict   src, 
                                                      double* __restrict           dst,
                                                      const uint32_t              Wsrc_v4, 
                                                      const uint2                 proc_dims)
{
    uint32_t tidx = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t tidy = threadIdx.y + blockDim.y * blockIdx.y;

    uint64_t LDG_dex = Wsrc_v4 * tidy + tidx;
    uint64_t STG_dex = blockIdx.y * gridDim.x + blockIdx.x;
    
    uint32_t proc_W_v2 = decx::utils::ceil<uint32_t>(proc_dims.x, 2);

    /*
    * Shared memory for the reduced results of 8 warps in a block
    * Extended to 32 for warp-level loading
    * No need to set the remaining values to zero since for the warp reducing
    * all the redundancy will not invovled with correct parameters set
    */
    __shared__ float _sh_warp_reduce_res[32];

    decx::utils::_cuda_vec128 _recv;
    _recv._vd = decx::utils::vec2_set1_fp64(0);

    double _thread_sum = 0, _warp_reduce_res = 0;

    if (tidx < proc_W_v2 && tidy < proc_dims.y) {
        _recv._vd = src[LDG_dex];
        if (tidx == proc_W_v2 - 1) {
            for (int i = 2 - (proc_W_v2 * 2 - proc_dims.x); i < 2; ++i) {
                _recv._arrd[i] = 0.0;
            }
        }
    }

    _thread_sum = __dadd_rn(_recv._vd.x, _recv._vd.y);

    decx::reduce::GPUK::cu_warp_reduce<double, 32>(__dadd_rn, &_thread_sum, &_warp_reduce_res);

    if (threadIdx.x == 0) {
        _sh_warp_reduce_res[threadIdx.y] = _warp_reduce_res;
    }

    __syncthreads();

    if (threadIdx.y == 0) {
        _thread_sum = _sh_warp_reduce_res[threadIdx.x];
        __syncwarp(0xffffffff);

        decx::reduce::GPUK::cu_warp_reduce<double, 8>(__dadd_rn, &_thread_sum, &_warp_reduce_res);

        if (threadIdx.x == 0) {
            dst[STG_dex] = _warp_reduce_res;
        }
    }
}




__global__ void 
decx::reduce::GPUK::cu_warp_reduce_sum2D_flatten_fp16_L1(const float4 * __restrict   src, 
                                                      float* __restrict          dst,
                                                      const uint32_t              Wsrc_v4, 
                                                      const uint2                 proc_dims_v1)
{
    uint32_t tidx = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t tidy = threadIdx.y + blockDim.y * blockIdx.y;

    uint64_t LDG_dex = Wsrc_v4 * tidy + tidx;
    uint64_t STG_dex = blockIdx.y * gridDim.x + blockIdx.x;
    
    uint32_t proc_W_v8 = decx::utils::ceil<uint32_t>(proc_dims_v1.x, 8);

    /*
    * Shared memory for the reduced results of 8 warps in a block
    * Extended to 32 for warp-level loading
    * No need to set the remaining values to zero since for the warp reducing
    * all the redundancy will not invovled with correct parameters set
    */
    __shared__ float _sh_warp_reduce_res[32];

    decx::utils::_cuda_vec128 _recv;
    _recv._vf = decx::utils::vec4_set1_fp32(0);

    float _thread_sum = 0, _warp_reduce_res = 0;

    if (tidx < proc_W_v8 && tidy < proc_dims_v1.y) {
        _recv._vf = src[LDG_dex];
        if (tidx == proc_W_v8 - 1) {
            for (int i = 8 - (proc_W_v8 * 8 - proc_dims_v1.x); i < 8; ++i) {
                _recv._arrf[i] = 0.f;
            }
        }
    }

    _thread_sum = decx::reduce::GPUK::half8_sum(_recv._vf);

    decx::reduce::GPUK::cu_warp_reduce<float, 32>(__fadd_rn, &_thread_sum, &_warp_reduce_res);

    if (threadIdx.x == 0) {
        _sh_warp_reduce_res[threadIdx.y] = _warp_reduce_res;
    }

    __syncthreads();

    if (threadIdx.y == 0) {
        _thread_sum = _sh_warp_reduce_res[threadIdx.x];
        __syncwarp(0xffffffff);

        decx::reduce::GPUK::cu_warp_reduce<float, 8>(__fadd_rn, &_thread_sum, &_warp_reduce_res);

        if (threadIdx.x == 0) {
            dst[STG_dex] = _warp_reduce_res;
        }
    }
}



__global__ void 
decx::reduce::GPUK::cu_warp_reduce_sum2D_flatten_fp16_L2(const float4 * __restrict   src, 
                                                         __half* __restrict          dst,
                                                         const uint32_t              Wsrc_v4, 
                                                         const uint2                 proc_dims_v1)
{
    uint32_t tidx = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t tidy = threadIdx.y + blockDim.y * blockIdx.y;

    uint64_t LDG_dex = Wsrc_v4 * tidy + tidx;
    uint64_t STG_dex = blockIdx.y * gridDim.x + blockIdx.x;
    
    uint32_t proc_W_v8 = decx::utils::ceil<uint32_t>(proc_dims_v1.x, 8);

    /*
    * Shared memory for the reduced results of 8 warps in a block
    * Extended to 32 for warp-level loading
    * No need to set the remaining values to zero since for the warp reducing
    * all the redundancy will not invovled with correct parameters set
    */
    __shared__ float _sh_warp_reduce_res[32];

    decx::utils::_cuda_vec128 _recv;
    _recv._vf = decx::utils::vec4_set1_fp32(0);

    float _thread_sum = 0, _warp_reduce_res = 0;

    if (tidx < proc_W_v8 && tidy < proc_dims_v1.y) {
        _recv._vf = src[LDG_dex];
        if (tidx == proc_W_v8 - 1) {
            for (int i = 8 - (proc_W_v8 * 8 - proc_dims_v1.x); i < 8; ++i) {
                _recv._arrf[i] = 0.f;
            }
        }
    }

    _thread_sum = decx::reduce::GPUK::half8_sum(_recv._vf);

    decx::reduce::GPUK::cu_warp_reduce<float, 32>(__fadd_rn, &_thread_sum, &_warp_reduce_res);

    if (threadIdx.x == 0) {
        _sh_warp_reduce_res[threadIdx.y] = _warp_reduce_res;
    }

    __syncthreads();

    if (threadIdx.y == 0) {
        _thread_sum = _sh_warp_reduce_res[threadIdx.x];
        __syncwarp(0xffffffff);

        decx::reduce::GPUK::cu_warp_reduce<float, 8>(__fadd_rn, &_thread_sum, &_warp_reduce_res);

        if (threadIdx.x == 0) {
            dst[STG_dex] = __float2half_rn(_warp_reduce_res);
        }
    }
}





__global__ void 
decx::reduce::GPUK::cu_warp_reduce_sum2D_flatten_fp16_L3(const float4 * __restrict   src, 
                                                         __half* __restrict          dst,
                                                         const uint32_t              Wsrc_v4, 
                                                         const uint2                 proc_dims_v1)
{
    uint32_t tidx = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t tidy = threadIdx.y + blockDim.y * blockIdx.y;

    uint64_t LDG_dex = Wsrc_v4 * tidy + tidx;
    uint64_t STG_dex = blockIdx.y * gridDim.x + blockIdx.x;
    
    uint32_t proc_W_v8 = decx::utils::ceil<uint32_t>(proc_dims_v1.x, 8);

    /*
    * Shared memory for the reduced results of 8 warps in a block
    * Extended to 32 for warp-level loading
    * No need to set the remaining values to zero since for the warp reducing
    * all the redundancy will not invovled with correct parameters set
    */
    __shared__ __half _sh_warp_reduce_res[32];

    decx::utils::_cuda_vec128 _recv;
    _recv._vf = decx::utils::vec4_set1_fp32(0);

    __half _thread_sum = __float2half_rn(0), _warp_reduce_res = __float2half_rn(0);

    if (tidx < proc_W_v8 && tidy < proc_dims_v1.y) {
        _recv._vf = src[LDG_dex];
        if (tidx == proc_W_v8 - 1) {
            for (int i = 8 - (proc_W_v8 * 8 - proc_dims_v1.x); i < 8; ++i) {
                _recv._arrf[i] = 0.f;
            }
        }
    }

    _thread_sum = __float2half_rn(decx::reduce::GPUK::half8_sum(_recv._vf));

    decx::reduce::GPUK::cu_warp_reduce<__half, 32>(__hadd, &_thread_sum, &_warp_reduce_res);

    if (threadIdx.x == 0) {
        _sh_warp_reduce_res[threadIdx.y] = _warp_reduce_res;
    }

    __syncthreads();

    if (threadIdx.y == 0) {
        _thread_sum = _sh_warp_reduce_res[threadIdx.x];
        __syncwarp(0xffffffff);

        decx::reduce::GPUK::cu_warp_reduce<__half, 8>(__hadd, &_thread_sum, &_warp_reduce_res);

        if (threadIdx.x == 0) {
            dst[STG_dex] = _warp_reduce_res;
        }
    }
}




__global__ void 
decx::reduce::GPUK::cu_warp_reduce_sum2D_flatten_u8_i32(const int4 * __restrict     src, 
                                                        int32_t* __restrict         dst,
                                                        const uint32_t              Wsrc_v4, 
                                                        const uint2                 proc_dims_v1)
{
    uint32_t tidx = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t tidy = threadIdx.y + blockDim.y * blockIdx.y;

    uint64_t LDG_dex = Wsrc_v4 * tidy + tidx;
    uint64_t STG_dex = blockIdx.y * gridDim.x + blockIdx.x;
    
    uint32_t proc_W_v16 = decx::utils::ceil<uint32_t>(proc_dims_v1.x, 16);

    /*
    * Shared memory for the reduced results of 8 warps in a block
    * Extended to 32 for warp-level loading
    * No need to set the remaining values to zero since for the warp reducing
    * all the redundancy will not invovled with correct parameters set
    */
    __shared__ int32_t _sh_warp_reduce_res[32];

    decx::utils::_cuda_vec128 _recv;
    _recv._vi = decx::utils::vec4_set1_int32(0);

    int32_t _thread_sum = 0, _warp_reduce_res = 0;

    if (tidx < proc_W_v16 && tidy < proc_dims_v1.y) {
        _recv._vi = src[LDG_dex];
        /*
        * Because of the vec-load, some don't care values will be loaded
        * at the end of the thread grid (at the end of the process area as well)
        * For reduced summation process, set the don't care value(s) to all zero
        * to eliminate their effect
        */
        if (tidx == proc_W_v16 - 1) {
            uint32_t _left_u8 = proc_W_v16 * 16 - proc_dims_v1.x;

            for (int i = 4 - (_left_u8 / 4); i < 4; ++i) {
                _recv._arri[i] = 0;
            }
            int32_t tmp_frame = _recv._arri[3 - (_left_u8 / 4)];
            // [0, 0, 0, 0] [val, val, val, val] -> [4 5 6 7] & [offset]
            _recv._arri[3 - (_left_u8 / 4)] = __byte_perm(0, tmp_frame, (0xffff >> (4 * (_left_u8 % 4))) & 0x7654);
        }
    }

    _thread_sum = decx::reduce::GPUK::uchar16_sum(_recv._vi);

    decx::reduce::GPUK::cu_warp_reduce<int32_t, 32>(decx::utils::cuda::__i32_add, &_thread_sum, &_warp_reduce_res);

    if (threadIdx.x == 0) {
        _sh_warp_reduce_res[threadIdx.y] = _warp_reduce_res;
    }

    __syncthreads();

    if (threadIdx.y == 0) {
        _thread_sum = _sh_warp_reduce_res[threadIdx.x];
        __syncwarp(0xffffffff);

        decx::reduce::GPUK::cu_warp_reduce<int32_t, 8>(decx::utils::cuda::__i32_add, &_thread_sum, &_warp_reduce_res);

        if (threadIdx.x == 0) {
            dst[STG_dex] = _warp_reduce_res;
        }
    }
}




__global__ void
decx::reduce::GPUK::cu_block_reduce_sum1D_fp16_L1(const float4* __restrict     src, 
                                                    float* __restrict            dst, 
                                                    const uint64_t               proc_len_v8,
                                                    const uint64_t               proc_len_v1)
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

    decx::utils::_cuda_vec128 _recv;
    _recv._vf = decx::utils::vec4_set1_fp32(0);
    float tmp1, tmp2;

    if (LDG_dex < proc_len_v8) {
        _recv._vf = src[LDG_dex];
        if (LDG_dex == proc_len_v8 - 1) {
            for (int i = 8 - (proc_len_v8 * 8 - proc_len_v1); i < 8; ++i) {
                _recv._arrs[i] = 0;
            }
        }
    }

    tmp1 = decx::reduce::GPUK::half8_sum(_recv._vf);
    // warp reducing
    //decx::reduce::GPUK::cu_warp_reduce_fp32<float(float, float), 32>(__fadd_rn, &tmp1, &tmp2);
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

        //decx::reduce::GPUK::cu_warp_reduce_fp32<float(float, float), 8>(__fadd_rn, &tmp1, &tmp2);
        decx::reduce::GPUK::cu_warp_reduce<float, 8>(__fadd_rn, &tmp1, &tmp2);

        if (warp_lane_id == 0) {
            dst[blockIdx.x] = tmp2;
        }
    }
}



__global__ void
decx::reduce::GPUK::cu_block_reduce_sum1D_fp16_L2(const float4* __restrict     src, 
                                                  __half* __restrict           dst, 
                                                  const uint64_t               proc_len_v8,
                                                  const uint64_t               proc_len_v1)
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

    decx::utils::_cuda_vec128 _recv;
    _recv._vf = decx::utils::vec4_set1_fp32(0);
    float tmp1, tmp2;

    if (LDG_dex < proc_len_v8) {
        _recv._vf = src[LDG_dex];
        if (LDG_dex == proc_len_v8 - 1) {
            for (int i = 8 - (proc_len_v8 * 8 - proc_len_v1); i < 8; ++i) {
                _recv._arrs[i] = 0;
            }
        }
    }

    tmp1 = decx::reduce::GPUK::half8_sum(_recv._vf);
    // warp reducing
    //decx::reduce::GPUK::cu_warp_reduce_fp32<float(float, float), 32>(__fadd_rn, &tmp1, &tmp2);
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

        //decx::reduce::GPUK::cu_warp_reduce_fp32<float(float, float), 8>(__fadd_rn, &tmp1, &tmp2);
        decx::reduce::GPUK::cu_warp_reduce<float, 8>(__fadd_rn, &tmp1, &tmp2);

        if (warp_lane_id == 0) {
            dst[blockIdx.x] = __float2half_rn(tmp2);
        }
    }
}


__global__ void
decx::reduce::GPUK::cu_block_reduce_sum1D_fp16_L3(const float4* __restrict     src, 
                                                  __half* __restrict           dst, 
                                                  const uint64_t               proc_len_v8,
                                                  const uint64_t               proc_len_v1)
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
    __shared__ __half warp_reduce_results[32];

    decx::utils::_cuda_vec128 _recv;
    _recv._vf = decx::utils::vec4_set1_fp32(0);
    __half tmp1, tmp2;

    if (LDG_dex < proc_len_v8) {
        _recv._vf = src[LDG_dex];
        if (LDG_dex == proc_len_v8 - 1) {
            for (int i = 8 - (proc_len_v8 * 8 - proc_len_v1); i < 8; ++i) {
                _recv._arrs[i] = 0;
            }
        }
    }

    tmp1 = decx::reduce::GPUK::half8_sum(_recv._vf);
    // warp reducing
    //decx::reduce::GPUK::cu_warp_reduce_fp32<float(float, float), 32>(__fadd_rn, &tmp1, &tmp2);
    decx::reduce::GPUK::cu_warp_reduce<__half, 32>(__hadd, &tmp1, &tmp2);

    if (warp_lane_id == 0) {
        warp_reduce_results[local_warp_id] = tmp2;
    }

    __syncthreads();

    // let the 0th warp execute the warp-reducing process
    if (local_warp_id == 0) {
        tmp1 = warp_reduce_results[warp_lane_id];
        // synchronize this warp
        __syncwarp(0xffffffff);

        //decx::reduce::GPUK::cu_warp_reduce_fp32<float(float, float), 8>(__fadd_rn, &tmp1, &tmp2);
        decx::reduce::GPUK::cu_warp_reduce<__half, 8>(__hadd, &tmp1, &tmp2);

        if (warp_lane_id == 0) {
            dst[blockIdx.x] = tmp2;
        }
    }
}



__global__ void
decx::reduce::GPUK::cu_block_reduce_sum1D_int32(const int4* __restrict     src, 
                                               int* __restrict             dst, 
                                               const uint64_t               proc_len_v4,
                                               const uint64_t               proc_len_v1)
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
    __shared__ int32_t warp_reduce_results[32];

    decx::utils::_cuda_vec128 _recv;
    _recv._vi = decx::utils::vec4_set1_int32(0);
    int32_t tmp1, tmp2;

    if (LDG_dex < proc_len_v4) 
    {
        _recv._vi = src[LDG_dex];
        if (LDG_dex == proc_len_v4 - 1) {
            for (int i = 4 - (proc_len_v4 * 4 - proc_len_v1); i < 4; ++i) {
                _recv._arri[i] = 0;
            }
        }
    }

    tmp1 = decx::reduce::GPUK::int4_sum(_recv._vi);
    // warp reducing
    //decx::reduce::GPUK::cu_warp_reduce_int32<int32_t(int32_t, int32_t), 32>(decx::utils::cuda::__i32_add, &tmp1, &tmp2);
    decx::reduce::GPUK::cu_warp_reduce<int32_t, 32>(decx::utils::cuda::__i32_add, &tmp1, &tmp2);

    if (warp_lane_id == 0) {
        warp_reduce_results[local_warp_id] = tmp2;
    }

    __syncthreads();

    // let the 0th warp execute the warp-reducing process
    if (local_warp_id == 0) {
        tmp1 = warp_reduce_results[warp_lane_id];
        // synchronize this warp
        __syncwarp(0xffffffff);

        //decx::reduce::GPUK::cu_warp_reduce_int32<int32_t(int32_t, int32_t), 8>(decx::utils::cuda::__i32_add, &tmp1, &tmp2);
        decx::reduce::GPUK::cu_warp_reduce<int32_t, 8>(decx::utils::cuda::__i32_add, &tmp1, &tmp2);

        if (warp_lane_id == 0) {
            dst[blockIdx.x] = tmp2;
        }
    }
}



__global__ void
decx::reduce::GPUK::cu_block_reduce_sum1D_u8_i32(const int4* __restrict       src, 
                                                 int32_t* __restrict          dst, 
                                                 const uint64_t               proc_len_v16,
                                                 const uint64_t               proc_len_v1)
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
    __shared__ int32_t warp_reduce_results[32];

    decx::utils::_cuda_vec128 _recv;
    _recv._vf = decx::utils::vec4_set1_fp32(0);
    int32_t tmp1, tmp2;

    if (LDG_dex < proc_len_v16)
    {
        _recv._vi = src[LDG_dex];
        /*
        * Because of the vec-load, some don't care values will be loaded
        * at the end of the thread grid (at the end of the process area as well)
        * For reduced summation process, set the don't care value(s) to all zero
        * to eliminate their effect
        */
        if (LDG_dex == proc_len_v16 - 1) {
            uint32_t _left_u8 = proc_len_v16 * 16 - proc_len_v1;

            for (int i = 4 - (_left_u8 / 4); i < 4; ++i) {
                _recv._arri[i] = 0;
            }

            int32_t tmp_frame = _recv._arri[3 - (_left_u8 / 4)];
            // [0, 0, 0, 0] [val, val, val, val] -> [4 5 6 7] & [offset]
            _recv._arri[3 - (_left_u8 / 4)] = __byte_perm(0, tmp_frame, (0xffff >> (4 * (_left_u8 % 4))) & 0x7654);
        }
    }

    tmp1 = decx::reduce::GPUK::uchar16_sum(_recv._vi);
    //decx::reduce::GPUK::cu_warp_reduce_int32<int32_t(int32_t, int32_t), 32>(decx::utils::cuda::__i32_add, &tmp1, &tmp2);
    decx::reduce::GPUK::cu_warp_reduce<int32_t, 32>(decx::utils::cuda::__i32_add, &tmp1, &tmp2);

    if (warp_lane_id == 0) {
        warp_reduce_results[local_warp_id] = tmp2;
    }

    __syncthreads();

    // let the 0th warp execute the warp-reducing process
    if (local_warp_id == 0) {
        tmp1 = warp_reduce_results[warp_lane_id];
        // synchronize this warp
        __syncwarp(0xffffffff);

        //decx::reduce::GPUK::cu_warp_reduce_int32<int32_t(int32_t, int32_t), 8>(decx::utils::cuda::__i32_add, &tmp1, &tmp2);
        decx::reduce::GPUK::cu_warp_reduce<int32_t, 8>(decx::utils::cuda::__i32_add, &tmp1, &tmp2);

        if (warp_lane_id == 0) {
            dst[blockIdx.x] = tmp2;
        }
    }
}