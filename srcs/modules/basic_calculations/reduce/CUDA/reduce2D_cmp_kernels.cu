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


#include "reduce_cmp.cuh"


template <bool _is_max> __global__ void
decx::reduce::GPUK::cu_warp_reduce_cmp2D_h_fp32(const float4 * __restrict   src, 
                                                float* __restrict           dst,
                                                const uint32_t              Wsrc_v4, 
                                                uint32_t                    Wdst_v1, 
                                                const uint2                 proc_dims)
{
    uint32_t tidx = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t tidy = threadIdx.y + blockDim.y * blockIdx.y;

    uint64_t LDG_dex = Wsrc_v4 * tidy + tidx;
    uint64_t STG_dex = Wdst_v1 * tidy + blockIdx.x;

    uint32_t proc_W_v4 = decx::utils::ceil<uint32_t>(proc_dims.x, 4);

    float _one_of_element;
    // The first thread of a warp load the value from the very beginning of the matrix of each row
    if (threadIdx.x == 0) {
        if (tidy < proc_dims.y) {
            _one_of_element = *((float*)(src + Wsrc_v4 * tidy));
        }
    }
    __syncwarp(0xffffffff);

    // Sync the data for the remaining 31 threads in a warp
    _one_of_element = __shfl_sync(0xffffffff, _one_of_element, 0, 32);

    decx::utils::_cuda_vec128 _recv;
    _recv._vf = decx::utils::vec4_set1_fp32(_one_of_element);

    float _thread_sum = 0, _warp_reduce_res = 0;

    if (tidx < proc_W_v4 && tidy < proc_dims.y) {
        _recv._vf = src[LDG_dex];
        if (tidx == proc_W_v4 - 1) {
            for (int i = 4 - (proc_W_v4 * 4 - proc_dims.x); i < 4; ++i) {
                _recv._arrf[i] = _one_of_element;
            }
        }
    }

    if (_is_max) {
        _thread_sum = decx::reduce::GPUK::float4_max(_recv._vf);
        decx::reduce::GPUK::cu_warp_reduce_fp32<float(float, float), 32>(decx::utils::cuda::__fp32_max, &_thread_sum, &_warp_reduce_res);
    }
    else {
        _thread_sum = decx::reduce::GPUK::float4_min(_recv._vf);
        decx::reduce::GPUK::cu_warp_reduce_fp32<float(float, float), 32>(decx::utils::cuda::__fp32_min, &_thread_sum, &_warp_reduce_res);
    }

    if (threadIdx.x == 0 && tidy < proc_dims.y) {
        dst[STG_dex] = _warp_reduce_res;
    }
}


template __global__ void decx::reduce::GPUK::cu_warp_reduce_cmp2D_h_fp32<true>(const float4* __restrict, float* __restrict, const uint32_t, uint32_t, const uint2);
template __global__ void decx::reduce::GPUK::cu_warp_reduce_cmp2D_h_fp32<false>(const float4* __restrict, float* __restrict, const uint32_t, uint32_t, const uint2);



template <bool _is_max> __global__ void
decx::reduce::GPUK::cu_warp_reduce_cmp2D_h_fp16(const float4 * __restrict   src, 
                                                __half* __restrict          dst,
                                                const uint32_t              Wsrc_v8, 
                                                uint32_t                    Wdst_v1, 
                                                const uint2                 proc_dims)
{
#if __ABOVE_SM_53
    uint32_t tidx = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t tidy = threadIdx.y + blockDim.y * blockIdx.y;

    uint64_t LDG_dex = Wsrc_v8 * tidy + tidx;
    uint64_t STG_dex = Wdst_v1 * tidy + blockIdx.x;

    uint32_t proc_W_v8 = decx::utils::ceil<uint32_t>(proc_dims.x, 8);

    __half _one_of_element;
    // The first thread of a warp load the value from the very beginning of the matrix of each row
    if (threadIdx.x == 0) {
        if (tidy < proc_dims.y) {
            _one_of_element = *((__half*)(src + Wsrc_v8 * tidy));
        }
    }
    __syncwarp(0xffffffff);

    // Sync the data for the remaining 31 threads in a warp
    _one_of_element = __shfl_sync(0xffffffff, _one_of_element, 0, 32);

    decx::utils::_cuda_vec128 _recv;
    _recv._vf = decx::utils::vec8_set1_fp16(_one_of_element);

    __half _thread_sum, _warp_reduce_res;

    if (tidx < proc_W_v8 && tidy < proc_dims.y) {
        _recv._vf = src[LDG_dex];
        if (tidx == proc_W_v8 - 1) {
            for (int i = 8 - (proc_W_v8 * 8 - proc_dims.x); i < 8; ++i) {
                _recv._arrs[i] = *((ushort*)&_one_of_element);
            }
        }
    }
    if (_is_max) {
        _thread_sum = decx::reduce::GPUK::half8_max(_recv._arrh2);
    }
    else {
        _thread_sum = decx::reduce::GPUK::half8_min(_recv._arrh2);
    }

    if (_is_max) {
        decx::reduce::GPUK::cu_warp_reduce<__half, 32>(decx::utils::cuda::__half_max, &_thread_sum, &_warp_reduce_res);
    }
    else {
        decx::reduce::GPUK::cu_warp_reduce<__half, 32>(decx::utils::cuda::__half_min, &_thread_sum, &_warp_reduce_res);
    }

    if (threadIdx.x == 0 && tidy < proc_dims.y) {
        dst[STG_dex] = _warp_reduce_res;
    }
#endif
}


template __global__ void decx::reduce::GPUK::cu_warp_reduce_cmp2D_h_fp16<true>(const float4* __restrict, __half* __restrict, const uint32_t, uint32_t, const uint2);
template __global__ void decx::reduce::GPUK::cu_warp_reduce_cmp2D_h_fp16<false>(const float4* __restrict, __half* __restrict, const uint32_t, uint32_t, const uint2);





template <bool _is_max> __global__ void 
decx::reduce::GPUK::cu_warp_reduce_cmp2D_h_u8(const int4 * __restrict     src, 
                                              uint8_t* __restrict         dst,
                                              const uint32_t              Wsrc_v16, 
                                              uint32_t                    Wdst_v1, 
                                              const uint2                 proc_dims)
{
    uint32_t tidx = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t tidy = threadIdx.y + blockDim.y * blockIdx.y;

    uint64_t LDG_dex = Wsrc_v16 * tidy + tidx;
    uint64_t STG_dex = Wdst_v1 * tidy + blockIdx.x;

    uint32_t proc_W_v16 = decx::utils::ceil<uint32_t>(proc_dims.x, 16);

    uint8_t _one_of_element;
    // The first thread of a warp load the value from the very beginning of the matrix of each row
    if (threadIdx.x == 0) {
        if (tidy < proc_dims.y) {
            _one_of_element = *((uint8_t*)(src + Wsrc_v16 * tidy));
        }
    }
    __syncwarp(0xffffffff);

    // Sync the data for the remaining 31 threads in a warp
    _one_of_element = __shfl_sync(0xffffffff, _one_of_element, 0, 32);

    decx::utils::_cuda_vec128 _recv;
    _recv._vi = decx::utils::vec16_set1_u8(_one_of_element);

    uint32_t _thread_sum, _warp_reduce_res;

    if (tidx < proc_W_v16 && tidy < proc_dims.y) {
        _recv._vi = src[LDG_dex];
        /*
        * Because of the vec-load, some don't care values will be loaded
        * at the end of the thread grid (at the end of the process area as well)
        * For reduced summation process, set the don't care value(s) to all zero
        * to eliminate their effect
        */
        if (tidx == proc_W_v16 - 1) {
            uint32_t _left_u8 = proc_W_v16 * 16 - proc_dims.x;
            uchar4 _fill_vec4 = make_uchar4(_one_of_element, _one_of_element, _one_of_element, _one_of_element);

            for (int i = 4 - (_left_u8 / 4); i < 4; ++i) {
                _recv._arri[i] = *((int32_t*)&_fill_vec4);
            }
            int32_t tmp_frame = _recv._arri[3 - (_left_u8 / 4)];
            // [0, 0, 0, 0] [val, val, val, val] -> [4 5 6 7] & [offset]
            _recv._arri[3 - (_left_u8 / 4)] = __byte_perm(*((int32_t*)&_fill_vec4), tmp_frame, (0xffff >> (4 * (_left_u8 % 4))) & 0x7654);
        }
    }

    if (_is_max) {
        _thread_sum = (uint32_t)decx::reduce::GPUK::uchar16_max(_recv._vi);
        decx::reduce::GPUK::cu_warp_reduce<int32_t, 32>(decx::utils::cuda::__i32_max, &_thread_sum, &_warp_reduce_res);
    }
    else {
        _thread_sum = (uint32_t)decx::reduce::GPUK::uchar16_min(_recv._vi);
        decx::reduce::GPUK::cu_warp_reduce<int32_t, 32>(decx::utils::cuda::__i32_min, &_thread_sum, &_warp_reduce_res);
    }

    if (threadIdx.x == 0 && tidy < proc_dims.y) {
        dst[STG_dex] = (uint8_t)_warp_reduce_res;
    }
}

template __global__ void decx::reduce::GPUK::cu_warp_reduce_cmp2D_h_u8<true>(const int4* __restrict, uint8_t* __restrict, const uint32_t, uint32_t, const uint2);
template __global__ void decx::reduce::GPUK::cu_warp_reduce_cmp2D_h_u8<false>(const int4* __restrict, uint8_t* __restrict, const uint32_t, uint32_t, const uint2);




//
//template <bool _is_max>  __global__ void
//decx::reduce::GPUK::cu_warp_reduce_cmp2D_h_fp32_transp(const float4 * __restrict   src, 
//                                                       float* __restrict           dst,
//                                                       const uint32_t              Wsrc_v4, 
//                                                       uint32_t                    Wdst_v1, 
//                                                       const uint2                 proc_dims)
//{
//    uint32_t tidx = threadIdx.x + blockDim.x * blockIdx.x;
//    uint32_t tidy = threadIdx.y + blockDim.y * blockIdx.y;
//
//    uint64_t LDG_dex = Wsrc_v4 * tidy + tidx;
//    uint64_t STG_dex = Wdst_v1 * blockIdx.x + tidy;
//
//    uint32_t proc_W_v4 = decx::utils::ceil<uint32_t>(proc_dims.x, 4);
//
//    float _one_of_element;
//    // The first thread of a warp load the value from the very beginning of the matrix of each row
//    if (threadIdx.x == 0) {
//        if (tidy < proc_dims.y) {
//            _one_of_element = *((float*)(src + Wsrc_v4 * tidy));
//        }
//    }
//
//    __syncwarp(0xffffffff);
//
//    // Sync the data for the remaining 31 threads in a warp
//    _one_of_element = __shfl_sync(0xffffffff, _one_of_element, 0, 32);
//
//    decx::utils::_cuda_vec128 _recv;
//    _recv._vf = decx::utils::vec4_set1_fp32(_one_of_element);
//
//    float _thread_sum = 0, _warp_reduce_res = 0;
//
//    if (tidx < proc_W_v4 && tidy < proc_dims.y) {
//        _recv._vf = src[LDG_dex];
//        if (tidx == proc_W_v4 - 1) {
//            for (int i = 4 - (proc_W_v4 * 4 - proc_dims.x); i < 4; ++i) {
//                _recv._arrf[i] = _one_of_element;
//            }
//        }
//    }
//
//    if (_is_max) {
//        _thread_sum = decx::reduce::GPUK::float4_max(_recv._vf);
//        decx::reduce::GPUK::cu_warp_reduce_fp32<float(float, float), 32>(decx::utils::cuda::__fp32_max, &_thread_sum, &_warp_reduce_res);
//    }
//    else {
//        _thread_sum = decx::reduce::GPUK::float4_min(_recv._vf);
//        decx::reduce::GPUK::cu_warp_reduce_fp32<float(float, float), 32>(decx::utils::cuda::__fp32_min, &_thread_sum, &_warp_reduce_res);
//    }
//
//    if (threadIdx.x == 0 && tidy < proc_dims.y) {
//        dst[STG_dex] = _warp_reduce_res;
//    }
//}
//
//template __global__ void decx::reduce::GPUK::cu_warp_reduce_cmp2D_h_fp32_transp<true>(const float4* __restrict, float* __restrict, const uint32_t, uint32_t, const uint2);
//template __global__ void decx::reduce::GPUK::cu_warp_reduce_cmp2D_h_fp32_transp<false>(const float4* __restrict, float* __restrict, const uint32_t, uint32_t, const uint2);
//
//
//
//
//
//template <bool _is_max>  __global__ void
//decx::reduce::GPUK::cu_warp_reduce_cmp2D_h_fp16_transp(const float4 * __restrict   src, 
//                                                       __half* __restrict          dst,
//                                                       const uint32_t              Wsrc_v8, 
//                                                       uint32_t                    Wdst_v1, 
//                                                       const uint2                 proc_dims)
//{
//    uint32_t tidx = threadIdx.x + blockDim.x * blockIdx.x;
//    uint32_t tidy = threadIdx.y + blockDim.y * blockIdx.y;
//
//    uint64_t LDG_dex = Wsrc_v8 * tidy + tidx;
//    uint64_t STG_dex = Wdst_v1 * blockIdx.x + tidy;
//
//    uint32_t proc_W_v8 = decx::utils::ceil<uint32_t>(proc_dims.x, 8);
//
//    __half _one_of_element;
//    // The first thread of a warp load the value from the very beginning of the matrix of each row
//    if (threadIdx.x == 0) {
//        if (tidy < proc_dims.y) {
//            _one_of_element = *((__half*)(src + Wsrc_v8 * tidy));
//        }
//    }
//
//    __syncwarp(0xffffffff);
//
//    // Sync the data for the remaining 31 threads in a warp
//    _one_of_element = __shfl_sync(0xffffffff, _one_of_element, 0, 32);
//
//    decx::utils::_cuda_vec128 _recv;
//    _recv._vf = decx::utils::vec8_set1_fp16(_one_of_element);
//
//    __half _thread_sum, _warp_reduce_res;
//
//    if (tidx < proc_W_v8 && tidy < proc_dims.y) {
//        _recv._vf = src[LDG_dex];
//        if (tidx == proc_W_v8 - 1) {
//            for (int i = 8 - (proc_W_v8 * 8 - proc_dims.x); i < 8; ++i) {
//                _recv._arrf[i] = _one_of_element;
//            }
//        }
//    }
//
//    if (_is_max) {
//        _thread_sum = decx::reduce::GPUK::half8_max(_recv._arrh2);
//        decx::reduce::GPUK::cu_warp_reduce<__half, 32>(decx::utils::cuda::__half_max, &_thread_sum, &_warp_reduce_res);
//    }
//    else {
//        _thread_sum = decx::reduce::GPUK::half8_min(_recv._arrh2);
//        decx::reduce::GPUK::cu_warp_reduce<__half, 32>(decx::utils::cuda::__half_min, &_thread_sum, &_warp_reduce_res);
//    }
//
//    if (threadIdx.x == 0 && tidy < proc_dims.y) {
//        dst[STG_dex] = _warp_reduce_res;
//    }
//}
//
//template __global__ void decx::reduce::GPUK::cu_warp_reduce_cmp2D_h_fp16_transp<true>(const float4* __restrict, __half* __restrict, const uint32_t, uint32_t, const uint2);
//template __global__ void decx::reduce::GPUK::cu_warp_reduce_cmp2D_h_fp16_transp<false>(const float4* __restrict, __half* __restrict, const uint32_t, uint32_t, const uint2);
//
//
//
//template <bool _is_max>  __global__ void
//decx::reduce::GPUK::cu_warp_reduce_cmp2D_h_u8_transp(const int4 * __restrict     src, 
//                                                     uint8_t* __restrict         dst,
//                                                     const uint32_t              Wsrc_v16, 
//                                                     uint32_t                    Wdst_v1, 
//                                                     const uint2                 proc_dims)
//{
//    uint32_t tidx = threadIdx.x + blockDim.x * blockIdx.x;
//    uint32_t tidy = threadIdx.y + blockDim.y * blockIdx.y;
//
//    uint64_t LDG_dex = Wsrc_v16 * tidy + tidx;
//    uint64_t STG_dex = Wdst_v1 * blockIdx.x + tidy;
//
//    uint32_t proc_W_v16 = decx::utils::ceil<uint32_t>(proc_dims.x, 16);
//
//    uint8_t _one_of_element;
//    // The first thread of a warp load the value from the very beginning of the matrix of each row
//    if (threadIdx.x == 0) {
//        if (tidy < proc_dims.y) {
//            _one_of_element = *((uint8_t*)(src + Wsrc_v16 * tidy));
//        }
//    }
//    __syncwarp(0xffffffff);
//
//    // Sync the data for the remaining 31 threads in a warp
//    _one_of_element = __shfl_sync(0xffffffff, _one_of_element, 0, 32);
//
//    decx::utils::_cuda_vec128 _recv;
//    _recv._vi = decx::utils::vec16_set1_u8(_one_of_element);
//
//    int32_t _thread_sum, _warp_reduce_res;
//
//    if (tidx < proc_W_v16 && tidy < proc_dims.y) {
//        _recv._vi = src[LDG_dex];
//        /*
//        * Because of the vec-load, some don't care values will be loaded
//        * at the end of the thread grid (at the end of the process area as well)
//        * For reduced summation process, set the don't care value(s) to all zero
//        * to eliminate their effect
//        */
//        if (tidx == proc_W_v16 - 1) {
//            uint32_t _left_u8 = proc_W_v16 * 16 - proc_dims.x;
//            uchar4 _fill_vec4 = make_uchar4(_one_of_element, _one_of_element, _one_of_element, _one_of_element);
//
//            for (int i = 4 - (_left_u8 / 4); i < 4; ++i) {
//                _recv._arri[i] = *((int32_t*)&_fill_vec4);
//            }
//            int32_t tmp_frame = _recv._arri[3 - (_left_u8 / 4)];
//            // [0, 0, 0, 0] [val, val, val, val] -> [4 5 6 7] & [offset]
//            _recv._arri[3 - (_left_u8 / 4)] = __byte_perm(*((int32_t*)&_fill_vec4), tmp_frame, (0xffff >> (4 * (_left_u8 % 4))) & 0x7654);
//        }
//    }
//
//    if (_is_max) {
//        _thread_sum = (int32_t)decx::reduce::GPUK::uchar16_max(_recv._vi);
//        decx::reduce::GPUK::cu_warp_reduce<int32_t, 32>(decx::utils::cuda::__i32_max, &_thread_sum, &_warp_reduce_res);
//    }
//    else {
//        _thread_sum = (int32_t)decx::reduce::GPUK::uchar16_min(_recv._vi);
//        decx::reduce::GPUK::cu_warp_reduce<int32_t, 32>(decx::utils::cuda::__i32_min, &_thread_sum, &_warp_reduce_res);
//    }
//
//    if (threadIdx.x == 0 && tidy < proc_dims.y) {
//        dst[STG_dex] = (uint8_t)_warp_reduce_res;
//    }
//}
//
//template __global__ void decx::reduce::GPUK::cu_warp_reduce_cmp2D_h_u8_transp<true>(const int4* __restrict, uint8_t* __restrict, const uint32_t, uint32_t, const uint2);
//template __global__ void decx::reduce::GPUK::cu_warp_reduce_cmp2D_h_u8_transp<false>(const int4* __restrict, uint8_t* __restrict, const uint32_t, uint32_t, const uint2);
//
//
//



template <bool _is_max> __global__ void
decx::reduce::GPUK::cu_warp_reduce_cmp2D_v_fp32(const float4 * __restrict   src, 
                                                float4* __restrict          dst,
                                                const uint32_t              Wsrc_v4, 
                                                uint32_t                    Wdst_v4, 
                                                const uint2                 proc_dims_v4)
{
    uint32_t tidx = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t tidy = threadIdx.y + blockDim.y * blockIdx.y;

    uint64_t LDG_dex = Wsrc_v4 * tidy + tidx;
    uint64_t STG_dex = blockIdx.y * Wdst_v4 + tidx;

    __shared__ float4 _workspace[8][32 + 1];

    decx::utils::_cuda_vec128 _recv;
    _recv._vf = decx::utils::vec4_set1_fp32(0);

    if (threadIdx.y == 0) {
        if (tidx < proc_dims_v4.x) {
            _recv._vf = src[tidx];
        }
        _workspace[0][threadIdx.x] = _recv._vf;
    }

    __syncthreads();
    _recv._vf = _workspace[0][threadIdx.x];
    __syncthreads();

    float2 tmp1, tmp2, tmp3, tmp4;
    
    if (tidx < proc_dims_v4.x && tidy < proc_dims_v4.y) {
        _recv._vf = src[LDG_dex];
    }

    _workspace[threadIdx.y][threadIdx.x] = _recv._vf;

    __syncthreads();

    tmp1 = ((float2*)_workspace[threadIdx.x % 8])[threadIdx.y * 4 + threadIdx.x / 8];
    tmp2 = ((float2*)_workspace[threadIdx.x % 8])[32 + threadIdx.y * 4 + threadIdx.x / 8];

    __syncwarp(0xffffffff);

    if (_is_max) {
        decx::reduce::GPUK::cu_warp_reduce_fp64<double(double, double), 32, 4>(decx::utils::cuda::__float_max2, ((double*)&tmp1), ((double*)&tmp3));
        decx::reduce::GPUK::cu_warp_reduce_fp64<double(double, double), 32, 4>(decx::utils::cuda::__float_max2, ((double*)&tmp2), ((double*)&tmp4));
    }
    else {
        decx::reduce::GPUK::cu_warp_reduce_fp64<double(double, double), 32, 4>(decx::utils::cuda::__float_min2, ((double*)&tmp1), ((double*)&tmp3));
        decx::reduce::GPUK::cu_warp_reduce_fp64<double(double, double), 32, 4>(decx::utils::cuda::__float_min2, ((double*)&tmp2), ((double*)&tmp4));
    }

    __syncthreads();

    if (threadIdx.x % 8 == 0) {
        ((float2*)_workspace[0])[threadIdx.y * 4 + threadIdx.x / 8] = tmp3;
        ((float2*)_workspace[0])[32 + threadIdx.y * 4 + threadIdx.x / 8] = tmp4;
    }

    __syncthreads();
    
    _recv._vf = _workspace[threadIdx.y][threadIdx.x];

    if (tidx < proc_dims_v4.x && threadIdx.y == 0) {
        dst[STG_dex] = _recv._vf;
    }
}

template __global__ void decx::reduce::GPUK::cu_warp_reduce_cmp2D_v_fp32<true>(const float4* __restrict, float4* __restrict, const uint32_t, uint32_t, const uint2);
template __global__ void decx::reduce::GPUK::cu_warp_reduce_cmp2D_v_fp32<false>(const float4* __restrict, float4* __restrict, const uint32_t, uint32_t, const uint2);



template <bool _is_max> __global__ void
decx::reduce::GPUK::cu_warp_reduce_cmp2D_v_fp16(const float4 * __restrict   src, 
                                                float4* __restrict          dst,
                                                const uint32_t              Wsrc_v8, 
                                                uint32_t                    Wdst_v8, 
                                                const uint2                 proc_dims_v8)
{
#if __ABOVE_SM_53
    uint32_t tidx = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t tidy = threadIdx.y + blockDim.y * blockIdx.y;

    uint64_t LDG_dex = Wsrc_v8 * tidy + tidx;
    uint64_t STG_dex = blockIdx.y * Wdst_v8 + tidx;

    __shared__ float4 _workspace[8][32 + 1];

    decx::utils::_cuda_vec128 _recv;
    decx::utils::_cuda_vec128 tmp1, tmp2;
    
    if (threadIdx.y == 0) {
        if (tidx < proc_dims_v8.x) {
            _recv._vf = src[tidx];
        }
        _workspace[0][threadIdx.x] = _recv._vf;
    }

    __syncthreads();
    _recv._vf = _workspace[0][threadIdx.x];
    __syncthreads();

    if (tidx < proc_dims_v8.x && tidy < proc_dims_v8.y) {
        _recv._vf = src[LDG_dex];
    }

    _workspace[threadIdx.y][threadIdx.x] = _recv._vf;

    __syncthreads();

    tmp1._vd.x = ((double*)_workspace[threadIdx.x % 8])[threadIdx.y * 4 + threadIdx.x / 8];
    tmp1._vd.y = ((double*)_workspace[threadIdx.x % 8])[32 + threadIdx.y * 4 + threadIdx.x / 8];

    __syncwarp(0xffffffff);
    constexpr decx::utils::cuda::cu_math_ops<__half2>& _d_op = _is_max ?
                                                               decx::utils::cuda::__half2_max : 
                                                               decx::utils::cuda::__half2_min;

    decx::reduce::GPUK::cu_warp_reduce<__half2, 32, 4>(_d_op, &tmp1._arrh2[0], &tmp2._arrh2[0]);
    decx::reduce::GPUK::cu_warp_reduce<__half2, 32, 4>(_d_op, &tmp1._arrh2[1], &tmp2._arrh2[1]);
    decx::reduce::GPUK::cu_warp_reduce<__half2, 32, 4>(_d_op, &tmp1._arrh2[2], &tmp2._arrh2[2]);
    decx::reduce::GPUK::cu_warp_reduce<__half2, 32, 4>(_d_op, &tmp1._arrh2[3], &tmp2._arrh2[3]);

    __syncthreads();

    if (threadIdx.x % 8 == 0) {
        ((double*)_workspace[threadIdx.x % 8])[threadIdx.y * 4 + threadIdx.x / 8] = tmp2._vd.x;
        ((double*)_workspace[threadIdx.x % 8])[32 + threadIdx.y * 4 + threadIdx.x / 8] = tmp2._vd.y;
    }

    __syncthreads();

    tmp1._vf = _workspace[threadIdx.y][threadIdx.x];

    if (tidx < proc_dims_v8.x && threadIdx.y == 0) {
        dst[STG_dex] = tmp1._vf;
    }
#endif
}

template __global__ void decx::reduce::GPUK::cu_warp_reduce_cmp2D_v_fp16<true>(const float4* __restrict, float4* __restrict, const uint32_t, uint32_t, const uint2);
template __global__ void decx::reduce::GPUK::cu_warp_reduce_cmp2D_v_fp16<false>(const float4* __restrict, float4* __restrict, const uint32_t, uint32_t, const uint2);




template <bool _is_max> __global__ void
decx::reduce::GPUK::cu_warp_reduce_cmp2D_v_u8(const int4 * __restrict     src, 
                                              int4* __restrict            dst,
                                              const uint32_t              Wsrc_v16, 
                                              uint32_t                    Wdst_v16, 
                                              const uint2                 proc_dims_v16)
{
    uint32_t tidx = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t tidy = threadIdx.y + blockDim.y * blockIdx.y;

    uint64_t LDG_dex = Wsrc_v16 * tidy + tidx;
    uint64_t STG_dex = blockIdx.y * Wdst_v16 + tidx;

    __shared__ int4 _workspace[8][32 + 1];

    decx::utils::_cuda_vec128 _recv;
    decx::utils::_cuda_vec64 tmp1, tmp2, tmp3, tmp4;
    
    if (threadIdx.y == 0) {
        if (tidx < proc_dims_v16.x) {
            if (tidx < proc_dims_v16.x) {
                _recv._vi = src[tidx];
            }
        }
        _workspace[0][threadIdx.x] = _recv._vi;
    }

    __syncthreads();
    _recv._vi = _workspace[0][threadIdx.x];
    __syncthreads();

    if (tidx < proc_dims_v16.x && tidy < proc_dims_v16.y) {
        _recv._vi = src[LDG_dex];
    }

    _workspace[threadIdx.y][threadIdx.x] = _recv._vi;

    __syncthreads();

    // checker load 4x4 uint8 values
    tmp1._vui2.x = ((uint32_t*)_workspace[threadIdx.x % 8])[threadIdx.y * 4 + threadIdx.x / 8];
    tmp2._vui2.x = ((uint32_t*)_workspace[threadIdx.x % 8])[32 + threadIdx.y * 4 + threadIdx.x / 8];
    tmp3._vui2.x = ((uint32_t*)_workspace[threadIdx.x % 8])[64 + threadIdx.y * 4 + threadIdx.x / 8];
    tmp4._vui2.x = ((uint32_t*)_workspace[threadIdx.x % 8])[96 + threadIdx.y * 4 + threadIdx.x / 8];

    __syncwarp(0xffffffff);

    constexpr decx::utils::cuda::cu_math_ops<uint32_t>& _d_op = _is_max ? __vmaxu4 : __vminu4;

    // execute warp-level 16-bit x2 reduce procedure on 16 x 8<reduced>, and store the results in tmp[1, 4].y
    decx::reduce::GPUK::cu_warp_reduce<uint32_t, 32, 4>(_d_op, &tmp1._vi2.x, &tmp1._vi2.y);
    decx::reduce::GPUK::cu_warp_reduce<uint32_t, 32, 4>(_d_op, &tmp2._vi2.x, &tmp2._vi2.y);
    decx::reduce::GPUK::cu_warp_reduce<uint32_t, 32, 4>(_d_op, &tmp3._vi2.x, &tmp3._vi2.y);
    decx::reduce::GPUK::cu_warp_reduce<uint32_t, 32, 4>(_d_op, &tmp4._vi2.x, &tmp4._vi2.y);

    __syncthreads();

    if (threadIdx.x % 8 == 0) {
        ((uint32_t*)_workspace[threadIdx.x % 8])[threadIdx.y * 4 + threadIdx.x / 8] = tmp1._vui2.y;
        ((uint32_t*)_workspace[threadIdx.x % 8])[32 + threadIdx.y * 4 + threadIdx.x / 8] = tmp2._vui2.y;
        ((uint32_t*)_workspace[threadIdx.x % 8])[64 + threadIdx.y * 4 + threadIdx.x / 8] = tmp3._vui2.y;
        ((uint32_t*)_workspace[threadIdx.x % 8])[96 + threadIdx.y * 4 + threadIdx.x / 8] = tmp4._vui2.y;
    }

    __syncthreads();

    _recv._vi = _workspace[threadIdx.y][threadIdx.x];

    if (tidx < proc_dims_v16.x && threadIdx.y == 0) {
        dst[STG_dex] = _recv._vi;
    }
}

template __global__ void decx::reduce::GPUK::cu_warp_reduce_cmp2D_v_u8<true>(const int4* __restrict, int4* __restrict, const uint32_t, uint32_t, const uint2);
template __global__ void decx::reduce::GPUK::cu_warp_reduce_cmp2D_v_u8<false>(const int4* __restrict, int4* __restrict, const uint32_t, uint32_t, const uint2);
