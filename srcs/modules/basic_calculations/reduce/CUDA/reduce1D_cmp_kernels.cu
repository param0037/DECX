/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#include "reduce_cmp.cuh"
#include "../../../core/utils/cuda_int32_math_functions.cuh"


template <bool _is_max>
__global__ void
decx::reduce::GPUK::cu_block_reduce_cmp1D_fp32(const float4* __restrict     src, 
                                               float* __restrict            dst, 
                                               const uint64_t               proc_len_v4,
                                               const uint64_t               proc_len_v1,
                                               const float                  _fill_val)
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
    _recv._vf = decx::utils::vec4_set1_fp32(_fill_val);
    float tmp1, tmp2;

    if (LDG_dex < proc_len_v4) 
    {
        _recv._vf = src[LDG_dex];
        if (LDG_dex == proc_len_v4 - 1) {
            for (int i = 4 - (proc_len_v4 * 4 - proc_len_v1); i < 4; ++i) {
                _recv._arrf[i] = _fill_val;
            }
        }
    }

    if (_is_max) {
        tmp1 = decx::reduce::GPUK::float4_reduce_max(_recv._vf);
    }
    else {
        tmp1 = decx::reduce::GPUK::float4_reduce_min(_recv._vf);
    }

    // warp reducing
    if (_is_max) {
        decx::reduce::GPUK::cu_warp_reduce_fp32<float(float, float), 32>(decx::utils::cuda::__fp32_max, &tmp1, &tmp2);
    }
    else {
        decx::reduce::GPUK::cu_warp_reduce_fp32<float(float, float), 32>(decx::utils::cuda::__fp32_min, &tmp1, &tmp2);
    }

    if (warp_lane_id == 0) {
        warp_reduce_results[local_warp_id] = tmp2;
    }

    __syncthreads();

    // let the 0th warp execute the warp-reducing process
    if (local_warp_id == 0) {
        tmp1 = warp_reduce_results[warp_lane_id];
        // synchronize this warp
        __syncwarp(0xffffffff);

        if (_is_max) {
            decx::reduce::GPUK::cu_warp_reduce_fp32<float(float, float), 8>(decx::utils::cuda::__fp32_max, &tmp1, &tmp2);
        }
        else {
            decx::reduce::GPUK::cu_warp_reduce_fp32<float(float, float), 8>(decx::utils::cuda::__fp32_min, &tmp1, &tmp2);
        }

        if (warp_lane_id == 0) {
            dst[blockIdx.x] = tmp2;
        }
    }
}

template __global__ void decx::reduce::GPUK::cu_block_reduce_cmp1D_fp32<true>(const float4* __restrict src, float* __restrict dst,
    const uint64_t proc_len_v4, const uint64_t proc_len_v1, const float _fill_val);
template __global__ void decx::reduce::GPUK::cu_block_reduce_cmp1D_fp32<false>(const float4* __restrict src, float* __restrict dst,
    const uint64_t proc_len_v4, const uint64_t proc_len_v1, const float _fill_val);



template <bool _is_max>
__global__ void
decx::reduce::GPUK::cu_block_reduce_cmp1D_u8(const float4* __restrict     src, 
                                             uint8_t* __restrict            dst, 
                                             const uint64_t               proc_len_v16,
                                             const uint64_t               proc_len_v1,
                                             const uint8_t                _fill_val)
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

    uchar4 _fill_val_v4 = make_uchar4(_fill_val, _fill_val, _fill_val, _fill_val);

    decx::utils::_cuda_vec128 _recv;
    _recv._vi = decx::utils::vec4_set1_int32(*((int32_t*)&_fill_val_v4));
    int32_t tmp1, tmp2;

    if (LDG_dex < proc_len_v16) 
    {
        _recv._vf = src[LDG_dex];
        if (LDG_dex == proc_len_v16 - 1) {
            uint32_t _left_u8 = proc_len_v16 * 16 - proc_len_v1;

            for (int i = 4 - (_left_u8 / 4); i < 4; ++i) {
                _recv._arri[i] = *((int*)&_fill_val_v4);
            }

            int32_t tmp_frame = _recv._arri[3 - (_left_u8 / 4)];
            // [0, 0, 0, 0] [val, val, val, val] -> [4 5 6 7] & [offset]
            _recv._arri[3 - (_left_u8 / 4)] = __byte_perm(*((int*)&_fill_val_v4), tmp_frame, (0xffff >> (4 * (_left_u8 % 4))) & 0x7654);
        }
    }

    if (_is_max) {
        tmp1 = decx::reduce::GPUK::uchar16_max(_recv._vi);
    }
    else {
        tmp1 = decx::reduce::GPUK::uchar16_min(_recv._vi);
    }
    // warp reducing
    if (_is_max) {
        decx::reduce::GPUK::cu_warp_reduce_int32<int32_t(int32_t, int32_t), 32>(decx::utils::cuda::__i32_max, &tmp1, &tmp2);
    }
    else {
        decx::reduce::GPUK::cu_warp_reduce_int32<int32_t(int32_t, int32_t), 32>(decx::utils::cuda::__i32_min, &tmp1, &tmp2);
    }

    if (warp_lane_id == 0) {
        warp_reduce_results[local_warp_id] = tmp2;
    }

    __syncthreads();

    // let the 0th warp execute the warp-reducing process
    if (local_warp_id == 0) {
        tmp1 = warp_reduce_results[warp_lane_id];
        // synchronize this warp
        __syncwarp(0xffffffff);

        if (_is_max) {
            decx::reduce::GPUK::cu_warp_reduce_int32<int32_t(int32_t, int32_t), 8>(decx::utils::cuda::__i32_max, &tmp1, &tmp2);
        }
        else {
            decx::reduce::GPUK::cu_warp_reduce_int32<int32_t(int32_t, int32_t), 8>(decx::utils::cuda::__i32_min, &tmp1, &tmp2);
        }

        if (warp_lane_id == 0) {
            dst[blockIdx.x] = tmp2;
        }
    }
}


template __global__ void decx::reduce::GPUK::cu_block_reduce_cmp1D_u8<true>(const float4* __restrict src, uint8_t* __restrict dst,
    const uint64_t proc_len_v4, const uint64_t proc_len_v1, const uint8_t _fill_val);
template __global__ void decx::reduce::GPUK::cu_block_reduce_cmp1D_u8<false>(const float4* __restrict src, uint8_t* __restrict dst,
    const uint64_t proc_len_v4, const uint64_t proc_len_v1, const uint8_t _fill_val);



template <bool _is_max>
__global__ void
decx::reduce::GPUK::cu_block_reduce_cmp1D_fp16(const float4* __restrict     src, 
                                               __half* __restrict            dst, 
                                               const uint64_t               proc_len_v8,
                                               const uint64_t               proc_len_v1,
                                               const __half                 _fill_val)
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

    __half2 fill_val_vec2 = make_half2(_fill_val, _fill_val);

    decx::utils::_cuda_vec128 _recv;
    _recv._vf = decx::utils::vec4_set1_fp32(*((float*)&fill_val_vec2));
    __half tmp1, tmp2;

    if (LDG_dex < proc_len_v8) 
    {
        _recv._vf = src[LDG_dex];
        if (LDG_dex == proc_len_v8 - 1) {
            for (int i = 8 - (proc_len_v8 * 8 - proc_len_v1); i < 8; ++i) {
                _recv._arrh[i] = _fill_val;
            }
        }
    }

    if (_is_max) {
        tmp1 = decx::reduce::GPUK::half8_max(_recv._arrh2);
    }
    else {
        tmp1 = decx::reduce::GPUK::half8_min(_recv._arrh2);
    }

    // warp reducing
    if (_is_max) {
        decx::reduce::GPUK::cu_warp_reduce_fp16<__half(__half, __half), 32>(decx::utils::cuda::__half_max, &tmp1, &tmp2);
    }
    else {
        decx::reduce::GPUK::cu_warp_reduce_fp16<__half(__half, __half), 32>(decx::utils::cuda::__half_min, &tmp1, &tmp2);
    }

    if (warp_lane_id == 0) {
        warp_reduce_results[local_warp_id] = tmp2;
    }

    __syncthreads();

    // let the 0th warp execute the warp-reducing process
    if (local_warp_id == 0) {
        tmp1 = warp_reduce_results[warp_lane_id];
        // synchronize this warp
        __syncwarp(0xffffffff);

        if (_is_max) {
            decx::reduce::GPUK::cu_warp_reduce_fp16<__half(__half, __half), 8>(decx::utils::cuda::__half_max, &tmp1, &tmp2);
        }
        else {
            decx::reduce::GPUK::cu_warp_reduce_fp16<__half(__half, __half), 8>(decx::utils::cuda::__half_min, &tmp1, &tmp2);
        }

        if (warp_lane_id == 0) {
            dst[blockIdx.x] = tmp2;
        }
    }
}


template __global__ void decx::reduce::GPUK::cu_block_reduce_cmp1D_fp16<true>(const float4* __restrict src, __half* __restrict dst,
    const uint64_t proc_len_v4, const uint64_t proc_len_v1, const __half _fill_val);
template __global__ void decx::reduce::GPUK::cu_block_reduce_cmp1D_fp16<false>(const float4* __restrict src, __half* __restrict dst,
    const uint64_t proc_len_v4, const uint64_t proc_len_v1, const __half _fill_val);
