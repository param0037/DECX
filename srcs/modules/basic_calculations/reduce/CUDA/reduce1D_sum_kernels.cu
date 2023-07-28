/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#include "reduce_sum.cuh"


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

    tmp1 = decx::reduce::GPUK::float4_reduce_sum(_recv._vf);
    // warp reducing
    decx::reduce::GPUK::cu_warp_reduce_fp32<float(float, float), 32>(__fadd_rn, &tmp1, &tmp2);

    if (warp_lane_id == 0) {
        warp_reduce_results[local_warp_id] = tmp2;
    }

    __syncthreads();

    // let the 0th warp execute the warp-reducing process
    if (local_warp_id == 0) {
        tmp1 = warp_reduce_results[warp_lane_id];
        // synchronize this warp
        __syncwarp(0xffffffff);

        decx::reduce::GPUK::cu_warp_reduce_fp32<float(float, float), 8>(__fadd_rn, &tmp1, &tmp2);

        if (warp_lane_id == 0) {
            dst[blockIdx.x] = tmp2;
        }
    }
}



__global__ void
decx::reduce::GPUK::cu_block_reduce_sum1D_fp16_fp32(const float4* __restrict     src, 
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

    tmp1 = decx::reduce::GPUK::half8_reduce_sum(_recv._vf);
    // warp reducing
    decx::reduce::GPUK::cu_warp_reduce_fp32<float(float, float), 32>(__fadd_rn, &tmp1, &tmp2);

    if (warp_lane_id == 0) {
        warp_reduce_results[local_warp_id] = tmp2;
    }

    __syncthreads();

    // let the 0th warp execute the warp-reducing process
    if (local_warp_id == 0) {
        tmp1 = warp_reduce_results[warp_lane_id];
        // synchronize this warp
        __syncwarp(0xffffffff);

        decx::reduce::GPUK::cu_warp_reduce_fp32<float(float, float), 8>(__fadd_rn, &tmp1, &tmp2);

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

    tmp1 = decx::reduce::GPUK::int4_reduce_sum(_recv._vi);
    // warp reducing
    decx::reduce::GPUK::cu_warp_reduce_int32<int32_t(int32_t, int32_t), 32>(decx::utils::cuda::__i32_add, &tmp1, &tmp2);

    if (warp_lane_id == 0) {
        warp_reduce_results[local_warp_id] = tmp2;
    }

    __syncthreads();

    // let the 0th warp execute the warp-reducing process
    if (local_warp_id == 0) {
        tmp1 = warp_reduce_results[warp_lane_id];
        // synchronize this warp
        __syncwarp(0xffffffff);

        decx::reduce::GPUK::cu_warp_reduce_int32<int32_t(int32_t, int32_t), 8>(decx::utils::cuda::__i32_add, &tmp1, &tmp2);

        if (warp_lane_id == 0) {
            dst[blockIdx.x] = tmp2;
        }
    }
}



__global__ void
decx::reduce::GPUK::cu_block_reduce_sum1D_u8_i32(const int4* __restrict     src, 
                                                 int* __restrict            dst, 
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

    tmp1 = decx::reduce::GPUK::uchar16_reduce_sum(_recv._vi);
    decx::reduce::GPUK::cu_warp_reduce_int32<int32_t(int32_t, int32_t), 32>(decx::utils::cuda::__i32_add, &tmp1, &tmp2);

    if (warp_lane_id == 0) {
        warp_reduce_results[local_warp_id] = tmp2;
    }

    __syncthreads();

    // let the 0th warp execute the warp-reducing process
    if (local_warp_id == 0) {
        tmp1 = warp_reduce_results[warp_lane_id];
        // synchronize this warp
        __syncwarp(0xffffffff);

        decx::reduce::GPUK::cu_warp_reduce_int32<int32_t(int32_t, int32_t), 8>(decx::utils::cuda::__i32_add, &tmp1, &tmp2);

        if (warp_lane_id == 0) {
            dst[blockIdx.x] = tmp2;
        }
    }
}