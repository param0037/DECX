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
#include "../../../core/utils/decx_cuda_math_functions.cuh"


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
        tmp1 = decx::reduce::GPUK::float4_max(_recv._vf);
    }
    else {
        tmp1 = decx::reduce::GPUK::float4_min(_recv._vf);
    }

    // warp reducing
    if (_is_max) {
        decx::reduce::GPUK::cu_warp_reduce<float, 32>(decx::utils::cuda::__fp32_max, &tmp1, &tmp2);
    }
    else {
        decx::reduce::GPUK::cu_warp_reduce<float, 32>(decx::utils::cuda::__fp32_min, &tmp1, &tmp2);
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
            //decx::reduce::GPUK::cu_warp_reduce_fp32<float(float, float), 8>(decx::utils::cuda::__fp32_max, &tmp1, &tmp2);
            decx::reduce::GPUK::cu_warp_reduce<float, 8>(decx::utils::cuda::__fp32_max, &tmp1, &tmp2);
        }
        else {
            //decx::reduce::GPUK::cu_warp_reduce_fp32<float(float, float), 8>(decx::utils::cuda::__fp32_min, &tmp1, &tmp2);
            decx::reduce::GPUK::cu_warp_reduce<float, 8>(decx::utils::cuda::__fp32_min, &tmp1, &tmp2);
        }

        if (warp_lane_id == 0) {
            dst[blockIdx.x] = tmp2;
        }
    }
}

template __global__ void decx::reduce::GPUK::cu_block_reduce_cmp1D_fp32<true>(const float4* __restrict, float* __restrict, const uint64_t, const uint64_t, const float);
template __global__ void decx::reduce::GPUK::cu_block_reduce_cmp1D_fp32<false>(const float4* __restrict, float* __restrict, const uint64_t, const uint64_t, const float);





template <bool _is_max>
__global__ void
decx::reduce::GPUK::cu_block_reduce_cmp1D_int32(const int4* __restrict     src, 
                                               int32_t* __restrict            dst, 
                                               const uint64_t               proc_len_v4,
                                               const uint64_t               proc_len_v1,
                                               const int32_t                  _fill_val)
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
    _recv._vi = decx::utils::vec4_set1_int32(_fill_val);
    int32_t tmp1, tmp2;

    if (LDG_dex < proc_len_v4) 
    {
        _recv._vi = src[LDG_dex];
        if (LDG_dex == proc_len_v4 - 1) {
            for (uint8_t i = 4 - (proc_len_v4 * 4 - proc_len_v1); i < 4; ++i) {
                _recv._arri[i] = _fill_val;
            }
        }
    }

    if (_is_max) {
        tmp1 = decx::reduce::GPUK::int4_max(_recv._vi);
    }
    else {
        tmp1 = decx::reduce::GPUK::int4_min(_recv._vi);
    }

    // warp reducing
    if (_is_max) {
        decx::reduce::GPUK::cu_warp_reduce<int32_t, 32>(decx::utils::cuda::__i32_max, &tmp1, &tmp2);
    }
    else {
        decx::reduce::GPUK::cu_warp_reduce<int32_t, 32>(decx::utils::cuda::__i32_min, &tmp1, &tmp2);
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
            decx::reduce::GPUK::cu_warp_reduce<int32_t, 8>(decx::utils::cuda::__i32_max, &tmp1, &tmp2);
        }
        else {
            decx::reduce::GPUK::cu_warp_reduce<int32_t, 8>(decx::utils::cuda::__i32_min, &tmp1, &tmp2);
        }

        if (warp_lane_id == 0) {
            dst[blockIdx.x] = tmp2;
        }
    }
}

template __global__ void decx::reduce::GPUK::cu_block_reduce_cmp1D_int32<true>(const int4* __restrict, int32_t* __restrict, const uint64_t, const uint64_t, const int32_t);
template __global__ void decx::reduce::GPUK::cu_block_reduce_cmp1D_int32<false>(const int4* __restrict, int32_t* __restrict, const uint64_t, const uint64_t, const int32_t);




template <bool _is_max>
__global__ void
decx::reduce::GPUK::cu_block_reduce_cmp1D_fp64(const double2* __restrict     src, 
                                               double* __restrict            dst, 
                                               const uint64_t               proc_len_v2,
                                               const uint64_t               proc_len_v1,
                                               const double                  _fill_val)
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
    _recv._vd = decx::utils::vec2_set1_fp64(_fill_val);
    double tmp1, tmp2;

    if (LDG_dex < proc_len_v2) 
    {
        _recv._vd = src[LDG_dex];
        if (LDG_dex == proc_len_v2 - 1 && proc_len_v1 % 2 == 1) {
            _recv._arrd[1] = _fill_val;
        }
    }

    if (_is_max) {
        tmp1 = decx::utils::cuda::__fp64_max(_recv._vd.x, _recv._vd.y);
    }
    else {
        tmp1 = decx::utils::cuda::__fp64_min(_recv._vd.x, _recv._vd.y);
    }

    // warp reducing
    if (_is_max) {
        //decx::reduce::GPUK::cu_warp_reduce_fp64<double(double, double), 32>(decx::utils::cuda::__fp64_max, &tmp1, &tmp2);
        decx::reduce::GPUK::cu_warp_reduce<double, 32>(decx::utils::cuda::__fp64_max, &tmp1, &tmp2);
    }
    else {
        //decx::reduce::GPUK::cu_warp_reduce_fp64<double(double, double), 32>(decx::utils::cuda::__fp64_min, &tmp1, &tmp2);
        decx::reduce::GPUK::cu_warp_reduce<double, 32>(decx::utils::cuda::__fp64_min, &tmp1, &tmp2);
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
            //decx::reduce::GPUK::cu_warp_reduce_fp64<double(double, double), 8>(decx::utils::cuda::__fp64_max, &tmp1, &tmp2);
            decx::reduce::GPUK::cu_warp_reduce<double, 8>(decx::utils::cuda::__fp64_max, &tmp1, &tmp2);
        }
        else {
            //decx::reduce::GPUK::cu_warp_reduce_fp64<double(double, double), 8>(decx::utils::cuda::__fp64_min, &tmp1, &tmp2);
            decx::reduce::GPUK::cu_warp_reduce<double, 8>(decx::utils::cuda::__fp64_min, &tmp1, &tmp2);
        }

        if (warp_lane_id == 0) {
            dst[blockIdx.x] = tmp2;
        }
    }
}

template __global__ void decx::reduce::GPUK::cu_block_reduce_cmp1D_fp64<true>(const double2* __restrict, double* __restrict, const uint64_t, const uint64_t, const double);
template __global__ void decx::reduce::GPUK::cu_block_reduce_cmp1D_fp64<false>(const double2* __restrict, double* __restrict, const uint64_t, const uint64_t, const double);



template <bool _is_max>
__global__ void
decx::reduce::GPUK::cu_block_reduce_cmp1D_u8(const int4* __restrict       src, 
                                             uint8_t* __restrict          dst, 
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
        _recv._vi = src[LDG_dex];
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
        //decx::reduce::GPUK::cu_warp_reduce_int32<int32_t(int32_t, int32_t), 32>(decx::utils::cuda::__i32_max, &tmp1, &tmp2);
        decx::reduce::GPUK::cu_warp_reduce<int32_t, 32>(decx::utils::cuda::__i32_max, &tmp1, &tmp2);
    }
    else {
        //decx::reduce::GPUK::cu_warp_reduce_int32<int32_t(int32_t, int32_t), 32>(decx::utils::cuda::__i32_min, &tmp1, &tmp2);
        decx::reduce::GPUK::cu_warp_reduce<int32_t, 32>(decx::utils::cuda::__i32_min, &tmp1, &tmp2);
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
            //decx::reduce::GPUK::cu_warp_reduce_int32<int32_t(int32_t, int32_t), 8>(decx::utils::cuda::__i32_max, &tmp1, &tmp2);
            decx::reduce::GPUK::cu_warp_reduce<int32_t, 8>(decx::utils::cuda::__i32_max, &tmp1, &tmp2);
        }
        else {
            //decx::reduce::GPUK::cu_warp_reduce_int32<int32_t(int32_t, int32_t), 8>(decx::utils::cuda::__i32_min, &tmp1, &tmp2);
            decx::reduce::GPUK::cu_warp_reduce<int32_t, 8>(decx::utils::cuda::__i32_min, &tmp1, &tmp2);
        }

        if (warp_lane_id == 0) {
            dst[blockIdx.x] = tmp2;
        }
    }
}


template __global__ void decx::reduce::GPUK::cu_block_reduce_cmp1D_u8<true>(const int4* __restrict src, uint8_t* __restrict dst,
    const uint64_t proc_len_v4, const uint64_t proc_len_v1, const uint8_t _fill_val);
template __global__ void decx::reduce::GPUK::cu_block_reduce_cmp1D_u8<false>(const int4* __restrict src, uint8_t* __restrict dst,
    const uint64_t proc_len_v4, const uint64_t proc_len_v1, const uint8_t _fill_val);



template <bool _is_max>
__global__ void
decx::reduce::GPUK::cu_block_reduce_cmp1D_fp16(const float4* __restrict     src, 
                                               __half* __restrict            dst, 
                                               const uint64_t               proc_len_v8,
                                               const uint64_t               proc_len_v1,
                                               const __half                 _fill_val)
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
        //decx::reduce::GPUK::cu_warp_reduce_fp16<__half(__half, __half), 32>(decx::utils::cuda::__half_max, &tmp1, &tmp2);
        decx::reduce::GPUK::cu_warp_reduce<__half, 32>(decx::utils::cuda::__half_max, &tmp1, &tmp2);
    }
    else {
        //decx::reduce::GPUK::cu_warp_reduce_fp16<__half(__half, __half), 32>(decx::utils::cuda::__half_min, &tmp1, &tmp2);
        decx::reduce::GPUK::cu_warp_reduce<__half, 32>(decx::utils::cuda::__half_min, &tmp1, &tmp2);
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
            //decx::reduce::GPUK::cu_warp_reduce_fp16<__half(__half, __half), 8>(decx::utils::cuda::__half_max, &tmp1, &tmp2);
            decx::reduce::GPUK::cu_warp_reduce<__half, 8>(decx::utils::cuda::__half_max, &tmp1, &tmp2);
        }
        else {
            //decx::reduce::GPUK::cu_warp_reduce_fp16<__half(__half, __half), 8>(decx::utils::cuda::__half_min, &tmp1, &tmp2);
            decx::reduce::GPUK::cu_warp_reduce<__half, 8>(decx::utils::cuda::__half_min, &tmp1, &tmp2);
        }

        if (warp_lane_id == 0) {
            dst[blockIdx.x] = tmp2;
        }
    }
#endif
}


template __global__ void decx::reduce::GPUK::cu_block_reduce_cmp1D_fp16<true>(const float4* __restrict src, __half* __restrict dst,
    const uint64_t proc_len_v4, const uint64_t proc_len_v1, const __half _fill_val);
template __global__ void decx::reduce::GPUK::cu_block_reduce_cmp1D_fp16<false>(const float4* __restrict src, __half* __restrict dst,
    const uint64_t proc_len_v4, const uint64_t proc_len_v1, const __half _fill_val);



template <bool _is_max> __global__ void 
decx::reduce::GPUK::cu_warp_reduce_cmp2D_flatten_fp32(const float4 * __restrict   src, 
                                                 float* __restrict           dst,
                                                 const uint32_t              Wsrc_v4, 
                                                 const uint2                 proc_dims,
                                                 const float                 _fill_val)
{
    uint32_t tidx = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t tidy = threadIdx.y + blockDim.y * blockIdx.y;

    uint64_t LDG_dex = Wsrc_v4 * tidy + tidx;
    uint64_t STG_dex = blockIdx.y * gridDim.x + blockIdx.x;

    uint32_t proc_W_v4 = decx::utils::ceil<uint32_t>(proc_dims.x, 4);

    /**
    * Shared memory for the reduced results of 8 warps in a block
    * Extended to 32 for warp-level loading
    * No need to set the remaining values to zero since for the warp reducing
    * all the redundancy will not invovled with correct parameters set
    */
    __shared__ float _sh_warp_reduce_res[32];

    decx::utils::_cuda_vec128 _recv;
    _recv._vf = decx::utils::vec4_set1_fp32(_fill_val);

    float _thread_sum = 0, _warp_reduce_res = 0;

    if (tidx < proc_W_v4 && tidy < proc_dims.y) {
        _recv._vf = src[LDG_dex];
        if (tidx == proc_W_v4 - 1) {
            for (int i = 4 - (proc_W_v4 * 4 - proc_dims.x); i < 4; ++i) {
                _recv._arrf[i] = _fill_val;
            }
        }
    }

    if (_is_max) {
        _thread_sum = decx::reduce::GPUK::float4_max(_recv._vf);
        //decx::reduce::GPUK::cu_warp_reduce_fp32<float(float, float), 32>(decx::utils::cuda::__fp32_max, &_thread_sum, &_warp_reduce_res);
        decx::reduce::GPUK::cu_warp_reduce<float, 32>(decx::utils::cuda::__fp32_max, &_thread_sum, &_warp_reduce_res);
    }
    else {
        _thread_sum = decx::reduce::GPUK::float4_min(_recv._vf);
        //decx::reduce::GPUK::cu_warp_reduce_fp32<float(float, float), 32>(decx::utils::cuda::__fp32_min, &_thread_sum, &_warp_reduce_res);
        decx::reduce::GPUK::cu_warp_reduce<float, 32>(decx::utils::cuda::__fp32_min, &_thread_sum, &_warp_reduce_res);
    }

    if (threadIdx.x == 0) {
        _sh_warp_reduce_res[threadIdx.y] = _warp_reduce_res;
    }

    __syncthreads();

    if (threadIdx.y == 0) {
        _thread_sum = _sh_warp_reduce_res[threadIdx.x];
        __syncwarp(0xffffffff);

        if (_is_max) {
            //decx::reduce::GPUK::cu_warp_reduce_fp32<float(float, float), 8>(decx::utils::cuda::__fp32_max, &_thread_sum, &_warp_reduce_res);
            decx::reduce::GPUK::cu_warp_reduce<float, 8>(decx::utils::cuda::__fp32_max, &_thread_sum, &_warp_reduce_res);
        }
        else {
            //decx::reduce::GPUK::cu_warp_reduce_fp32<float(float, float), 8>(decx::utils::cuda::__fp32_min, &_thread_sum, &_warp_reduce_res);
            decx::reduce::GPUK::cu_warp_reduce<float, 8>(decx::utils::cuda::__fp32_min, &_thread_sum, &_warp_reduce_res);
        }

        if (threadIdx.x == 0) {
            dst[STG_dex] = _warp_reduce_res;
        }
    }
}


template __global__ void decx::reduce::GPUK::cu_warp_reduce_cmp2D_flatten_fp32<true>(const float4* __restrict, float* __restrict, const uint32_t, const uint2, const float); 
template __global__ void decx::reduce::GPUK::cu_warp_reduce_cmp2D_flatten_fp32<false>(const float4* __restrict, float* __restrict, const uint32_t, const uint2, const float);




template <bool _is_max> __global__ void 
decx::reduce::GPUK::cu_warp_reduce_cmp2D_flatten_int32(const int4 * __restrict      src, 
                                                       int32_t* __restrict          dst,
                                                       const uint32_t               Wsrc_v4, 
                                                       const uint2                  proc_dims,
                                                       const int32_t                _fill_val)
{
    uint32_t tidx = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t tidy = threadIdx.y + blockDim.y * blockIdx.y;

    uint64_t LDG_dex = Wsrc_v4 * tidy + tidx;
    uint64_t STG_dex = blockIdx.y * gridDim.x + blockIdx.x;

    uint32_t proc_W_v4 = decx::utils::ceil<uint32_t>(proc_dims.x, 4);

    /**
    * Shared memory for the reduced results of 8 warps in a block
    * Extended to 32 for warp-level loading
    * No need to set the remaining values to zero since for the warp reducing
    * all the redundancy will not invovled with correct parameters set
    */
    __shared__ int32_t _sh_warp_reduce_res[32];

    decx::utils::_cuda_vec128 _recv;
    _recv._vi = decx::utils::vec4_set1_int32(_fill_val);

    int32_t _thread_sum = 0, _warp_reduce_res = 0;

    if (tidx < proc_W_v4 && tidy < proc_dims.y) {
        _recv._vi = src[LDG_dex];
        if (tidx == proc_W_v4 - 1) {
            for (int i = 4 - (proc_W_v4 * 4 - proc_dims.x); i < 4; ++i) {
                _recv._arri[i] = _fill_val;
            }
        }
    }

    if (_is_max) {
        _thread_sum = decx::reduce::GPUK::int4_max(_recv._vi);
        decx::reduce::GPUK::cu_warp_reduce<int32_t, 32>(decx::utils::cuda::__i32_max, &_thread_sum, &_warp_reduce_res);
    }
    else {
        _thread_sum = decx::reduce::GPUK::int4_min(_recv._vi);
        decx::reduce::GPUK::cu_warp_reduce<int32_t, 32>(decx::utils::cuda::__i32_min, &_thread_sum, &_warp_reduce_res);
    }

    if (threadIdx.x == 0) {
        _sh_warp_reduce_res[threadIdx.y] = _warp_reduce_res;
    }

    __syncthreads();

    if (threadIdx.y == 0) {
        _thread_sum = _sh_warp_reduce_res[threadIdx.x];
        __syncwarp(0xffffffff);

        if (_is_max) {
            decx::reduce::GPUK::cu_warp_reduce<int32_t, 8>(decx::utils::cuda::__i32_max, &_thread_sum, &_warp_reduce_res);
        }
        else {
            decx::reduce::GPUK::cu_warp_reduce<int32_t, 8>(decx::utils::cuda::__i32_min, &_thread_sum, &_warp_reduce_res);
        }

        if (threadIdx.x == 0) {
            dst[STG_dex] = _warp_reduce_res;
        }
    }
}


template __global__ void decx::reduce::GPUK::cu_warp_reduce_cmp2D_flatten_int32<true>(const int4* __restrict, int32_t* __restrict, const uint32_t, const uint2, const int32_t); 
template __global__ void decx::reduce::GPUK::cu_warp_reduce_cmp2D_flatten_int32<false>(const int4* __restrict, int32_t* __restrict, const uint32_t, const uint2, const int32_t);





template <bool _is_max> __global__ void 
decx::reduce::GPUK::cu_warp_reduce_cmp2D_flatten_fp64(const double2 * __restrict   src, 
                                                      double* __restrict           dst,
                                                      const uint32_t              Wsrc_v2, 
                                                      const uint2                 proc_dims,
                                                      const double                 _fill_val)
{
    uint32_t tidx = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t tidy = threadIdx.y + blockDim.y * blockIdx.y;

    uint64_t LDG_dex = Wsrc_v2 * tidy + tidx;
    uint64_t STG_dex = blockIdx.y * gridDim.x + blockIdx.x;

    uint32_t proc_W_v2 = decx::utils::ceil<uint32_t>(proc_dims.x, 2);

    /**
    * Shared memory for the reduced results of 8 warps in a block
    * Extended to 32 for warp-level loading
    * No need to set the remaining values to zero since for the warp reducing
    * all the redundancy will not invovled with correct parameters set
    */
    __shared__ double _sh_warp_reduce_res[32];

    decx::utils::_cuda_vec128 _recv;
    _recv._vd = decx::utils::vec2_set1_fp64(_fill_val);

    double _thread_sum = 0, _warp_reduce_res = 0;

    if (tidx < proc_W_v2 && tidy < proc_dims.y) {
        _recv._vd = src[LDG_dex];
        if (tidx == proc_W_v2 - 1) {
            for (int i = 2 - (proc_W_v2 * 2 - proc_dims.x); i < 2; ++i) {
                _recv._arrd[i] = _fill_val;
            }
        }
    }

    if (_is_max) {
        _thread_sum = max(_recv._vd.x, _recv._vd.y);
        decx::reduce::GPUK::cu_warp_reduce<double, 32>(decx::utils::cuda::__fp64_max, &_thread_sum, &_warp_reduce_res);
    }
    else {
        _thread_sum = min(_recv._vd.x, _recv._vd.y);
        decx::reduce::GPUK::cu_warp_reduce<double, 32>(decx::utils::cuda::__fp64_min, &_thread_sum, &_warp_reduce_res);
    }

    if (threadIdx.x == 0) {
        _sh_warp_reduce_res[threadIdx.y] = _warp_reduce_res;
    }

    __syncthreads();

    if (threadIdx.y == 0) {
        _thread_sum = _sh_warp_reduce_res[threadIdx.x];
        __syncwarp(0xffffffff);

        if (_is_max) {
            decx::reduce::GPUK::cu_warp_reduce<double, 8>(decx::utils::cuda::__fp64_max, &_thread_sum, &_warp_reduce_res);
        }
        else {
            decx::reduce::GPUK::cu_warp_reduce<double, 8>(decx::utils::cuda::__fp64_min, &_thread_sum, &_warp_reduce_res);
        }

        if (threadIdx.x == 0) {
            dst[STG_dex] = _warp_reduce_res;
        }
    }
}


template __global__ void decx::reduce::GPUK::cu_warp_reduce_cmp2D_flatten_fp64<true>(const double2* __restrict, double* __restrict, const uint32_t, const uint2, const double); 
template __global__ void decx::reduce::GPUK::cu_warp_reduce_cmp2D_flatten_fp64<false>(const double2* __restrict, double* __restrict, const uint32_t, const uint2, const double);





template <bool _is_max> __global__ void 
decx::reduce::GPUK::cu_warp_reduce_cmp2D_flatten_fp16(const float4 * __restrict   src, 
                                                      __half* __restrict          dst,
                                                      const uint32_t              Wsrc_v8, 
                                                      const uint2                 proc_dims,
                                                      const __half                _fill_val)
{
#if __ABOVE_SM_53
    uint32_t tidx = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t tidy = threadIdx.y + blockDim.y * blockIdx.y;

    uint64_t LDG_dex = Wsrc_v8 * tidy + tidx;
    uint64_t STG_dex = blockIdx.y * gridDim.x + blockIdx.x;

    uint32_t proc_W_v8 = decx::utils::ceil<uint32_t>(proc_dims.x, 8);

    /**
    * Shared memory for the reduced results of 8 warps in a block
    * Extended to 32 for warp-level loading
    * No need to set the remaining values to zero since for the warp reducing
    * all the redundancy will not invovled with correct parameters set
    */
    __shared__ __half _sh_warp_reduce_res[32];

    decx::utils::_cuda_vec128 _recv;
    _recv._vf = decx::utils::vec8_set1_fp16(_fill_val);

    __half _thread_sum, _warp_reduce_res;

    if (tidx < proc_W_v8 && tidy < proc_dims.y) {
        _recv._vf = src[LDG_dex];
        if (tidx == proc_W_v8 - 1) {
            for (int i = 8 - (proc_W_v8 * 8 - proc_dims.x); i < 8; ++i) {
                _recv._arrh[i] = _fill_val;
            }
        }
    }

    if (_is_max) {
        _thread_sum = decx::reduce::GPUK::half8_max(_recv._arrh2);
        decx::reduce::GPUK::cu_warp_reduce<__half, 32>(decx::utils::cuda::__half_max, &_thread_sum, &_warp_reduce_res);
    }
    else {
        _thread_sum = decx::reduce::GPUK::half8_min(_recv._arrh2);
        decx::reduce::GPUK::cu_warp_reduce<__half, 32>(decx::utils::cuda::__half_min, &_thread_sum, &_warp_reduce_res);
    }

    if (threadIdx.x == 0) {
        _sh_warp_reduce_res[threadIdx.y] = _warp_reduce_res;
    }

    __syncthreads();

    if (threadIdx.y == 0) {
        _thread_sum = _sh_warp_reduce_res[threadIdx.x];
        __syncwarp(0xffffffff);

        if (_is_max) {
            decx::reduce::GPUK::cu_warp_reduce<__half, 8>(decx::utils::cuda::__half_max, &_thread_sum, &_warp_reduce_res);
        }
        else {
            decx::reduce::GPUK::cu_warp_reduce<__half, 8>(decx::utils::cuda::__half_min, &_thread_sum, &_warp_reduce_res);
        }

        if (threadIdx.x == 0) {
            dst[STG_dex] = _warp_reduce_res;
        }
    }
#endif
}


template __global__ void decx::reduce::GPUK::cu_warp_reduce_cmp2D_flatten_fp16<true>(const float4* __restrict, __half* __restrict, const uint32_t, const uint2, const __half); 
template __global__ void decx::reduce::GPUK::cu_warp_reduce_cmp2D_flatten_fp16<false>(const float4* __restrict, __half* __restrict, const uint32_t, const uint2, const __half);




template <bool _is_max> __global__ void 
decx::reduce::GPUK::cu_warp_reduce_cmp2D_flatten_u8(const int4 * __restrict     src, 
                                                    uint8_t* __restrict         dst,
                                                    const uint32_t              Wsrc_v16, 
                                                    const uint2                 proc_dims,
                                                    const uint8_t               _fill_val)
{
    uint32_t tidx = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t tidy = threadIdx.y + blockDim.y * blockIdx.y;

    uint64_t LDG_dex = Wsrc_v16 * tidy + tidx;
    uint64_t STG_dex = blockIdx.y * gridDim.x + blockIdx.x;

    uint32_t proc_W_v16 = decx::utils::ceil<uint32_t>(proc_dims.x, 16);

    /**
    * Shared memory for the reduced results of 8 warps in a block
    * Extended to 32 for warp-level loading
    * No need to set the remaining values to zero since for the warp reducing
    * all the redundancy will not invovled with correct parameters set
    */
    __shared__ int32_t _sh_warp_reduce_res[32];

    decx::utils::_cuda_vec128 _recv;
    _recv._vi = decx::utils::vec16_set1_u8(_fill_val);

    int32_t _thread_sum, _warp_reduce_res;

    if (tidx < proc_W_v16 && tidy < proc_dims.y) {
        _recv._vi = src[LDG_dex];

        if (tidx == proc_W_v16 - 1) {
            uint32_t _left_u8 = proc_W_v16 * 16 - proc_dims.x;
            uchar4 _fill_vec4 = make_uchar4(_fill_val, _fill_val, _fill_val, _fill_val);

            for (int i = 4 - (_left_u8 / 4); i < 4; ++i) {
                _recv._arri[i] = *((int32_t*)&_fill_vec4);
            }
            int32_t tmp_frame = _recv._arri[3 - (_left_u8 / 4)];
            // [0, 0, 0, 0] [val, val, val, val] -> [4 5 6 7] & [offset]
            _recv._arri[3 - (_left_u8 / 4)] = __byte_perm(*((int32_t*)&_fill_vec4), tmp_frame, (0xffff >> (4 * (_left_u8 % 4))) & 0x7654);
        }
    }

    if (_is_max) {
        _thread_sum = decx::reduce::GPUK::uchar16_max(_recv._vi);
        decx::reduce::GPUK::cu_warp_reduce<int32_t, 32>(decx::utils::cuda::__i32_max, &_thread_sum, &_warp_reduce_res);
    }
    else {
        _thread_sum = decx::reduce::GPUK::uchar16_min(_recv._vi);
        decx::reduce::GPUK::cu_warp_reduce<int32_t, 32>(decx::utils::cuda::__i32_min, &_thread_sum, &_warp_reduce_res);
    }

    if (threadIdx.x == 0) {
        _sh_warp_reduce_res[threadIdx.y] = _warp_reduce_res;
    }

    __syncthreads();

    if (threadIdx.y == 0) {
        _thread_sum = _sh_warp_reduce_res[threadIdx.x];
        __syncwarp(0xffffffff);

        if (_is_max) {
            decx::reduce::GPUK::cu_warp_reduce<int32_t, 8>(decx::utils::cuda::__i32_max, &_thread_sum, &_warp_reduce_res);
        }
        else {
            decx::reduce::GPUK::cu_warp_reduce<int32_t, 8>(decx::utils::cuda::__i32_min, &_thread_sum, &_warp_reduce_res);
        }

        if (threadIdx.x == 0) {
            dst[STG_dex] = _warp_reduce_res;
        }
    }
}


template __global__ void decx::reduce::GPUK::cu_warp_reduce_cmp2D_flatten_u8<true>(const int4* __restrict, uint8_t* __restrict, const uint32_t, const uint2, const uint8_t); 
template __global__ void decx::reduce::GPUK::cu_warp_reduce_cmp2D_flatten_u8<false>(const int4* __restrict, uint8_t* __restrict, const uint32_t, const uint2, const uint8_t);