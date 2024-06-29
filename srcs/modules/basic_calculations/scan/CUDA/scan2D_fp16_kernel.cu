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

#include "scan.cuh"


__global__ void 
decx::scan::GPUK::cu_h_block_inclusive_scan_fp16_2D(const float4* __restrict     src, 
                                                   float4* __restrict           warp_status, 
                                                   float4* __restrict           dst,
                                                   const uint32_t               Wsrc_v8,
                                                   const uint32_t               Wdst_v4,
                                                   const uint32_t               Wstatus,
                                                   const uint2                  proc_dim_v4)
{
#if __ABOVE_SM_53
    // in-block (local) warp_id = threadIdx.y
    uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;      // H
    uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;      // V
    // in-warp (local) lane_id = threadIdx.x % 32
    uint32_t lane_id = (threadIdx.x & 0x1f);        // local thread index within a warp

    uint64_t LDG_dex = (uint64_t)tidx + (uint64_t)tidy * (uint64_t)Wsrc_v8;
    uint64_t STG_dex = (uint64_t)tidx * 2 + (uint64_t)tidy * (uint64_t)Wdst_v4;

    decx::scan::scan_warp_pred_fp32 _status;

    decx::utils::_cuda_vec128 _recv;
    float tmp1_fp32, tmp2_fp32;
    half2& tmp1_h2 = *((half2*)&tmp1_fp32);
    half2& tmp2_h2 = *((half2*)&tmp2_fp32);

    float4 block_scan_res[2];

    if (tidx < proc_dim_v4.x / 2 && tidy < proc_dim_v4.y) {
        _recv._vf = src[LDG_dex];
    }

    uint64_t STG_status_dex = (uint64_t)blockIdx.x + (uint64_t)tidy * (uint64_t)Wstatus;

    decx::scan::GPUK::_inclusive_scan_half8(_recv, &_recv);

    decx::scan::GPUK::cu_warp_exclusive_scan_fp16<__half(__half, __half), 32>(__hadd, &_recv._arrh[7], &tmp1_h2.x, lane_id);
    _recv._vf = decx::utils::add_scalar_vec4(_recv, tmp1_h2.x);

    block_scan_res[0].x = __half2float(_recv._arrh[0]);
    block_scan_res[0].y = __half2float(_recv._arrh[1]);
    block_scan_res[0].z = __half2float(_recv._arrh[2]);
    block_scan_res[0].w = __half2float(_recv._arrh[3]);
    block_scan_res[1].x = __half2float(_recv._arrh[4]);
    block_scan_res[1].y = __half2float(_recv._arrh[5]);
    block_scan_res[1].z = __half2float(_recv._arrh[6]);
    block_scan_res[1].w = __half2float(_recv._arrh[7]);

    if (lane_id == warpSize - 1) {
        if (blockIdx.x == 0) {
            _status._warp_status = decx::scan::_scan_warp_status::PREFIX_AVAILABLE;
            _status._prefix_sum = block_scan_res[1].w;
            warp_status[STG_status_dex] = *((float4*)&_status);
        }
        else {
            _status._warp_status = decx::scan::_scan_warp_status::AGGREGATE_AVAILABLE;
            _status._warp_aggregate = block_scan_res[1].w;
            warp_status[STG_status_dex] = *((float4*)&_status);
        }
    }

    if (tidy < proc_dim_v4.y) {
        if (tidx * 2 < proc_dim_v4.x) { dst[STG_dex] = block_scan_res[0]; }
        if (tidx * 2 + 1 < proc_dim_v4.x) { dst[STG_dex + 1] = block_scan_res[1]; }
    }
#endif
}



__global__ void 
decx::scan::GPUK::cu_h_block_exclusive_scan_fp16_2D(const float4* __restrict     src, 
                                                   float4* __restrict           warp_status, 
                                                   float4* __restrict           dst,
                                                   const uint32_t               Wsrc_v8,
                                                   const uint32_t               Wdst_v4,
                                                   const uint32_t               Wstatus,
                                                   const uint2                  proc_dim_v4)
{
#if __ABOVE_SM_53
    // in-block (local) warp_id = threadIdx.y
    uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;      // H
    uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;      // V
    // in-warp (local) lane_id = threadIdx.x % 32
    uint32_t lane_id = (threadIdx.x & 0x1f);        // local thread index within a warp

    uint64_t LDG_dex = (uint64_t)tidx + (uint64_t)tidy * (uint64_t)Wsrc_v8;
    uint64_t STG_dex = (uint64_t)tidx * 2 + (uint64_t)tidy * (uint64_t)Wdst_v4;

    decx::scan::scan_warp_pred_fp32 _status;

    decx::utils::_cuda_vec128 _recv;
    float tmp1_fp32, tmp2_fp32;
    half2& tmp1_h2 = *((half2*)&tmp1_fp32);
    half2& tmp2_h2 = *((half2*)&tmp2_fp32);
    __half _end_value;

    float4 block_scan_res[2];

    if (tidx < proc_dim_v4.x / 2 && tidy < proc_dim_v4.y) {
        _recv._vf = src[LDG_dex];
        _end_value = _recv._arrh[7];
    }

    uint64_t STG_status_dex = (uint64_t)blockIdx.x + (uint64_t)tidy * (uint64_t)Wstatus;

    decx::scan::GPUK::_exclusive_scan_half8(_recv, &_recv);
    tmp2_h2.x = __hadd(_recv._arrh[7], _end_value);

    decx::scan::GPUK::cu_warp_exclusive_scan_fp16<__half(__half, __half), 32>(__hadd, &tmp2_h2.x, &tmp1_h2.x, lane_id);
    _recv._vf = decx::utils::add_scalar_vec4(_recv, tmp1_h2.x);

    block_scan_res[0].x = __half2float(_recv._arrh[0]);
    block_scan_res[0].y = __half2float(_recv._arrh[1]);
    block_scan_res[0].z = __half2float(_recv._arrh[2]);
    block_scan_res[0].w = __half2float(_recv._arrh[3]);
    block_scan_res[1].x = __half2float(_recv._arrh[4]);
    block_scan_res[1].y = __half2float(_recv._arrh[5]);
    block_scan_res[1].z = __half2float(_recv._arrh[6]);
    block_scan_res[1].w = __half2float(_recv._arrh[7]);

    if (lane_id == warpSize - 1) {
        if (blockIdx.x == 0) {
            _status._warp_status = decx::scan::_scan_warp_status::PREFIX_AVAILABLE;
            _status._prefix_sum = __fadd_rn(block_scan_res[1].w, __half2float(_end_value));
            _status._end_value = __half2float(_end_value);
            warp_status[STG_status_dex] = *((float4*)&_status);
        }
        else {
            _status._warp_status = decx::scan::_scan_warp_status::AGGREGATE_AVAILABLE;
            _status._warp_aggregate = __fadd_rn(block_scan_res[1].w, __half2float(_end_value));
            _status._end_value = __half2float(_end_value);
            warp_status[STG_status_dex] = *((float4*)&_status);
        }
    }

    if (tidy < proc_dim_v4.y) {
        if (tidx * 2 < proc_dim_v4.x) { dst[STG_dex] = block_scan_res[0]; }
        if (tidx * 2 + 1 < proc_dim_v4.x) { dst[STG_dex + 1] = block_scan_res[1]; }
    }
#endif
}



__global__ void 
decx::scan::GPUK::cu_v_block_inclusive_scan_fp16_2D_v2(const float* __restrict      src,
                                                      float4* __restrict           warp_status, 
                                                      float2* __restrict           dst,
                                                      const uint32_t               Wsrc_v2,
                                                      const uint32_t               Wdst_v2,
                                                      const uint32_t               Wstatus_TP,
                                                      const uint2                  proc_dim_v2)
{
#if __ABOVE_SM_53
    // in-block (local) warp_id = threadIdx.y
    uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;      // H
    uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;      // V

    uint64_t LDG_dex = (uint64_t)tidx + (uint64_t)tidy * 4 * (uint64_t)Wsrc_v2;
    uint64_t STG_dex = (uint64_t)tidx + (uint64_t)tidy * 4 * (uint64_t)Wdst_v2;
    uint64_t STG_status_dex = tidx * 2 * Wstatus_TP + blockIdx.y;

    decx::utils::_cuda_vec128 _recv;
    half2 tmp1, tmp2;
    decx::scan::scan_warp_pred_fp32 _status;
    
    __shared__ float _work_space[8][32 + 1];

    if (tidx < proc_dim_v2.x) {
        if (tidy * 4 < proc_dim_v2.y)  _recv._vf.x = src[LDG_dex];
        if (tidy * 4 + 1 < proc_dim_v2.y)  _recv._vf.y = src[LDG_dex + Wsrc_v2];
        if (tidy * 4 + 2 < proc_dim_v2.y)  _recv._vf.z = src[LDG_dex + Wsrc_v2 * 2];
        if (tidy * 4 + 3 < proc_dim_v2.y)  _recv._vf.w = src[LDG_dex + Wsrc_v2 * 3];

        _recv._vf = decx::scan::GPUK::_inclusive_scan_half4_2way(_recv);

        _work_space[threadIdx.y][threadIdx.x] = _recv._vf.w;        // The last half2
    }

    __syncthreads();

#pragma unroll 3
    for (int i = 0; i < 3; ++i) 
    {
        *((float*)&tmp1) = _work_space[threadIdx.y][threadIdx.x];

        if (threadIdx.y > (1 << i) - 1) {
            *((float*)&tmp2) = _work_space[threadIdx.y - (1 << i)][threadIdx.x];
            tmp1 = __hadd2(tmp1, tmp2);
        }
        __syncthreads();
        if (i < 2) {
            _work_space[threadIdx.y][threadIdx.x] = *((float*)&tmp1);
        }
        else {
            *((__half2*)&_work_space[threadIdx.y][threadIdx.x]) = __hsub2(tmp1, _recv._arrh2[3]);
        }
        __syncthreads();
    }

    // get aggregate for each float4
    *((float*)&tmp1) = _work_space[threadIdx.y][threadIdx.x];

    if (threadIdx.y == blockDim.y - 1/* && tidy < proc_dim.y*/)
    {
        tmp2 = __hadd2(tmp1, _recv._arrh2[3]);
        if (blockIdx.y == 0) {
            _status._prefix_sum = __half2float(tmp2.x);
            _status._warp_status = decx::scan::_scan_warp_status::PREFIX_AVAILABLE;
            warp_status[STG_status_dex] = *((float4*)&_status);

            _status._prefix_sum = __half2float(tmp2.y);
            warp_status[STG_status_dex + Wstatus_TP] = *((float4*)&_status);
        }
        else {
            _status._warp_aggregate = __half2float(tmp2.x);
            _status._warp_status = decx::scan::_scan_warp_status::AGGREGATE_AVAILABLE;
            warp_status[STG_status_dex] = *((float4*)&_status);

            _status._warp_aggregate = __half2float(tmp2.y);
            warp_status[STG_status_dex + Wstatus_TP] = *((float4*)&_status);
        }
    }

    if (tidx < proc_dim_v2.x && tidy < proc_dim_v2.y) 
    {
        _recv._vf = decx::utils::add_scalar_vec4(_recv, tmp1);

        if (tidy * 4 < proc_dim_v2.y) { 
            dst[STG_dex] = make_float2(__half2float(_recv._arrh2[0].x), __half2float(_recv._arrh2[0].y));
        }
        if (tidy * 4 + 1 < proc_dim_v2.y) { 
            dst[STG_dex + Wdst_v2] = make_float2(__half2float(_recv._arrh2[1].x), __half2float(_recv._arrh2[1].y));
        }
        if (tidy * 4 + 2 < proc_dim_v2.y) { 
            dst[STG_dex + Wdst_v2 * 2] = make_float2(__half2float(_recv._arrh2[2].x), __half2float(_recv._arrh2[2].y));
        }
        if (tidy * 4 + 3 < proc_dim_v2.y) {
            dst[STG_dex + Wdst_v2 * 3] = make_float2(__half2float(_recv._arrh2[3].x), __half2float(_recv._arrh2[3].y));
        }
    }
#endif
}



__global__ void 
decx::scan::GPUK::cu_v_block_exclusive_scan_fp16_2D_v2(const float* __restrict      src,
                                                      float4* __restrict           warp_status, 
                                                      float2* __restrict           dst,
                                                      const uint32_t               Wsrc_v2,
                                                      const uint32_t               Wdst_v2,
                                                      const uint32_t               Wstatus_TP,
                                                      const uint2                  proc_dim_v2)
{
#if __ABOVE_SM_53
    // in-block (local) warp_id = threadIdx.y
    uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;      // H
    uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;      // V

    uint64_t LDG_dex = (uint64_t)tidx + (uint64_t)tidy * 4 * (uint64_t)Wsrc_v2;
    uint64_t STG_dex = (uint64_t)tidx + (uint64_t)tidy * 4 * (uint64_t)Wdst_v2;
    uint64_t STG_status_dex = tidx * 2 * Wstatus_TP + blockIdx.y;

    decx::utils::_cuda_vec128 _recv;
    half2 tmp1, tmp2, _end_values;
    decx::scan::scan_warp_pred_fp32 _status;
    
    __shared__ float _work_space[8][32 + 1];

    if (tidx < proc_dim_v2.x) {
        if (tidy * 4 < proc_dim_v2.y)  _recv._vf.x = src[LDG_dex];
        if (tidy * 4 + 1 < proc_dim_v2.y)  _recv._vf.y = src[LDG_dex + Wsrc_v2];
        if (tidy * 4 + 2 < proc_dim_v2.y)  _recv._vf.z = src[LDG_dex + Wsrc_v2 * 2];
        if (tidy * 4 + 3 < proc_dim_v2.y)  _recv._vf.w = src[LDG_dex + Wsrc_v2 * 3];

        _end_values = _recv._arrh2[3];
        _recv._vf = decx::scan::GPUK::_exclusive_scan_half4_2way(_recv);

        *((__half2*)&_work_space[threadIdx.y][threadIdx.x]) = __hadd2(_recv._arrh2[3], _end_values);        // The last half2
    }

    __syncthreads();

#pragma unroll 3
    for (int i = 0; i < 3; ++i) 
    {
        *((float*)&tmp1) = _work_space[threadIdx.y][threadIdx.x];

        if (threadIdx.y > (1 << i) - 1) {
            *((float*)&tmp2) = _work_space[threadIdx.y - (1 << i)][threadIdx.x];
            tmp1 = __hadd2(tmp1, tmp2);
        }
        __syncthreads();
        if (i < 2) {
            _work_space[threadIdx.y][threadIdx.x] = *((float*)&tmp1);
        }
        else {
            *((__half2*)&_work_space[threadIdx.y][threadIdx.x]) = __hsub2(tmp1, __hadd2(_recv._arrh2[3], _end_values));
        }
        __syncthreads();
    }

    // get aggregate for each float4
    *((float*)&tmp1) = _work_space[threadIdx.y][threadIdx.x];

    if (threadIdx.y == blockDim.y - 1/* && tidy < proc_dim.y*/)
    {
        tmp2 = __hadd2(__hadd2(tmp1, _recv._arrh2[3]), _end_values);
        if (blockIdx.y == 0) {
            _status._prefix_sum = __half2float(tmp2.x);
            _status._end_value = __half2float(_end_values.x);
            _status._warp_status = decx::scan::_scan_warp_status::PREFIX_AVAILABLE;
            warp_status[STG_status_dex] = *((float4*)&_status);

            _status._prefix_sum = __half2float(tmp2.y);
            _status._end_value = __half2float(_end_values.y);
            warp_status[STG_status_dex + Wstatus_TP] = *((float4*)&_status);
        }
        else {
            _status._warp_aggregate = __half2float(tmp2.x);
            _status._end_value = __half2float(_end_values.x);
            _status._warp_status = decx::scan::_scan_warp_status::AGGREGATE_AVAILABLE;
            warp_status[STG_status_dex] = *((float4*)&_status);

            _status._warp_aggregate = __half2float(tmp2.y);
            _status._end_value = __half2float(_end_values.y);
            warp_status[STG_status_dex + Wstatus_TP] = *((float4*)&_status);
        }
    }

    if (tidx < proc_dim_v2.x && tidy < proc_dim_v2.y) 
    {
        _recv._vf = decx::utils::add_scalar_vec4(_recv, tmp1);

        if (tidy * 4 < proc_dim_v2.y) { 
            dst[STG_dex] = make_float2(__half2float(_recv._arrh2[0].x), __half2float(_recv._arrh2[0].y));
        }
        if (tidy * 4 + 1 < proc_dim_v2.y) { 
            dst[STG_dex + Wdst_v2] = make_float2(__half2float(_recv._arrh2[1].x), __half2float(_recv._arrh2[1].y));
        }
        if (tidy * 4 + 2 < proc_dim_v2.y) { 
            dst[STG_dex + Wdst_v2 * 2] = make_float2(__half2float(_recv._arrh2[2].x), __half2float(_recv._arrh2[2].y));
        }
        if (tidy * 4 + 3 < proc_dim_v2.y) {
            dst[STG_dex + Wdst_v2 * 3] = make_float2(__half2float(_recv._arrh2[3].x), __half2float(_recv._arrh2[3].y));
        }
    }
#endif
}



template <bool _is_exclusive>
__global__ void 
decx::scan::GPUK::cu_h_scan_DLB_fp32_2D_v8(float4* __restrict     block_status, 
                                        float4* __restrict     dst, 
                                        const uint             Wmat_v4,
                                        const uint             Wstatus, 
                                        const uint2            proc_dim_v8)
{
    // in-block (local) warp_id = threadIdx.y
    uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;      // H
    uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;      // V
    // in-warp (local) lane_id = threadIdx.x % 32
    uint32_t lane_id = threadIdx.x;        // local thread index within a warp

    uint64_t index = (uint64_t)tidx * 2 + (uint64_t)tidy * (uint64_t)Wmat_v4;
    uint64_t STG_status_dex = (uint64_t)blockIdx.x + (uint64_t)tidy * (uint64_t)Wstatus;

    int search_id_x = blockIdx.x - warpSize + lane_id;

    int _crit;
    float4 _recv[2];
    float tmp, tmp1;
    float warp_lookback_aggregate = 0;

    __shared__ float _warp_aggregates[8];

    decx::scan::scan_warp_pred_fp32 _previous_status;
    decx::scan::scan_warp_pred_fp32 _status;

    if (tidx < proc_dim_v8.x && tidy < proc_dim_v8.y) {
        _recv[0] = dst[index];
        _recv[1] = dst[index + 1];
    }

    if (blockIdx.x != 0) 
    {
        for (int i = blockIdx.x - 1; i > -1; --i)
        {
            _previous_status._prefix_sum = 0.f;
            _previous_status._warp_aggregate = 0.f;
            _previous_status._warp_status = decx::scan::_scan_warp_status::AGGREGATE_AVAILABLE;
            if (search_id_x > -1) {
                *((float4*)&_previous_status) = block_status[search_id_x + (uint64_t)tidy * (uint64_t)Wstatus];
            }

            _crit = __ballot_sync(0xffffffff, _previous_status._warp_status == decx::scan::_scan_warp_status::PREFIX_AVAILABLE);
            if (!((bool)_crit)) {
                decx::scan::GPUK::cu_warp_inclusive_scan_fp32<float(float, float), 32>(__fadd_rn, &_previous_status._warp_aggregate, &tmp, lane_id);
                warp_lookback_aggregate = __fadd_rn(warp_lookback_aggregate, tmp);
            }
            else {
                break;
            }

            __syncwarp(0xffffffff);

            search_id_x -= warpSize;
        }

        int critical_position = __ffs(_crit) - 1;
        _previous_status._warp_aggregate = (lane_id == critical_position ? _previous_status._prefix_sum : _previous_status._warp_aggregate);
        _previous_status._warp_aggregate = (lane_id < critical_position ? 0 : _previous_status._warp_aggregate);

        __syncwarp(0xffffffff);

        decx::scan::GPUK::cu_warp_inclusive_scan_fp32<float(float, float), 32>(__fadd_rn, &_previous_status._warp_aggregate, &tmp, lane_id);
        warp_lookback_aggregate = __fadd_rn(warp_lookback_aggregate, tmp);

        __syncwarp(0xffffffff);

        if (lane_id == warpSize - 1) {
            *((float4*)&_status) = block_status[STG_status_dex];

            if (_is_exclusive) {
                _status._prefix_sum = __fadd_rn(__fadd_rn(_recv[1].w, warp_lookback_aggregate), _status._end_value);
            }
            else {
                _status._prefix_sum = __fadd_rn(_recv[1].w, warp_lookback_aggregate);
            }
            _status._warp_status = decx::scan::_scan_warp_status::PREFIX_AVAILABLE;
            block_status[STG_status_dex] = *((float4*)&_status);

            _warp_aggregates[threadIdx.y] = warp_lookback_aggregate;
        }
    }

    __syncthreads();

    if (blockIdx.x != 0) {
        warp_lookback_aggregate = _warp_aggregates[threadIdx.y];
        _recv[0] = decx::utils::add_scalar_vec4(_recv[0], warp_lookback_aggregate);
        _recv[1] = decx::utils::add_scalar_vec4(_recv[1], warp_lookback_aggregate);
    }

    if (tidx < proc_dim_v8.x && tidy < proc_dim_v8.y) {
        dst[index] = _recv[0];
        dst[index + 1] = _recv[1];
    }
}


template __global__ void decx::scan::GPUK::cu_h_scan_DLB_fp32_2D_v8<true>(float4* __restrict _status, float4* __restrict dst, const uint Wmat_v4,
    const uint Wstatus, const uint2 proc_dim_v4);

template __global__ void decx::scan::GPUK::cu_h_scan_DLB_fp32_2D_v8<false>(float4* __restrict _status, float4* __restrict dst, const uint Wmat_v4,
    const uint Wstatus, const uint2 proc_dim_v4);