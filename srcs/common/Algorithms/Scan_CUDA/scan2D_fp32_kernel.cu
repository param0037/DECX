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
decx::scan::GPUK::cu_h_block_inclusive_scan_fp32_2D(const float4* __restrict     src, 
                                                   float4* __restrict           warp_status, 
                                                   float4* __restrict           dst,
                                                   const uint32_t               Wmat_v4,
                                                   const uint32_t               Wstatus,
                                                   const uint2                  proc_dim_v4)
{
    // in-block (local) warp_id = threadIdx.y
    uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;      // H
    uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;      // V
    // in-warp (local) lane_id = threadIdx.x % 32
    uint32_t lane_id = (threadIdx.x & 0x1f);        // local thread index within a warp

    uint64_t index = (uint64_t)tidx + (uint64_t)tidy * (uint64_t)Wmat_v4;

    decx::scan::scan_warp_pred_fp32 _status;

    float4 _recv;
    float tmp, tmp1;

    if (tidx < proc_dim_v4.x && tidy < proc_dim_v4.y) {
        _recv = src[index];
    }

    uint64_t STG_status_dex = (uint64_t)blockIdx.x + (uint64_t)tidy * (uint64_t)Wstatus;

    _recv = decx::scan::GPUK::_inclusive_scan_float4(_recv);

    decx::scan::GPUK::cu_warp_exclusive_scan_fp32<float(float, float), 32>(__fadd_rn, &_recv.w, &tmp, lane_id);
    _recv = decx::utils::add_scalar_vec4(_recv, tmp);

    if (lane_id == warpSize - 1) {
        if (blockIdx.x == 0) {
            _status._warp_status = decx::scan::_scan_warp_status::PREFIX_AVAILABLE;
            _status._prefix_sum = _recv.w;
            _status._warp_aggregate = _recv.w;
            warp_status[STG_status_dex] = *((float4*)&_status);
        }
        else {
            _status._warp_status = decx::scan::_scan_warp_status::AGGREGATE_AVAILABLE;
            _status._warp_aggregate = _recv.w;
            warp_status[STG_status_dex] = *((float4*)&_status);
        }
    }

    if (tidx < proc_dim_v4.x && tidy < proc_dim_v4.y) {
        dst[index] = _recv;
    }
}



__global__ void 
decx::scan::GPUK::cu_h_block_exclusive_scan_fp32_2D(const float4* __restrict     src, 
                                                   float4* __restrict           warp_status, 
                                                   float4* __restrict           dst,
                                                   const uint32_t               Wmat_v4,
                                                   const uint32_t               Wstatus,
                                                   const uint2                  proc_dim_v4)
{
    // in-block (local) warp_id = threadIdx.y
    uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;      // H
    uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;      // V
    // in-warp (local) lane_id = threadIdx.x % 32
    uint32_t lane_id = (threadIdx.x & 0x1f);        // local thread index within a warp

    uint64_t index = (uint64_t)tidx + (uint64_t)tidy * (uint64_t)Wmat_v4;

    decx::scan::scan_warp_pred_fp32 _status;

    float4 _recv;
    float tmp, tmp1, _end_value = 0;

    if (tidx < proc_dim_v4.x && tidy < proc_dim_v4.y) {
        _recv = src[index];
        _end_value = _recv.w;
    }

    uint64_t STG_status_dex = (uint64_t)blockIdx.x + (uint64_t)tidy * (uint64_t)Wstatus;

    _recv = decx::scan::GPUK::_exclusive_scan_float4(_recv);
    tmp1 = __fadd_rn(_recv.w, _end_value);

    decx::scan::GPUK::cu_warp_exclusive_scan_fp32<float(float, float), 32>(__fadd_rn, &tmp1, &tmp, lane_id);
    _recv = decx::utils::add_scalar_vec4(_recv, tmp);

    if (lane_id == warpSize - 1) {
        if (blockIdx.x == 0) {
            _status._warp_status = decx::scan::_scan_warp_status::PREFIX_AVAILABLE;
            _status._prefix_sum = __fadd_rn(_recv.w, _end_value);
            _status._end_value = _end_value;
            _status._warp_aggregate = __fadd_rn(_recv.w, _end_value);
            warp_status[STG_status_dex] = *((float4*)&_status);
        }
        else {
            _status._warp_status = decx::scan::_scan_warp_status::AGGREGATE_AVAILABLE;
            _status._warp_aggregate = __fadd_rn(_recv.w, _end_value);
            _status._end_value = _end_value;
            warp_status[STG_status_dex] = *((float4*)&_status);
        }
    }

    if (tidx < proc_dim_v4.x && tidy < proc_dim_v4.y) {
        dst[index] = _recv;
    }
}


template <bool _is_inplace>
__global__ void 
decx::scan::GPUK::cu_v_block_inclusive_scan_fp32_2D(const float* __restrict      src,
                                                   float4* __restrict           warp_status, 
                                                   float* __restrict            dst,
                                                   const uint32_t               Wmat,
                                                   const uint32_t               Wstatus_TP,
                                                   const uint2                  proc_dim)
{
    // in-block (local) warp_id = threadIdx.y
    uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;      // H
    uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;      // V

    uint64_t index = (uint64_t)tidx + (uint64_t)tidy * 4 * (uint64_t)Wmat;
    uint64_t STG_status_dex = tidx * Wstatus_TP + blockIdx.y;

    float4 _recv;
    float tmp1, tmp2;
    decx::scan::scan_warp_pred_fp32 _status;
    
    __shared__ float _work_space[8][32 + 1];

    if (tidx < proc_dim.x) {
        if (_is_inplace) {
            if (tidy * 4 < proc_dim.y)  _recv.x = dst[index];
            if (tidy * 4 + 1 < proc_dim.y)  _recv.y = dst[index + Wmat];
            if (tidy * 4 + 2 < proc_dim.y)  _recv.z = dst[index + Wmat * 2];
            if (tidy * 4 + 3 < proc_dim.y)  _recv.w = dst[index + Wmat * 3];
        }
        else {
            if (tidy * 4 < proc_dim.y)  _recv.x = src[index];
            if (tidy * 4 + 1 < proc_dim.y)  _recv.y = src[index + Wmat];
            if (tidy * 4 + 2 < proc_dim.y)  _recv.z = src[index + Wmat * 2];
            if (tidy * 4 + 3 < proc_dim.y)  _recv.w = src[index + Wmat * 3];
        }

        _recv = decx::scan::GPUK::_inclusive_scan_float4(_recv);

        _work_space[threadIdx.y][threadIdx.x] = _recv.w;
    }

    __syncthreads();

#pragma unroll 3
    for (int i = 0; i < 3; ++i) 
    {
        tmp1 = _work_space[threadIdx.y][threadIdx.x];

        if (threadIdx.y > (1 << i) - 1) {
            tmp2 = _work_space[threadIdx.y - (1 << i)][threadIdx.x];
            tmp1 = __fadd_rn(tmp1, tmp2);
        }
        __syncthreads();
        if (i < 2) {
            _work_space[threadIdx.y][threadIdx.x] = tmp1;
        }
        else {
            _work_space[threadIdx.y][threadIdx.x] = __fsub_rn(tmp1, _recv.w);
        }
        __syncthreads();
    }

    // get aggregate for each float4
    tmp1 = _work_space[threadIdx.y][threadIdx.x];

    if (threadIdx.y == blockDim.y - 1)
    {
        if (blockIdx.y == 0) {
            _status._prefix_sum = tmp1 + _recv.w;
            _status._warp_status = decx::scan::_scan_warp_status::PREFIX_AVAILABLE;
            warp_status[STG_status_dex] = *((float4*)&_status);
        }
        else {
            _status._warp_aggregate = tmp1 + _recv.w;
            _status._warp_status = decx::scan::_scan_warp_status::AGGREGATE_AVAILABLE;
            warp_status[STG_status_dex] = *((float4*)&_status);
        }
    }

    if (tidx < proc_dim.x && tidy < proc_dim.y) 
    {
        _recv = decx::utils::add_scalar_vec4(_recv, tmp1);

        if (tidy * 4 < proc_dim.y)  dst[index] = _recv.x;
        if (tidy * 4 + 1 < proc_dim.y)  dst[index + Wmat] = _recv.y;
        if (tidy * 4 + 2 < proc_dim.y)  dst[index + Wmat * 2] = _recv.z;
        if (tidy * 4 + 3 < proc_dim.y)  dst[index + Wmat * 3] = _recv.w;
    }
}

template __global__ void decx::scan::GPUK::cu_v_block_inclusive_scan_fp32_2D<true>(const float* __restrict src, float4* __restrict _status, float* __restrict dst,
    const uint32_t Wmat, const uint32_t Wstatus, const uint2 proc_dim);
template __global__ void decx::scan::GPUK::cu_v_block_inclusive_scan_fp32_2D<false>(const float* __restrict src, float4* __restrict _status, float* __restrict dst,
    const uint32_t Wmat, const uint32_t Wstatus, const uint2 proc_dim);




template <bool _is_inplace>
__global__ void 
decx::scan::GPUK::cu_v_block_exclusive_scan_fp32_2D(const float* __restrict      src,
                                                   float4* __restrict           warp_status, 
                                                   float* __restrict            dst,
                                                   const uint32_t               Wmat,
                                                   const uint32_t               Wstatus_TP,
                                                   const uint2                  proc_dim)
{
    // in-block (local) warp_id = threadIdx.y
    uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;      // H
    uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;      // V

    uint64_t index = (uint64_t)tidx + (uint64_t)tidy * 4 * (uint64_t)Wmat;
    uint64_t STG_status_dex = tidx * Wstatus_TP + blockIdx.y;

    float4 _recv = decx::utils::vec4_set1_fp32(0);
    float tmp1, tmp2, _end_value;
    decx::scan::scan_warp_pred_fp32 _status;
    
    __shared__ float _work_space[8][32 + 1];

    if (tidx < proc_dim.x) {
        if (_is_inplace) {
            if (tidy * 4 < proc_dim.y)  _recv.x = dst[index];
            if (tidy * 4 + 1 < proc_dim.y)  _recv.y = dst[index + Wmat];
            if (tidy * 4 + 2 < proc_dim.y)  _recv.z = dst[index + Wmat * 2];
            if (tidy * 4 + 3 < proc_dim.y)  _recv.w = dst[index + Wmat * 3];
        }
        else {
            if (tidy * 4 < proc_dim.y)  _recv.x = src[index];
            if (tidy * 4 + 1 < proc_dim.y)  _recv.y = src[index + Wmat];
            if (tidy * 4 + 2 < proc_dim.y)  _recv.z = src[index + Wmat * 2];
            if (tidy * 4 + 3 < proc_dim.y)  _recv.w = src[index + Wmat * 3];
        }
        _end_value = _recv.w;
        _recv = decx::scan::GPUK::_exclusive_scan_float4(_recv);

        _work_space[threadIdx.y][threadIdx.x] = __fadd_rn(_recv.w, _end_value);
    }

    __syncthreads();

#pragma unroll 3
    for (int i = 0; i < 3; ++i) 
    {
        tmp1 = _work_space[threadIdx.y][threadIdx.x];

        if (threadIdx.y > (1 << i) - 1) {
            tmp2 = _work_space[threadIdx.y - (1 << i)][threadIdx.x];
            tmp1 = __fadd_rn(tmp1, tmp2);
        }
        __syncthreads();
        if (i < 2) {
            _work_space[threadIdx.y][threadIdx.x] = tmp1;
        }
        else {
            _work_space[threadIdx.y][threadIdx.x] = __fsub_rn(tmp1, __fadd_rn(_recv.w, _end_value));
        }
        __syncthreads();
    }

    // get aggregate for each float4
    tmp1 = _work_space[threadIdx.y][threadIdx.x];

    if (threadIdx.y == blockDim.y - 1/* && tidy < proc_dim.y*/)
    {
        if (blockIdx.y == 0) {
            _status._prefix_sum = __fadd_rn(__fadd_rn(tmp1, _recv.w), _end_value);
            _status._end_value = _end_value;
            _status._warp_status = decx::scan::_scan_warp_status::PREFIX_AVAILABLE;
            warp_status[STG_status_dex] = *((float4*)&_status);
        }
        else {
            _status._warp_aggregate = __fadd_rn(__fadd_rn(tmp1, _recv.w), _end_value);
            _status._end_value = _end_value;
            _status._warp_status = decx::scan::_scan_warp_status::AGGREGATE_AVAILABLE;
            warp_status[STG_status_dex] = *((float4*)&_status);
        }
    }

    if (tidx < proc_dim.x && tidy < proc_dim.y) 
    {
        _recv = decx::utils::add_scalar_vec4(_recv, tmp1);

        if (tidy * 4 < proc_dim.y)  dst[index] = _recv.x;
        if (tidy * 4 + 1 < proc_dim.y)  dst[index + Wmat] = _recv.y;
        if (tidy * 4 + 2 < proc_dim.y)  dst[index + Wmat * 2] = _recv.z;
        if (tidy * 4 + 3 < proc_dim.y)  dst[index + Wmat * 3] = _recv.w;
    }
}

template __global__ void decx::scan::GPUK::cu_v_block_exclusive_scan_fp32_2D<true>(const float* __restrict src, float4* __restrict _status, float* __restrict dst,
    const uint32_t Wmat, const uint32_t Wstatus, const uint2 proc_dim);
template __global__ void decx::scan::GPUK::cu_v_block_exclusive_scan_fp32_2D<false>(const float* __restrict src, float4* __restrict _status, float* __restrict dst,
    const uint32_t Wmat, const uint32_t Wstatus, const uint2 proc_dim);



template <bool _is_exclusive>
__global__ void 
decx::scan::GPUK::cu_h_scan_DLB_fp32_2D(float4* __restrict     block_status, 
                                        float4* __restrict     dst, 
                                        const uint             Wmat_v4,
                                        const uint             Wstatus, 
                                        const uint2            proc_dim_v4)
{
    // in-block (local) warp_id = threadIdx.y
    uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;      // H
    uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;      // V
    // in-warp (local) lane_id = threadIdx.x % 32
    uint32_t lane_id = threadIdx.x;        // local thread index within a warp

    uint64_t index = (uint64_t)tidx + (uint64_t)tidy * (uint64_t)Wmat_v4;
    uint64_t STG_status_dex = (uint64_t)blockIdx.x + (uint64_t)tidy * (uint64_t)Wstatus;

    int search_id_x = blockIdx.x - warpSize + lane_id;

    int _crit;
    float4 _recv;
    float tmp, tmp1;
    float warp_lookback_aggregate = 0;

    __shared__ float _warp_aggregates[8];

    decx::scan::scan_warp_pred_fp32 _previous_status;
    decx::scan::scan_warp_pred_fp32 _status;

    if (tidx < proc_dim_v4.x && tidy < proc_dim_v4.y) {
        _recv = dst[index];
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
                _status._prefix_sum = __fadd_rn(__fadd_rn(_recv.w, warp_lookback_aggregate), _status._end_value);
            }
            else {
                _status._prefix_sum = __fadd_rn(_recv.w, warp_lookback_aggregate);
            }
            _status._warp_status = decx::scan::_scan_warp_status::PREFIX_AVAILABLE;
            block_status[STG_status_dex] = *((float4*)&_status);

            _warp_aggregates[threadIdx.y] = warp_lookback_aggregate;
        }
    }

    __syncthreads();

    if (blockIdx.x != 0) {
        warp_lookback_aggregate = _warp_aggregates[threadIdx.y];
        _recv = decx::utils::add_scalar_vec4(_recv, warp_lookback_aggregate);
    }

    if (tidx < proc_dim_v4.x && tidy < proc_dim_v4.y) {
        dst[index] = _recv;
    }
}


template __global__ void decx::scan::GPUK::cu_h_scan_DLB_fp32_2D<true>(float4* __restrict _status, float4* __restrict dst, const uint Wmat_v4,
    const uint Wstatus, const uint2 proc_dim_v4);

template __global__ void decx::scan::GPUK::cu_h_scan_DLB_fp32_2D<false>(float4* __restrict _status, float4* __restrict dst, const uint Wmat_v4,
    const uint Wstatus, const uint2 proc_dim_v4);




template <bool _is_exclusive>
__global__ void 
decx::scan::GPUK::cu_v_scan_DLB_fp32_2D(float4* __restrict      _warp_status, 
                                        float* __restrict       dst, 
                                        const uint              Wmat,
                                        const uint              Wstatus_TP, 
                                        const uint2             proc_dim)
{
    // in-block (local) warp_id = threadIdx.y
    uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;      // H
    uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;      // V

    uint64_t STG_LDG_dex = (uint64_t)tidx + (uint64_t)tidy * (uint64_t)Wmat;
    
    __shared__ float end_values[32];
    __shared__ float prefix_sums[32];

    float _recv;
    float tmp, warp_lookback_aggregate = 0;
    decx::scan::scan_warp_pred_fp32 _previous_status, _status;

    int _crit;

    if (tidx < proc_dim.x && tidy < proc_dim.y) {
        _recv = dst[STG_LDG_dex];
    }

    if (threadIdx.y == blockDim.y - 1) {
        end_values[threadIdx.x] = _recv;
    }

    __syncthreads();

    uint64_t base = (threadIdx.y + blockIdx.x * blockDim.y) * Wstatus_TP;
    int search_id_x = blockIdx.y - warpSize + threadIdx.x;

    if (blockIdx.y != 0) 
    {
        for (int i = blockIdx.y - 1; i > -1; --i) 
        {
            _previous_status._prefix_sum = 0.f;
            _previous_status._warp_aggregate = 0.f;
            _previous_status._warp_status = decx::scan::_scan_warp_status::AGGREGATE_AVAILABLE;
            if (search_id_x > -1) {
                *((float4*)&_previous_status) = _warp_status[search_id_x + base];
            }

            _crit = __ballot_sync(0xffffffff, _previous_status._warp_status == decx::scan::_scan_warp_status::PREFIX_AVAILABLE);
            if (!((bool)_crit)) {
                decx::scan::GPUK::cu_warp_inclusive_scan_fp32<float(float, float), 32>(__fadd_rn, &_previous_status._warp_aggregate, &tmp, threadIdx.x);
                warp_lookback_aggregate = __fadd_rn(warp_lookback_aggregate, tmp);
            }
            else {
                break;
            }

            __syncwarp(0xffffffff);

            search_id_x -= warpSize;
        }

        int critical_position = __ffs(_crit) - 1;
        _previous_status._warp_aggregate = (threadIdx.x == critical_position ? _previous_status._prefix_sum : _previous_status._warp_aggregate);
        _previous_status._warp_aggregate = (threadIdx.x < critical_position ? 0 : _previous_status._warp_aggregate);

        __syncwarp(0xffffffff);

        decx::scan::GPUK::cu_warp_inclusive_scan_fp32<float(float, float), 32>(__fadd_rn, &_previous_status._warp_aggregate, &tmp, threadIdx.x);
        warp_lookback_aggregate = __fadd_rn(warp_lookback_aggregate, tmp);

        __syncwarp(0xffffffff);

        if (threadIdx.x == warpSize - 1) {
            *((float4*)&_status) = _warp_status[base + blockIdx.y];

            prefix_sums[threadIdx.y] = warp_lookback_aggregate;

            if (_is_exclusive) {
                _status._prefix_sum = __fadd_rn(__fadd_rn(end_values[threadIdx.y], warp_lookback_aggregate), _status._end_value);
            }
            else {
                _status._prefix_sum = __fadd_rn(end_values[threadIdx.y], warp_lookback_aggregate);
            }
            _status._warp_status = decx::scan::_scan_warp_status::PREFIX_AVAILABLE;

            _warp_status[base + blockIdx.y] = *((float4*)&_status);
        }

        __syncthreads();

        tmp = prefix_sums[threadIdx.x];
        _recv = __fadd_rn(_recv, tmp);
    }

    if (tidx < proc_dim.x && tidy < proc_dim.y) {
        dst[STG_LDG_dex] = _recv;
    }
}



template __global__ void decx::scan::GPUK::cu_v_scan_DLB_fp32_2D<true>(float4* __restrict _status, float* __restrict dst, const uint Wmat_v4,
    const uint Wstatus, const uint2 proc_dim_v4);

template __global__ void decx::scan::GPUK::cu_v_scan_DLB_fp32_2D<false>(float4* __restrict _status, float* __restrict dst, const uint Wmat_v4,
    const uint Wstatus, const uint2 proc_dim_v4);