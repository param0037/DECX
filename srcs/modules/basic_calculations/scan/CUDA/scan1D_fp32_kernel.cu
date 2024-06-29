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
decx::scan::GPUK::cu_block_inclusive_scan_fp32_1D(const float4* __restrict   src,
                                                 float4* __restrict         _block_status,
                                                 float4* __restrict         dst,
                                                 const uint64_t             proc_len_v4)
{
    uint64_t tid = (uint64_t)threadIdx.x + (uint64_t)blockIdx.x * (uint64_t)blockDim.x;
    uint32_t lane_id = (threadIdx.x & 0x1f);        // local thread index within a warp
    int local_warp_id = threadIdx.x / warpSize;
    int global_warp_id = tid / warpSize;
    int _crit;

    __shared__ float _warp_aggregates[32];       // 256 / 32 (threads per warp) = 8 warps, but extended to 32

    float4 _recv;
    float tmp, tmp1;

    decx::scan::scan_warp_pred_fp32 _previous_status;
    _previous_status._prefix_sum = 0;
    _previous_status._warp_aggregate = 0;
    _previous_status._warp_status = decx::scan::_scan_warp_status::AGGREGATE_AVAILABLE;

    // first load the data linearly from global memory to shared memory
    if (tid < proc_len_v4) {
        _recv = src[tid];
    }
    _recv = decx::scan::GPUK::_inclusive_scan_float4(_recv);

    decx::scan::GPUK::cu_warp_exclusive_scan_fp32<float(float, float), 32>(__fadd_rn, &_recv.w, &tmp, lane_id);
    _recv = decx::utils::add_scalar_vec4(_recv, tmp);

    if (lane_id == warpSize - 1) {
        _warp_aggregates[local_warp_id] = _recv.w;
    }

    __syncthreads();

    if (local_warp_id == 0)
    {
        tmp1 = _warp_aggregates[threadIdx.x];
        tmp1 = threadIdx.x < 8 ? tmp1 : 0;
        __syncwarp(0xffffffff);
        decx::scan::GPUK::cu_warp_exclusive_scan_fp32<float(float, float), 8>(__fadd_rn, &tmp1, &tmp, lane_id);
        __syncwarp(0xffffffff);

        if (lane_id < 8) { _warp_aggregates[lane_id] = tmp; }
    }

    __syncthreads();

    tmp = _warp_aggregates[local_warp_id];
    _recv = decx::utils::add_scalar_vec4(_recv, tmp);

    __syncthreads();

    decx::scan::scan_warp_pred_fp32 _status;
    
    if (threadIdx.x == blockDim.x - 1)
    {
        if (blockIdx.x == 0)
        {
            _status._warp_status = decx::scan::_scan_warp_status::PREFIX_AVAILABLE;
            _status._prefix_sum = _recv.w;
            _status._warp_aggregate = _recv.w;
            _block_status[0] = *((float4*)&_status);
        }
        else {
            _status._warp_status = decx::scan::_scan_warp_status::AGGREGATE_AVAILABLE;
            _status._warp_aggregate = _recv.w;
            _block_status[blockIdx.x] = *((float4*)&_status);
        }
    }

    if (tid < proc_len_v4) {
        dst[tid] = _recv;
    }
}



__global__ void
decx::scan::GPUK::cu_block_exclusive_scan_fp32_1D(const float4* __restrict   src,
                                                 float4* __restrict         _block_status,
                                                 float4* __restrict         dst,
                                                 const uint64_t             proc_len_v4)
{
    uint64_t tid = (uint64_t)threadIdx.x + (uint64_t)blockIdx.x * (uint64_t)blockDim.x;
    uint32_t lane_id = (threadIdx.x & 0x1f);        // local thread index within a warp
    int local_warp_id = threadIdx.x / warpSize;
    int global_warp_id = tid / warpSize;
    int _crit;

    __shared__ float _warp_aggregates[32];       // 256 / 32 (threads per warp) = 8 warps, but extended to 32

    float4 _recv;
    float tmp, tmp1;

    float _end_value, _inclusive_aggregate;

    decx::scan::scan_warp_pred_fp32 _previous_status;
    _previous_status._prefix_sum = 0;
    _previous_status._warp_aggregate = 0;
    _previous_status._warp_status = decx::scan::_scan_warp_status::AGGREGATE_AVAILABLE;

    // first load the data linearly from global memory to shared memory
    if (tid < proc_len_v4)
    {
        _recv = src[tid];
    }
    _end_value = _recv.w;
    _recv = decx::scan::GPUK::_exclusive_scan_float4(_recv);
    _inclusive_aggregate = __fadd_rn(_recv.w, _end_value);

    decx::scan::GPUK::cu_warp_exclusive_scan_fp32<float(float, float), 32>(__fadd_rn, &_inclusive_aggregate, &tmp, lane_id);
    _recv = decx::utils::add_scalar_vec4(_recv, tmp);

    if (lane_id == warpSize - 1) {
        _warp_aggregates[local_warp_id] = __fadd_rn(_recv.w, _end_value);
    }

    __syncthreads();

    if (local_warp_id == 0)
    {
        tmp1 = _warp_aggregates[threadIdx.x];
        tmp1 = threadIdx.x < 8 ? tmp1 : 0;
        __syncwarp(0xffffffff);
        decx::scan::GPUK::cu_warp_exclusive_scan_fp32<float(float, float), 8>(__fadd_rn, &tmp1, &tmp, lane_id);
        __syncwarp(0xffffffff);

        if (lane_id < 8) { _warp_aggregates[lane_id] = tmp; }
    }

    __syncthreads();

    tmp = _warp_aggregates[local_warp_id];
    _recv = decx::utils::add_scalar_vec4(_recv, tmp);

    __syncthreads();

    decx::scan::scan_warp_pred_fp32 _status;
    
    if (threadIdx.x == blockDim.x - 1)
    {
        if (blockIdx.x == 0)
        {
            _status._warp_status = decx::scan::_scan_warp_status::PREFIX_AVAILABLE;
            _status._end_value = _end_value;
            _status._prefix_sum = __fadd_rn(_recv.w, _end_value);
            _status._warp_aggregate = __fadd_rn(_recv.w, _end_value);
            _block_status[0] = *((float4*)&_status);
        }
        else {
            _status._warp_status = decx::scan::_scan_warp_status::AGGREGATE_AVAILABLE;
            _status._end_value = _end_value;
            _status._warp_aggregate = __fadd_rn(_recv.w, _end_value);
            _block_status[blockIdx.x] = *((float4*)&_status);
        }
    }

    if (tid < proc_len_v4) {
        dst[tid] = _recv;
    }
}


template <bool _is_exclusive>
__global__ void 
decx::scan::GPUK::cu_scan_DLB_fp32_1D(float4* __restrict      _block_status, 
                                        float4* __restrict      dst, 
                                        const uint64_t          proc_len_v4)
{
    uint64_t tid = (uint64_t)threadIdx.x + (uint64_t)blockIdx.x * (uint64_t)blockDim.x;
    uint32_t lane_id = (threadIdx.x & 0x1f);        // local thread index within a warp
    int local_warp_id = threadIdx.x / warpSize;
    int global_warp_id = tid / warpSize;
    int _crit;

    int search_id = blockIdx.x - warpSize + lane_id;

    __shared__ float _warp_aggregates;

    float4 _recv;
    float tmp, tmp1;

    float warp_lookback_aggregate = 0;

    decx::scan::scan_warp_pred_fp32 _previous_status;
    decx::scan::scan_warp_pred_fp32 _status;

    if (tid < proc_len_v4) {
        _recv = dst[tid];
    }

    __syncthreads();

    if (local_warp_id == (blockDim.x / warpSize) - 1 && blockIdx.x != 0)
    {
        for (int i = blockIdx.x - 1; i > -1; --i)
        {
            _previous_status._prefix_sum = 0.f;
            _previous_status._warp_aggregate = 0.f;
            _previous_status._warp_status = decx::scan::_scan_warp_status::AGGREGATE_AVAILABLE;
            if (search_id > -1) {
                *((float4*)&_previous_status) = _block_status[search_id];
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

            search_id -= warpSize;
        }

        int critical_position = __ffs(_crit) - 1;
        _previous_status._warp_aggregate = (lane_id == critical_position ? _previous_status._prefix_sum : _previous_status._warp_aggregate);
        _previous_status._warp_aggregate = (lane_id < critical_position ? 0 : _previous_status._warp_aggregate);

        __syncwarp(0xffffffff);

        decx::scan::GPUK::cu_warp_inclusive_scan_fp32<float(float, float), 32>(__fadd_rn, &_previous_status._warp_aggregate, &tmp, lane_id);
        warp_lookback_aggregate = __fadd_rn(warp_lookback_aggregate, tmp);

        __syncwarp(0xffffffff);

        if (lane_id == warpSize - 1) {
            *((float4*)&_status) = _block_status[blockIdx.x];

            if (_is_exclusive) {
                _status._prefix_sum = __fadd_rn(__fadd_rn(_recv.w, warp_lookback_aggregate), _status._end_value);
            }
            else {
                _status._prefix_sum = __fadd_rn(_recv.w, warp_lookback_aggregate);
            }
            _status._warp_status = decx::scan::_scan_warp_status::PREFIX_AVAILABLE;
            _block_status[blockIdx.x] = *((float4*)&_status);

            _warp_aggregates = warp_lookback_aggregate;
        }
    }

    __syncthreads();

    if (tid < proc_len_v4) {
        warp_lookback_aggregate = _warp_aggregates;
        _recv = decx::utils::add_scalar_vec4(_recv, warp_lookback_aggregate);
        dst[tid] = _recv;
    }
}


template __global__ void decx::scan::GPUK::cu_scan_DLB_fp32_1D<true>(float4* _status, float4* dst, const uint64_t proc_len_v4);

template __global__ void decx::scan::GPUK::cu_scan_DLB_fp32_1D<false>(float4* _status, float4* dst, const uint64_t proc_len_v4);