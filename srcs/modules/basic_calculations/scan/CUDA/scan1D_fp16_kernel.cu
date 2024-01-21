/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#include "scan.cuh"


__global__ void
decx::scan::GPUK::cu_block_inclusive_scan_fp16_1D(const float4* __restrict   src,
                                                 float4* __restrict         _block_status,
                                                 float4* __restrict         dst,
                                                 const uint64_t             proc_len_v8)
{
#if __ABOVE_SM_53
    uint64_t tid = (uint64_t)threadIdx.x + (uint64_t)blockIdx.x * (uint64_t)blockDim.x;
    uint32_t lane_id = (threadIdx.x & 0x1f);        // local thread index within a warp
    int local_warp_id = threadIdx.x / warpSize;
    int global_warp_id = tid / warpSize;
    int _crit;

    __shared__ float _warp_aggregates[32];       // 256 / 32 (threads per warp) = 8 warps, but extended to 32

    decx::utils::_cuda_vec128 _recv;
    __half2 tmp_h2, tmp1_h2;
    float& tmp_fp32 = *((float*)&tmp_h2), &tmp1_fp32 = *((float*)&tmp1_h2);
    float4 block_scan_res[2];

    decx::scan::scan_warp_pred_fp32 _previous_status;
    _previous_status._prefix_sum = 0;
    _previous_status._warp_aggregate = 0;
    _previous_status._warp_status = decx::scan::_scan_warp_status::AGGREGATE_AVAILABLE;

    // first load the data linearly from global memory to shared memory
    if (tid < proc_len_v8)
    {
        _recv._vf = src[tid];
    }
    decx::scan::GPUK::_inclusive_scan_half8(_recv, &_recv);

    decx::scan::GPUK::cu_warp_exclusive_scan_fp16<__half(__half, __half), 32>(__hadd, &_recv._arrh[7], &tmp_h2.x, lane_id);
    _recv._vf = decx::utils::add_scalar_vec4(_recv, tmp_h2.x);

    if (lane_id == warpSize - 1) {
        _warp_aggregates[local_warp_id] = __half2float(_recv._arrh[7]);
    }

    __syncthreads();

    if (local_warp_id == 0)
    {
        tmp1_fp32 = _warp_aggregates[threadIdx.x];
        tmp1_fp32 = threadIdx.x < 8 ? tmp1_fp32 : 0.f;
        __syncwarp(0xffffffff);
        decx::scan::GPUK::cu_warp_exclusive_scan_fp32<float(float, float), 8>(__fadd_rn, &tmp1_fp32, &tmp_fp32, lane_id);
        __syncwarp(0xffffffff);

        if (lane_id < 8) { _warp_aggregates[lane_id] = tmp_fp32; }
    }

    __syncthreads();

    tmp_fp32 = _warp_aggregates[local_warp_id];

    block_scan_res[0].x = __fadd_rn(__half2float(_recv._arrh[0]), tmp_fp32);
    block_scan_res[0].y = __fadd_rn(__half2float(_recv._arrh[1]), tmp_fp32);
    block_scan_res[0].z = __fadd_rn(__half2float(_recv._arrh[2]), tmp_fp32);
    block_scan_res[0].w = __fadd_rn(__half2float(_recv._arrh[3]), tmp_fp32);
    block_scan_res[1].x = __fadd_rn(__half2float(_recv._arrh[4]), tmp_fp32);
    block_scan_res[1].y = __fadd_rn(__half2float(_recv._arrh[5]), tmp_fp32);
    block_scan_res[1].z = __fadd_rn(__half2float(_recv._arrh[6]), tmp_fp32);
    block_scan_res[1].w = __fadd_rn(__half2float(_recv._arrh[7]), tmp_fp32);

    __syncthreads();

    decx::scan::scan_warp_pred_fp32 _status;
    
    if (threadIdx.x == blockDim.x - 1)
    {
        if (blockIdx.x == 0)
        {
            _status._warp_status = decx::scan::_scan_warp_status::PREFIX_AVAILABLE;
            _status._prefix_sum = block_scan_res[1].w;
            _status._warp_aggregate = block_scan_res[1].w;
            _block_status[0] = *((float4*)&_status);
        }
        else {
            _status._warp_status = decx::scan::_scan_warp_status::AGGREGATE_AVAILABLE;
            _status._warp_aggregate = block_scan_res[1].w;
            _block_status[blockIdx.x] = *((float4*)&_status);
        }
    }

    if (tid < proc_len_v8) {
        dst[tid * 2] = block_scan_res[0];
        dst[tid * 2 + 1] = block_scan_res[1];
    }
#endif
}



__global__ void
decx::scan::GPUK::cu_block_exclusive_scan_fp16_1D(const float4* __restrict   src,
                                                 float4* __restrict         _block_status,
                                                 float4* __restrict         dst,
                                                 const uint64_t             proc_len_v8)
{
#if __ABOVE_SM_53
    uint64_t tid = (uint64_t)threadIdx.x + (uint64_t)blockIdx.x * (uint64_t)blockDim.x;
    uint32_t lane_id = (threadIdx.x & 0x1f);        // local thread index within a warp
    int local_warp_id = threadIdx.x / warpSize;
    int global_warp_id = tid / warpSize;
    int _crit;

    __shared__ float _warp_aggregates[32];       // 256 / 32 (threads per warp) = 8 warps, but extended to 32

    decx::utils::_cuda_vec128 _recv;
    __half2 tmp_h2, tmp1_h2;
    float& tmp_fp32 = *((float*)&tmp_h2), &tmp1_fp32 = *((float*)&tmp1_h2);
    float4 block_scan_res[2];

    __half _end_value;

    decx::scan::scan_warp_pred_fp32 _previous_status;
    _previous_status._prefix_sum = 0;
    _previous_status._warp_aggregate = 0;
    _previous_status._warp_status = decx::scan::_scan_warp_status::AGGREGATE_AVAILABLE;

    // first load the data linearly from global memory to shared memory
    if (tid < proc_len_v8)
    {
        _recv._vf = src[tid];
        _end_value = _recv._arrh[7];
    }
    decx::scan::GPUK::_exclusive_scan_half8_inp(&_recv);
    tmp1_h2.x = __hadd(_recv._arrh[7], _end_value);

    decx::scan::GPUK::cu_warp_exclusive_scan_fp16<__half(__half, __half), 32>(__hadd, &tmp1_h2.x, &tmp_h2.x, lane_id);
    _recv._vf = decx::utils::add_scalar_vec4(_recv, tmp_h2.x);

    if (lane_id == warpSize - 1) {
        _warp_aggregates[local_warp_id] = __half2float(__hadd(_recv._arrh[7], _end_value));
    }

    __syncthreads();

    if (local_warp_id == 0)
    {
        tmp1_fp32 = _warp_aggregates[threadIdx.x];
        tmp1_fp32 = threadIdx.x < 8 ? tmp1_fp32 : 0.f;
        __syncwarp(0xffffffff);
        decx::scan::GPUK::cu_warp_exclusive_scan_fp32<float(float, float), 8>(__fadd_rn, &tmp1_fp32, &tmp_fp32, lane_id);
        __syncwarp(0xffffffff);

        if (lane_id < 8) { _warp_aggregates[lane_id] = tmp_fp32; }
    }

    __syncthreads();

    tmp_fp32 = _warp_aggregates[local_warp_id];

    block_scan_res[0].x = __fadd_rn(__half2float(_recv._arrh[0]), tmp_fp32);
    block_scan_res[0].y = __fadd_rn(__half2float(_recv._arrh[1]), tmp_fp32);
    block_scan_res[0].z = __fadd_rn(__half2float(_recv._arrh[2]), tmp_fp32);
    block_scan_res[0].w = __fadd_rn(__half2float(_recv._arrh[3]), tmp_fp32);
    block_scan_res[1].x = __fadd_rn(__half2float(_recv._arrh[4]), tmp_fp32);
    block_scan_res[1].y = __fadd_rn(__half2float(_recv._arrh[5]), tmp_fp32);
    block_scan_res[1].z = __fadd_rn(__half2float(_recv._arrh[6]), tmp_fp32);
    block_scan_res[1].w = __fadd_rn(__half2float(_recv._arrh[7]), tmp_fp32);

    __syncthreads();

    decx::scan::scan_warp_pred_fp32 _status;
    
    if (threadIdx.x == blockDim.x - 1)
    {
        if (blockIdx.x == 0)
        {
            _status._warp_status = decx::scan::_scan_warp_status::PREFIX_AVAILABLE;
            _status._prefix_sum = __fadd_rn(block_scan_res[1].w, __half2float(_end_value));
            _status._end_value = __half2float(_end_value);
            _status._warp_aggregate = __fadd_rn(block_scan_res[1].w, __half2float(_end_value));
            _block_status[0] = *((float4*)&_status);
        }
        else {
            _status._warp_status = decx::scan::_scan_warp_status::AGGREGATE_AVAILABLE;
            _status._warp_aggregate = __fadd_rn(block_scan_res[1].w, __half2float(_end_value));
            _status._end_value = __half2float(_end_value);
            _block_status[blockIdx.x] = *((float4*)&_status);
        }
    }

    if (tid < proc_len_v8) {
        dst[tid * 2] = block_scan_res[0];
        dst[tid * 2 + 1] = block_scan_res[1];
    }
#endif
}




template <bool _is_exclusive>
__global__ void 
decx::scan::GPUK::cu_scan_DLB_fp32_1D_v8(float4* __restrict      _block_status, 
                                                   float4* __restrict      dst, 
                                                   const uint64_t          proc_len_v8)
{
    uint64_t tid = (uint64_t)threadIdx.x + (uint64_t)blockIdx.x * (uint64_t)blockDim.x;
    uint32_t lane_id = (threadIdx.x & 0x1f);        // local thread index within a warp
    int local_warp_id = threadIdx.x / warpSize;
    int global_warp_id = tid / warpSize;
    int _crit;

    int search_id = blockIdx.x - warpSize + lane_id;

    __shared__ float _warp_aggregates;

    float4 _recv[2];
    float tmp, tmp1;

    float warp_lookback_aggregate = 0;

    decx::scan::scan_warp_pred_fp32 _previous_status;
    decx::scan::scan_warp_pred_fp32 _status;

    _previous_status._prefix_sum = 0;
    _previous_status._warp_aggregate = 0;
    _previous_status._warp_status = decx::scan::_scan_warp_status::AGGREGATE_AVAILABLE;

    if (tid < proc_len_v8)
    {
        _recv[0] = dst[tid * 2];
        _recv[1] = dst[tid * 2 + 1];
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
                _status._prefix_sum = __fadd_rn(__fadd_rn(_recv[1].w, warp_lookback_aggregate), _status._end_value);
            }
            else {
                _status._prefix_sum = __fadd_rn(_recv[1].w, warp_lookback_aggregate);
            }
            _status._warp_status = decx::scan::_scan_warp_status::PREFIX_AVAILABLE;
            _block_status[blockIdx.x] = *((float4*)&_status);

            _warp_aggregates = warp_lookback_aggregate;
        }
    }

    __syncthreads();

    if (tid < proc_len_v8) {
        warp_lookback_aggregate = _warp_aggregates;
        _recv[0] = decx::utils::add_scalar_vec4(_recv[0], warp_lookback_aggregate);
        _recv[1] = decx::utils::add_scalar_vec4(_recv[1], warp_lookback_aggregate);

        dst[tid * 2] = _recv[0];
        dst[tid * 2 + 1] = _recv[1];
    }
}


template __global__ void decx::scan::GPUK::cu_scan_DLB_fp32_1D_v8<true>(float4* _status, float4* dst, const uint64_t proc_len_v4);
template __global__ void decx::scan::GPUK::cu_scan_DLB_fp32_1D_v8<false>(float4* _status, float4* dst, const uint64_t proc_len_v4);