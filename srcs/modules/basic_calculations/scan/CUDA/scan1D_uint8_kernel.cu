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
decx::scan::GPUK::cu_block_inclusive_scan_u8_u16_1D(const float2* __restrict   src,
                                                 float4* __restrict         _block_status,
                                                 int4* __restrict           dst,
                                                 const uint64_t             proc_len_v8)
{
    uint64_t tid = (uint64_t)threadIdx.x + (uint64_t)blockIdx.x * (uint64_t)blockDim.x;
    uint32_t lane_id = (threadIdx.x & 0x1f);        // local thread index within a warp
    int local_warp_id = threadIdx.x / warpSize;     // [0, 8)
    int global_warp_id = tid / warpSize;

    // 256 / 32 (threads per warp) = 8 warps, but extended to 256 * 16 = 4096 int32_t for collasping memory accessing
    __shared__ ushort _warp_aggregates[8];

    decx::utils::_cuda_vec64 _recv;
    decx::utils::_cuda_vec128 _thread_scan_res;
    ushort tmp, tmp1;
    
    /*decx::scan::scan_warp_pred_fp32 _previous_status;
    _previous_status._prefix_sum = 0;
    _previous_status._warp_aggregate = 0;
    _previous_status._warp_status = decx::scan::_scan_warp_status::AGGREGATE_AVAILABLE;*/

    // first load the data linearly from global memory to shared memory
    if (tid < proc_len_v8) {
        _recv._vf2 = src[tid];
    }
    decx::scan::GPUK::_inclusive_scan_uc8_u16(_recv, (ushort*)&_thread_scan_res);

    decx::scan::GPUK::cu_warp_exclusive_scan_u16<ushort(ushort, ushort), 32>(decx::utils::cuda::__u16_add, &_thread_scan_res._arrs[7], &tmp, lane_id);
    _thread_scan_res._vi = decx::utils::add_scalar_vec4(_thread_scan_res, tmp);

    if (lane_id == warpSize - 1) {
        _warp_aggregates[local_warp_id] = _thread_scan_res._arrs[7];
    }

    __syncthreads();

    if (local_warp_id == 0)
    {
        tmp1 = _warp_aggregates[threadIdx.x];
        tmp1 = threadIdx.x < 8 ? tmp1 : __float2half(0);
        __syncwarp(0xffffffff);
        decx::scan::GPUK::cu_warp_exclusive_scan_u16<ushort(ushort, ushort), 8>(decx::utils::cuda::__u16_add, &tmp1, &tmp, lane_id);
        __syncwarp(0xffffffff);

        if (lane_id < 8) { _warp_aggregates[lane_id] = tmp; }
    }

    __syncthreads();

    tmp = _warp_aggregates[local_warp_id];
    _thread_scan_res._vi = decx::utils::add_scalar_vec4(_thread_scan_res, tmp);

    __syncthreads();

    decx::scan::scan_warp_pred_int32 _status;
    
    if (threadIdx.x == blockDim.x - 1)
    {
        if (blockIdx.x == 0)
        {
            _status._warp_status = decx::scan::_scan_warp_status::PREFIX_AVAILABLE;
            _status._prefix_sum = _thread_scan_res._arrs[7];
        }
        else {
            _status._warp_status = decx::scan::_scan_warp_status::AGGREGATE_AVAILABLE;
        }
        _status._warp_aggregate = _thread_scan_res._arrs[7];
        _block_status[blockIdx.x] = *((float4*)&_status);
    }

    __syncthreads();

    uint64_t store_dex = (uint64_t)threadIdx.x + (uint64_t)blockIdx.x * (uint64_t)blockDim.x;
    if (store_dex < proc_len_v8) {
        dst[store_dex] = _thread_scan_res._vi;
    }
}



__global__ void
decx::scan::GPUK::cu_block_exclusive_scan_u8_fp16_1D(const float2* __restrict   src,
                                                 float4* __restrict         _block_status,
                                                 int4* __restrict           dst,
                                                 const uint64_t             proc_len_v8)
{
    uint64_t tid = (uint64_t)threadIdx.x + (uint64_t)blockIdx.x * (uint64_t)blockDim.x;
    uint32_t lane_id = (threadIdx.x & 0x1f);        // local thread index within a warp
    int local_warp_id = threadIdx.x / warpSize;     // [0, 8)
    int global_warp_id = tid / warpSize;

    // 256 / 32 (threads per warp) = 8 warps, but extended to 256 * 16 = 4096 int32_t for collasping memory accessing
    __shared__ ushort _warp_aggregates[8];

    decx::utils::_cuda_vec64 _recv;
    decx::utils::_cuda_vec128 _thread_scan_res;
    ushort tmp, tmp1;
    
    decx::scan::scan_warp_pred_fp32 _previous_status;
    /*_previous_status._prefix_sum = 0;
    _previous_status._warp_aggregate = 0;
    _previous_status._warp_status = decx::scan::_scan_warp_status::AGGREGATE_AVAILABLE;*/

    // first load the data linearly from global memory to shared memory
    if (tid < proc_len_v8) {
        _recv._vf2 = src[tid];
    }
    decx::scan::GPUK::_exclusive_scan_uc8_u16(_recv, (ushort*)&_thread_scan_res);

    tmp1 = _recv._v_uint8[7] + _thread_scan_res._arrs[7];
    decx::scan::GPUK::cu_warp_exclusive_scan_u16<ushort(ushort, ushort), 32>(decx::utils::cuda::__u16_add, &tmp1, &tmp, lane_id);
    _thread_scan_res._vi = decx::utils::add_scalar_vec4(_thread_scan_res, tmp);

    if (lane_id == warpSize - 1) {
        _warp_aggregates[local_warp_id] = _thread_scan_res._arrs[7] + _recv._v_uint8[7];
    }

    __syncthreads();

    if (local_warp_id == 0)
    {
        tmp1 = _warp_aggregates[threadIdx.x];
        tmp1 = threadIdx.x < 8 ? tmp1 : __float2half(0);
        __syncwarp(0xffffffff);
        decx::scan::GPUK::cu_warp_exclusive_scan_u16<ushort(ushort, ushort), 8>(decx::utils::cuda::__u16_add, &tmp1, &tmp, lane_id);
        __syncwarp(0xffffffff);

        if (lane_id < 8) { _warp_aggregates[lane_id] = tmp; }
    }

    __syncthreads();

    tmp = _warp_aggregates[local_warp_id];
    _thread_scan_res._vi = decx::utils::add_scalar_vec4(_thread_scan_res, tmp);

    __syncthreads();

    decx::scan::scan_warp_pred_int32 _status;
    
    if (threadIdx.x == blockDim.x - 1)
    {
        if (blockIdx.x == 0)
        {
            _status._warp_status = decx::scan::_scan_warp_status::PREFIX_AVAILABLE;
            _status._prefix_sum = _thread_scan_res._arrs[7] + _recv._v_uint8[7];
        }
        else {
            _status._warp_status = decx::scan::_scan_warp_status::AGGREGATE_AVAILABLE;
        }

        _status._warp_aggregate = _thread_scan_res._arrs[7] + _recv._v_uint8[7];
        _status._end_value = _recv._v_uint8[7];
        _block_status[blockIdx.x] = *((float4*)&_status);
    }

    __syncthreads();

    uint64_t store_dex = (uint64_t)threadIdx.x + (uint64_t)blockIdx.x * (uint64_t)blockDim.x;
    if (store_dex < proc_len_v8) {
        dst[store_dex] = _thread_scan_res._vi;
    }
}



template <bool _is_exclusive>
__global__ void
decx::scan::GPUK::cu_block_DLB_u16_i32_1D_v8(const float4* __restrict src,
                                        float4* __restrict      _block_status, 
                                        int4* __restrict        dst,
                                        const uint64_t          proc_len_v8)
{
    uint64_t tid = (uint64_t)threadIdx.x + (uint64_t)blockIdx.x * (uint64_t)blockDim.x;
    uint64_t LDG_STG_dex = (uint64_t)threadIdx.x + (uint64_t)blockIdx.x * (uint64_t)blockDim.x;

    uint32_t lane_id = (threadIdx.x & 0x1f);        // local thread index within a warp
    int local_warp_id = threadIdx.x / warpSize;
    int global_warp_id = tid / warpSize;
    int _crit;

    int search_id = blockIdx.x - warpSize + lane_id;

    __shared__ int _warp_aggregates;

    decx::utils::_cuda_vec128 _recv;
    int4 _scan_res[2];
    int tmp, tmp1;

    int warp_lookback_aggregate = 0;

    decx::scan::scan_warp_pred_int32 _previous_status;
    decx::scan::scan_warp_pred_int32 _status;

    if (LDG_STG_dex < proc_len_v8) {
        _recv._vf = src[LDG_STG_dex];
    }

    __syncthreads();

    if (local_warp_id == (blockDim.x / warpSize) - 1 && blockIdx.x != 0)
    {
        for (int i = blockIdx.x - 1; i > -1; --i)
        {
            _previous_status._prefix_sum = 0;
            _previous_status._warp_aggregate = 0;
            _previous_status._warp_status = decx::scan::_scan_warp_status::AGGREGATE_AVAILABLE;
            if (search_id > -1) {
                *((float4*)&_previous_status) = _block_status[search_id];
            }

            _crit = __ballot_sync(0xffffffff, _previous_status._warp_status == decx::scan::_scan_warp_status::PREFIX_AVAILABLE);
            if (!((bool)_crit)) {
                decx::scan::GPUK::cu_warp_inclusive_scan_int32<int(int, int), 32>(decx::utils::cuda::__i32_add, &_previous_status._warp_aggregate, &tmp, lane_id);
                warp_lookback_aggregate = warp_lookback_aggregate + tmp;
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

        decx::scan::GPUK::cu_warp_inclusive_scan_int32<int(int, int), 32>(decx::utils::cuda::__i32_add, &_previous_status._warp_aggregate, &tmp, lane_id);
        warp_lookback_aggregate = warp_lookback_aggregate + tmp;

        __syncwarp(0xffffffff);

        if (lane_id == warpSize - 1) {
            *((float4*)&_status) = _block_status[blockIdx.x];

            if (_is_exclusive) {
                _status._prefix_sum = _recv._arrs[7] + warp_lookback_aggregate + _status._end_value;
            }
            else {
                _status._prefix_sum = _recv._arrs[7] + warp_lookback_aggregate;
            }
            _status._warp_status = decx::scan::_scan_warp_status::PREFIX_AVAILABLE;
            _block_status[blockIdx.x] = *((float4*)&_status);

            _warp_aggregates = warp_lookback_aggregate;
        }
    }

    __syncthreads();

    warp_lookback_aggregate = (blockIdx.x != 0) ? _warp_aggregates : 0;

    _scan_res[0].x = _recv._arrs[0];
    _scan_res[0].y = _recv._arrs[1];
    _scan_res[0].z = _recv._arrs[2];
    _scan_res[0].w = _recv._arrs[3];

    _scan_res[1].x = _recv._arrs[4];
    _scan_res[1].y = _recv._arrs[5];
    _scan_res[1].z = _recv._arrs[6];
    _scan_res[1].w = _recv._arrs[7];

    if (blockIdx.x != 0) {
        _scan_res[0].x = _scan_res[0].x + warp_lookback_aggregate;
        _scan_res[0].y = _scan_res[0].y + warp_lookback_aggregate;
        _scan_res[0].z = _scan_res[0].z + warp_lookback_aggregate;
        _scan_res[0].w = _scan_res[0].w + warp_lookback_aggregate;

        _scan_res[1].x = _scan_res[1].x + warp_lookback_aggregate;
        _scan_res[1].y = _scan_res[1].y + warp_lookback_aggregate;
        _scan_res[1].z = _scan_res[1].z + warp_lookback_aggregate;
        _scan_res[1].w = _scan_res[1].w + warp_lookback_aggregate;
    }

    if (LDG_STG_dex < proc_len_v8) {
        dst[LDG_STG_dex * 2] = _scan_res[0];
        dst[LDG_STG_dex * 2 + 1] = _scan_res[1];
    }
}



template __global__ void decx::scan::GPUK::cu_block_DLB_u16_i32_1D_v8<true>(const float4* __restrict src, float4* _status, int4* dst, const uint64_t proc_len_v8);
template __global__ void decx::scan::GPUK::cu_block_DLB_u16_i32_1D_v8<false>(const float4* __restrict src, float4* _status, int4* dst, const uint64_t proc_len_v8);
