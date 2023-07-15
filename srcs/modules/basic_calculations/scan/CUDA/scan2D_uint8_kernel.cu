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
decx::scan::GPUK::cu_h_warp_inclusive_scan_u8_u16_2D(const float2* __restrict     src, 
                                                   float4* __restrict           warp_status, 
                                                   float4* __restrict             dst,
                                                   const uint32_t               Wsrc_v8,
                                                   const uint32_t               Wdst_v8_fp16,
                                                   const uint32_t               Wstatus,
                                                   const uint3                  LDG_STG_bounds)
{
    // in-block (local) warp_id = threadIdx.y
    uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;      // H
    uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;      // V
    // in-warp (local) lane_id = threadIdx.x % 32

    uint64_t LDG_dex = (uint64_t)tidx + (uint64_t)tidy * (uint64_t)Wsrc_v8;
    uint64_t STG_dex = (uint64_t)tidx + (uint64_t)tidy * (uint64_t)Wdst_v8_fp16;
    uint64_t STG_status_dex = (uint64_t)blockIdx.x + (uint64_t)tidy * Wstatus;

    decx::utils::_cuda_vec128 _thread_scan_res;
    ushort tmp, tmp1;

    decx::scan::scan_warp_pred_int32 _status;

    decx::utils::_cuda_vec64 _recv;
    
    if (tidy < LDG_STG_bounds.x && tidx < LDG_STG_bounds.y) {
        _recv._vf2 = src[LDG_dex];
    }

    decx::scan::GPUK::_inclusive_scan_uc8_u16(_recv, (ushort*)&_thread_scan_res);

    decx::scan::GPUK::cu_warp_exclusive_scan_u16<ushort(ushort, ushort), 32>(decx::utils::cuda::__u16_add, &_thread_scan_res._arrs[7], &tmp, threadIdx.x);
    _thread_scan_res._vi = decx::utils::add_scalar_vec4(_thread_scan_res, tmp);

    if (threadIdx.x == blockDim.x - 1) 
    {
        _status._warp_aggregate = _thread_scan_res._arrs[7];
        if (blockIdx.x == 0)
        {
            _status._warp_status = decx::scan::_scan_warp_status::PREFIX_AVAILABLE;
            _status._prefix_sum = _thread_scan_res._arrs[7];
        }
        else {
            _status._warp_status = decx::scan::_scan_warp_status::AGGREGATE_AVAILABLE;
        }
        warp_status[STG_status_dex] = *((float4*)&_status);
    }

    if (tidy < LDG_STG_bounds.x && tidx < LDG_STG_bounds.z) {
        dst[STG_dex] = _thread_scan_res._vf;
    }
}


__global__ void 
decx::scan::GPUK::cu_h_warp_exclusive_scan_u8_u16_2D(const float2* __restrict     src, 
                                                   float4* __restrict           warp_status, 
                                                   float4* __restrict             dst,
                                                   const uint32_t               Wsrc_v8,
                                                   const uint32_t               Wdst_v8_fp16,
                                                   const uint32_t               Wstatus,
                                                   const uint3                  LDG_STG_bounds)
{
    // in-block (local) warp_id = threadIdx.y
    uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;      // H
    uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;      // V
    // in-warp (local) lane_id = threadIdx.x % 32

    uint64_t LDG_dex = (uint64_t)tidx + (uint64_t)tidy * (uint64_t)Wsrc_v8;
    uint64_t STG_dex = (uint64_t)tidx + (uint64_t)tidy * (uint64_t)Wdst_v8_fp16;
    uint64_t STG_status_dex = (uint64_t)blockIdx.x + (uint64_t)tidy * Wstatus;

    decx::utils::_cuda_vec128 _thread_scan_res;
    ushort tmp, tmp1;

    decx::scan::scan_warp_pred_int32 _status;
    decx::utils::_cuda_vec64 _recv;
    
    if (tidy < LDG_STG_bounds.x && tidx < LDG_STG_bounds.y) {
        _recv._vf2 = src[LDG_dex];
    }

    decx::scan::GPUK::_exclusive_scan_uc8_u16(_recv, (ushort*)&_thread_scan_res);

    tmp1 = _thread_scan_res._arrs[7] + _recv._v_uint8[7];
    decx::scan::GPUK::cu_warp_exclusive_scan_u16<ushort(ushort, ushort), 32>(decx::utils::cuda::__u16_add, &tmp1, &tmp, threadIdx.x);
    _thread_scan_res._vi = decx::utils::add_scalar_vec4(_thread_scan_res, tmp);

    if (threadIdx.x == blockDim.x - 1) 
    {
        _status._warp_aggregate = _thread_scan_res._arrs[7] + _recv._v_uint8[7];
        _status._end_value = _recv._v_uint8[7];

        if (blockIdx.x == 0)
        {
            _status._warp_status = decx::scan::_scan_warp_status::PREFIX_AVAILABLE;
            _status._prefix_sum = _status._warp_aggregate;
        }
        else {
            _status._warp_status = decx::scan::_scan_warp_status::AGGREGATE_AVAILABLE;
        }
        warp_status[STG_status_dex] = *((float4*)&_status);
    }

    if (tidy < LDG_STG_bounds.x && tidx < LDG_STG_bounds.z) {
        dst[STG_dex] = _thread_scan_res._vf;
    }
}


__global__ void 
decx::scan::GPUK::cu_v_warp_inclusive_scan_u8_u16_2D_v4(const float* __restrict      src,
                                                      float4* __restrict           warp_status, 
                                                      double* __restrict           dst,
                                                      const uint32_t               Wsrc_v4,
                                                      const uint32_t               Wdst_v4,
                                                      const uint32_t               Wstatus_TP,
                                                      const uint2                  proc_dim_v4)
{
    // in-block (local) warp_id = threadIdx.y
    uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;      // H
    uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;      // V

    uint64_t LDG_dex = (uint64_t)tidx + (uint64_t)tidy * 4 * (uint64_t)Wsrc_v4;
    uint64_t STG_dex = (uint64_t)tidx + (uint64_t)tidy * 4 * (uint64_t)Wdst_v4;
    uint64_t STG_status_dex = tidx * 4 * Wstatus_TP + blockIdx.y;

    decx::utils::_cuda_vec128 _recv;
    double4 _thread_scan_res;
    int2 tmp1, tmp2;
    decx::scan::scan_warp_pred_int32 _status;
    
    __shared__ double _work_space[8][32 + 1];

    if (tidx < proc_dim_v4.x) {
        if (tidy * 4 < proc_dim_v4.y)  _recv._vf.x = src[LDG_dex];
        if (tidy * 4 + 1 < proc_dim_v4.y)  _recv._vf.y = src[LDG_dex + Wsrc_v4];
        if (tidy * 4 + 2 < proc_dim_v4.y)  _recv._vf.z = src[LDG_dex + Wsrc_v4 * 2];
        if (tidy * 4 + 3 < proc_dim_v4.y)  _recv._vf.w = src[LDG_dex + Wsrc_v4 * 3];

        decx::scan::GPUK::_inclusive_scan_u8_4way(_recv, (int*)&_thread_scan_res);

        _work_space[threadIdx.y][threadIdx.x] = _thread_scan_res.w;        // The last ushort4
    }

    __syncthreads();

#pragma unroll 3
    for (int i = 0; i < 3; ++i) 
    {
        *((double*)&tmp1) = _work_space[threadIdx.y][threadIdx.x];

        if (threadIdx.y > (1 << i) - 1) {
            *((double*)&tmp2) = _work_space[threadIdx.y - (1 << i)][threadIdx.x];
            tmp1.x = __vadd2(tmp1.x, tmp2.x);
            tmp1.y = __vadd2(tmp1.y, tmp2.y);
        }
        __syncthreads();
        if (i < 2) {
            _work_space[threadIdx.y][threadIdx.x] = *((double*)&tmp1);
        }
        else {
            *((int2*)&_work_space[threadIdx.y][threadIdx.x]) = make_int2(__vsub2(tmp1.x, ((int2*)&_thread_scan_res.w)->x),
                                                                         __vsub2(tmp1.y, ((int2*)&_thread_scan_res.w)->y));
        }
        __syncthreads();
    }

    // get aggregate for each float4
    *((double*)&tmp1) = _work_space[threadIdx.y][threadIdx.x];

    if (threadIdx.y == blockDim.y - 1)
    {
        tmp2.x = __vadd2(tmp1.x, ((int2*)&_thread_scan_res.w)->x);
        tmp2.y = __vadd2(tmp1.y, ((int2*)&_thread_scan_res.w)->y);
        if (blockIdx.y == 0) {
            _status._prefix_sum = ((ushort2*)&tmp2.x)->x;
            _status._warp_status = decx::scan::_scan_warp_status::PREFIX_AVAILABLE;
            warp_status[STG_status_dex] = *((float4*)&_status);

#pragma unroll 3
            for (int i = 1; i < 4; ++i) {
                _status._prefix_sum = ((ushort*)&tmp2)[i];
                warp_status[STG_status_dex + Wstatus_TP * i] = *((float4*)&_status);
            }
        }
        else {
            _status._warp_aggregate = ((ushort2*)&tmp2.x)->x;
            _status._warp_status = decx::scan::_scan_warp_status::AGGREGATE_AVAILABLE;
            warp_status[STG_status_dex] = *((float4*)&_status);

#pragma unroll 3
            for (int i = 1; i < 4; ++i) {
                _status._warp_aggregate = ((ushort*)&tmp2)[i];
                warp_status[STG_status_dex + Wstatus_TP * i] = *((float4*)&_status);
            }
        }
    }

    if (tidx < proc_dim_v4.x) 
    {
        _thread_scan_res = decx::utils::add_scalar_vec4(_thread_scan_res, tmp1);

        if (tidy * 4 < proc_dim_v4.y) {
            dst[STG_dex] = _thread_scan_res.x;
        }
        if (tidy * 4 + 1 < proc_dim_v4.y) { 
            dst[STG_dex + Wdst_v4] = _thread_scan_res.y;
        }
        if (tidy * 4 + 2 < proc_dim_v4.y) { 
            dst[STG_dex + Wdst_v4 * 2] = _thread_scan_res.z;
        }
        if (tidy * 4 + 3 < proc_dim_v4.y) {
            dst[STG_dex + Wdst_v4 * 3] = _thread_scan_res.w;
        }
    }
}


// u16_i32
template <bool _is_exclusive>
__global__ void 
decx::scan::GPUK::cu_h_scan_DLB_fp16_i32_2D_v8(const float4* __restrict src,
                                               float4* __restrict     block_status, 
                                               int4* __restrict       dst, 
                                               const uint             Wmat_v4_i32,
                                               const uint             Wmat_v8_fp16,
                                               const uint             Wstatus, 
                                               const uint2            proc_dim_v8)
{
    // in-block (local) warp_id = threadIdx.y
    uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;      // H
    uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;      // V
    // in-warp (local) lane_id = threadIdx.x % 32
    uint32_t lane_id = threadIdx.x;        // local thread index within a warp

    uint64_t LDG_dex = (uint64_t)tidx + (uint64_t)tidy * (uint64_t)Wmat_v8_fp16;
    uint64_t STG_dex = (uint64_t)tidx * 2 + (uint64_t)tidy * (uint64_t)Wmat_v4_i32;
    uint64_t STG_status_dex = (uint64_t)blockIdx.x + (uint64_t)tidy * (uint64_t)Wstatus;

    int search_id_x = blockIdx.x - warpSize + lane_id;

    int _crit;
    decx::utils::_cuda_vec128 _recv;
    int4 regs[2];
    int warp_lookback_aggregate = 0;

    __shared__ int _warp_aggregates[8];

    decx::scan::scan_warp_pred_int32 _previous_status;
    decx::scan::scan_warp_pred_int32 _status;

    if (tidx < proc_dim_v8.x && tidy < proc_dim_v8.y) {
        _recv._vf = src[LDG_dex];
    }

    __syncthreads();

    if (blockIdx.x != 0) 
    {
        for (int i = blockIdx.x - 1; i > -1; --i)
        {
            _previous_status._prefix_sum = 0;
            _previous_status._warp_aggregate = 0;
            _previous_status._warp_status = decx::scan::_scan_warp_status::AGGREGATE_AVAILABLE;
            if (search_id_x > -1) {
                *((float4*)&_previous_status) = block_status[search_id_x + (uint64_t)tidy * (uint64_t)Wstatus];
            }

            _crit = __ballot_sync(0xffffffff, _previous_status._warp_status == decx::scan::_scan_warp_status::PREFIX_AVAILABLE);

            if (!((bool)_crit)) {
                decx::scan::GPUK::cu_warp_inclusive_scan_int32<int(int, int), 32>(decx::utils::cuda::__i32_add, &_previous_status._warp_aggregate, &regs[0].x, lane_id);
                warp_lookback_aggregate = warp_lookback_aggregate + regs[0].x;
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

        decx::scan::GPUK::cu_warp_inclusive_scan_int32<int(int, int), 32>(decx::utils::cuda::__i32_add, &_previous_status._warp_aggregate, &regs[0].x, lane_id);
        warp_lookback_aggregate = warp_lookback_aggregate + regs[0].x;

        __syncwarp(0xffffffff);

        if (lane_id == warpSize - 1) {
            *((float4*)&_status) = block_status[STG_status_dex];

            if (_is_exclusive) {
                _status._prefix_sum = __half2int_rn(_recv._arrh[7]) + warp_lookback_aggregate + _status._end_value;
            }
            else {
                _status._prefix_sum = __half2int_rn(_recv._arrh[7]) + warp_lookback_aggregate;
            }
            _status._warp_status = decx::scan::_scan_warp_status::PREFIX_AVAILABLE;
            block_status[STG_status_dex] = *((float4*)&_status);

            _warp_aggregates[threadIdx.y] = warp_lookback_aggregate;
        }
    }

    __syncthreads();

    regs[0].x = _recv._arrs[0];
    regs[0].y = _recv._arrs[1];
    regs[0].z = _recv._arrs[2];
    regs[0].w = _recv._arrs[3];

    regs[1].x = _recv._arrs[4];
    regs[1].y = _recv._arrs[5];
    regs[1].z = _recv._arrs[6];
    regs[1].w = _recv._arrs[7];

    if (blockIdx.x != 0) {
        warp_lookback_aggregate = _warp_aggregates[threadIdx.y];
        regs[0].x = regs[0].x + warp_lookback_aggregate;
        regs[0].y = regs[0].y + warp_lookback_aggregate;
        regs[0].z = regs[0].z + warp_lookback_aggregate;
        regs[0].w = regs[0].w + warp_lookback_aggregate;

        regs[1].x = regs[1].x + warp_lookback_aggregate;
        regs[1].y = regs[1].y + warp_lookback_aggregate;
        regs[1].z = regs[1].z + warp_lookback_aggregate;
        regs[1].w = regs[1].w + warp_lookback_aggregate;
    }

    if (tidx < proc_dim_v8.x && tidy < proc_dim_v8.y) {
        dst[STG_dex] = regs[0];
        dst[STG_dex + 1] = regs[1];
    }
}


template __global__ void decx::scan::GPUK::cu_h_scan_DLB_fp16_i32_2D_v8<true>(const float4* __restrict src, float4* __restrict _status, int4* __restrict dst, const uint Wmat_v4_i32,
    const uint Wmat_v8_fp16, const uint Wstatus, const uint2 proc_dim_v4);

template __global__ void decx::scan::GPUK::cu_h_scan_DLB_fp16_i32_2D_v8<false>(const float4* __restrict src, float4* __restrict _status, int4* __restrict dst, const uint Wmat_v4_i32,
    const uint Wmat_v8_fp16, const uint Wstatus, const uint2 proc_dim_v4);


__global__ void 
decx::scan::GPUK::cu_v_warp_inclusive_scan_int32_2D(float4* __restrict           warp_status, 
                                                   int* __restrict              dst,
                                                   const uint32_t               Wmat,
                                                   const uint32_t               Wstatus_TP,
                                                   const uint2                  proc_dim)
{
    // in-block (local) warp_id = threadIdx.y
    uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;      // H
    uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;      // V

    uint64_t index = (uint64_t)tidx + (uint64_t)tidy * 4 * (uint64_t)Wmat;
    uint64_t STG_status_dex = tidx * Wstatus_TP + blockIdx.y;

    int4 _recv;
    int tmp1, tmp2;
    decx::scan::scan_warp_pred_int32 _status;
    
    __shared__ int _work_space[8][32 + 1];

    if (tidx < proc_dim.x) {
        if (tidy * 4 < proc_dim.y)  _recv.x = dst[index];
        if (tidy * 4 + 1 < proc_dim.y)  _recv.y = dst[index + Wmat];
        if (tidy * 4 + 2 < proc_dim.y)  _recv.z = dst[index + Wmat * 2];
        if (tidy * 4 + 3 < proc_dim.y)  _recv.w = dst[index + Wmat * 3];

        _recv = decx::scan::GPUK::_inclusive_scan_int4(_recv);

        _work_space[threadIdx.y][threadIdx.x] = _recv.w;
    }

    __syncthreads();

#pragma unroll 3
    for (int i = 0; i < 3; ++i) 
    {
        tmp1 = _work_space[threadIdx.y][threadIdx.x];

        if (threadIdx.y > (1 << i) - 1) {
            tmp2 = _work_space[threadIdx.y - (1 << i)][threadIdx.x];
            tmp1 = tmp1 + tmp2;
        }
        __syncthreads();
        if (i < 2) {
            _work_space[threadIdx.y][threadIdx.x] = tmp1;
        }
        else {
            _work_space[threadIdx.y][threadIdx.x] = tmp1 - _recv.w;
        }
        __syncthreads();
    }

    // get aggregate for each float4
    tmp1 = _work_space[threadIdx.y][threadIdx.x];

    if (threadIdx.y == blockDim.y - 1/* && tidy < proc_dim.y*/)
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



__global__ void 
decx::scan::GPUK::cu_v_warp_exclusive_scan_int32_2D(float4* __restrict           warp_status, 
                                                   int* __restrict              dst,
                                                   const uint32_t               Wmat,
                                                   const uint32_t               Wstatus_TP,
                                                   const uint2                  proc_dim)
{
    // in-block (local) warp_id = threadIdx.y
    uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;      // H
    uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;      // V

    uint64_t index = (uint64_t)tidx + (uint64_t)tidy * 4 * (uint64_t)Wmat;
    uint64_t STG_status_dex = tidx * Wstatus_TP + blockIdx.y;

    int4 _recv;
    int tmp1, tmp2, _end_value;
    decx::scan::scan_warp_pred_int32 _status;
    
    __shared__ int _work_space[8][32 + 1];

    if (tidx < proc_dim.x) {
        if (tidy * 4 < proc_dim.y)  _recv.x = dst[index];
        if (tidy * 4 + 1 < proc_dim.y)  _recv.y = dst[index + Wmat];
        if (tidy * 4 + 2 < proc_dim.y)  _recv.z = dst[index + Wmat * 2];
        if (tidy * 4 + 3 < proc_dim.y)  _recv.w = dst[index + Wmat * 3];

        _end_value = _recv.w;

        _recv = decx::scan::GPUK::_exclusive_scan_int4(_recv);
        
        _work_space[threadIdx.y][threadIdx.x] = _recv.w + _end_value;
    }

    __syncthreads();

#pragma unroll 3
    for (int i = 0; i < 3; ++i) 
    {
        tmp1 = _work_space[threadIdx.y][threadIdx.x];

        if (threadIdx.y > (1 << i) - 1) {
            tmp2 = _work_space[threadIdx.y - (1 << i)][threadIdx.x];
            tmp1 = tmp1 + tmp2;
        }
        __syncthreads();
        if (i < 2) {
            _work_space[threadIdx.y][threadIdx.x] = tmp1;
        }
        else {
            _work_space[threadIdx.y][threadIdx.x] = tmp1 - _recv.w - _end_value;
        }
        __syncthreads();
    }

    // get aggregate for each float4
    tmp1 = _work_space[threadIdx.y][threadIdx.x];

    if (threadIdx.y == blockDim.y - 1)
    {
        _status._end_value = _end_value;
        if (blockIdx.y == 0) {
            _status._prefix_sum = tmp1 + _recv.w + _end_value;
            _status._warp_status = decx::scan::_scan_warp_status::PREFIX_AVAILABLE;
            warp_status[STG_status_dex] = *((float4*)&_status);
        }
        else {
            _status._warp_aggregate = tmp1 + _recv.w + _end_value;
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




template <bool _is_exclusive>
__global__ void 
decx::scan::GPUK::cu_v_scan_DLB_int32_2D(float4* __restrict      _warp_status, 
                                        int* __restrict       dst, 
                                        const uint              Wmat,
                                        const uint              Wstatus_TP, 
                                        const uint2             proc_dim)
{
    // in-block (local) warp_id = threadIdx.y
    uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;      // H
    uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;      // V

    uint64_t STG_LDG_dex = (uint64_t)tidx + (uint64_t)tidy * (uint64_t)Wmat;
    
    __shared__ int end_values[32];
    __shared__ int prefix_sums[32];

    int _recv;
    int tmp, warp_lookback_aggregate = 0;
    decx::scan::scan_warp_pred_int32 _previous_status, _status;

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
                decx::scan::GPUK::cu_warp_inclusive_scan_int32<int(int, int), 32>(decx::utils::cuda::__i32_add, &_previous_status._warp_aggregate, &tmp, threadIdx.x);
                warp_lookback_aggregate = warp_lookback_aggregate + tmp;
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

        decx::scan::GPUK::cu_warp_inclusive_scan_int32<int(int, int), 32>(decx::utils::cuda::__i32_add, &_previous_status._warp_aggregate, &tmp, threadIdx.x);
        warp_lookback_aggregate = warp_lookback_aggregate + tmp;

        __syncwarp(0xffffffff);

        if (threadIdx.x == warpSize - 1) {
            *((float4*)&_status) = _warp_status[base + blockIdx.y];

            prefix_sums[threadIdx.y] = warp_lookback_aggregate;

            if (_is_exclusive) {
                _status._prefix_sum = end_values[threadIdx.y] + warp_lookback_aggregate + _status._end_value;
            }
            else {
                _status._prefix_sum = end_values[threadIdx.y] + warp_lookback_aggregate;
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



template __global__ void decx::scan::GPUK::cu_v_scan_DLB_int32_2D<true>(float4* __restrict _status, int* __restrict dst, const uint Wmat_v4,
    const uint Wstatus, const uint2 proc_dim_v4);

template __global__ void decx::scan::GPUK::cu_v_scan_DLB_int32_2D<false>(float4* __restrict _status, int* __restrict dst, const uint Wmat_v4,
    const uint Wstatus, const uint2 proc_dim_v4);



template <bool _is_exclusive>
__global__ void 
decx::scan::GPUK::cu_v_scan_DLB_u16_i32_2D(const ushort* __restrict src,
                                        float4* __restrict      _warp_status, 
                                        int* __restrict         dst, 
                                        const uint              Wdst,
                                        const uint              Wsrc,
                                        const uint              Wstatus_TP, 
                                        const uint2             proc_dim)
{
    // in-block (local) warp_id = threadIdx.y
    uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;      // H
    uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;      // V

    uint64_t LDG_dex = (uint64_t)tidx + (uint64_t)tidy * (uint64_t)Wsrc;
    uint64_t STG_dex = (uint64_t)tidx + (uint64_t)tidy * (uint64_t)Wdst;
    
    __shared__ ushort end_values[32];
    __shared__ ushort prefix_sums[32];

    ushort _recv;
    int tmp, warp_lookback_aggregate = 0;
    decx::scan::scan_warp_pred_int32 _previous_status, _status;

    int _crit;

    if (tidx < proc_dim.x && tidy < proc_dim.y) {
        _recv = src[LDG_dex];
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
                decx::scan::GPUK::cu_warp_inclusive_scan_int32<int(int, int), 32>(decx::utils::cuda::__i32_add, &_previous_status._warp_aggregate, &tmp, threadIdx.x);
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

        decx::scan::GPUK::cu_warp_inclusive_scan_int32<int(int, int), 32>(decx::utils::cuda::__i32_add, &_previous_status._warp_aggregate, &tmp, threadIdx.x);
        warp_lookback_aggregate = __fadd_rn(warp_lookback_aggregate, tmp);

        __syncwarp(0xffffffff);

        if (threadIdx.x == warpSize - 1) {
            *((float4*)&_status) = _warp_status[base + blockIdx.y];

            prefix_sums[threadIdx.y] = warp_lookback_aggregate;

            if (_is_exclusive) {
                _status._prefix_sum = end_values[threadIdx.y] + warp_lookback_aggregate + _status._end_value;
            }
            else {
                _status._prefix_sum = end_values[threadIdx.y] + warp_lookback_aggregate;
            }
            _status._warp_status = decx::scan::_scan_warp_status::PREFIX_AVAILABLE;

            _warp_status[base + blockIdx.y] = *((float4*)&_status);
        }

        __syncthreads();

        tmp = prefix_sums[threadIdx.x];
    }

    if (blockIdx.y != 0) {
        tmp = _recv + tmp;
    }
    else {
        tmp = _recv;
    }

    if (tidx < proc_dim.x && tidy < proc_dim.y) {
        dst[STG_dex] = tmp;
    }
}



template __global__ void decx::scan::GPUK::cu_v_scan_DLB_u16_i32_2D<true>(const ushort* __restrict src, float4* __restrict _status, int* __restrict dst,
    const uint Wsrc, const uint Wdst, const uint Wstatus, const uint2 proc_dim_v4);

template __global__ void decx::scan::GPUK::cu_v_scan_DLB_u16_i32_2D<false>(const ushort* __restrict src, float4* __restrict _status, int* __restrict dst,
    const uint Wsrc, const uint Wdst, const uint Wstatus, const uint2 proc_dim_v4);