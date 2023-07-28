/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _SCAN_CUH_
#define _SCAN_CUH_


#include "../../../core/basic.h"
#include "../../../core/cudaStream_management/cudaEvent_queue.h"
#include "../../../core/cudaStream_management/cudaStream_queue.h"
#include "scan_kernel_utils.cuh"
#include "scan_mode.h"



#define _WARP_SCAN_BLOCK_SIZE_ 256



namespace decx
{
namespace scan
{
    enum _scan_warp_status
    {
        AGGREGATE_AVAILABLE = 0,
        PREFIX_AVAILABLE = 1
    };


    struct __align__(16) scan_warp_pred_fp32
    {
        float _warp_aggregate;
        float _prefix_sum;
        float _end_value;
        uint _warp_status;

        __device__ __host__ scan_warp_pred_fp32()
        {
            this->_end_value = 0.f;
            this->_warp_aggregate = 0.f;
            this->_prefix_sum = 0.f;
            this->_warp_status = AGGREGATE_AVAILABLE;
        }
    };


    struct __align__(16) scan_warp_pred_int32
    {
        int _warp_aggregate;
        int _prefix_sum;
        int _end_value;
        uint _warp_status;

        __device__ __host__ scan_warp_pred_int32()
        {
            this->_end_value = 0;
            this->_warp_aggregate = 0;
            this->_prefix_sum = 0;
            this->_warp_status = AGGREGATE_AVAILABLE;
        }
    };
}
}


namespace decx
{
namespace scan
{
    namespace GPUK {
        __global__ void cu_block_inclusive_scan_fp32_1D(const float4* src, float4* _status, float4* dst,
            const uint64_t proc_len_v4);


        __global__ void cu_block_exclusive_scan_fp32_1D(const float4* src, float4* _status, float4* dst,
            const uint64_t proc_len_v4);


        __global__ void cu_block_inclusive_scan_u8_u16_1D(const float2* src, float4* _status, int4* dst,
            const uint64_t proc_len_v8);


        __global__ void cu_block_exclusive_scan_u8_fp16_1D(const float2* src, float4* _status, int4* dst,
            const uint64_t proc_len_v8);


        template <bool _is_exclusive>
        __global__ void cu_block_DLB_u16_i32_1D_v8(const float4* __restrict src, float4* __restrict _status, int4* __restrict dst,
            const uint64_t proc_len_v4);


        __global__ void cu_block_inclusive_scan_fp16_1D(const float4* src, float4* _status, float4* dst,
            const uint64_t proc_len_v4);


        __global__ void cu_block_exclusive_scan_fp16_1D(const float4* src, float4* _status, float4* dst,
            const uint64_t proc_len_v4);

        // DLB
        template <bool _is_exclusive>
        __global__ void cu_scan_DLB_fp32_1D(float4* __restrict _status, float4* __restrict dst, const uint64_t proc_len_v4);


        template <bool _is_exclusive>
        __global__ void cu_scan_DLB_fp32_1D_v8(float4* __restrict _status, float4* __restrict dst, const uint64_t proc_len_v4);
    }
}
}


namespace decx
{
    namespace scan {
        namespace GPUK 
        {
// ------------------------------------------------- fp32 ------------------------------------------------------------
            // [32, 8] ([warp_size, 8])
            __global__ void cu_h_block_inclusive_scan_fp32_2D(const float4* src, float4* _status, float4* dst,
                const uint32_t Wmat_v4, const uint32_t Wstatus, const uint2 proc_dim_v4);

            __global__ void cu_h_block_exclusive_scan_fp32_2D(const float4* src, float4* _status, float4* dst,
                const uint32_t Wmat_v4, const uint32_t Wstatus, const uint2 proc_dim_v4);

            template <bool _is_inplace>
            // [32, 8] ([warp_size, 8])
            __global__ void cu_v_block_inclusive_scan_fp32_2D(const float* __restrict src, float4* __restrict _status, float* __restrict dst,
                const uint32_t Wmat, const uint32_t Wstatus, const uint2 proc_dim);

            template <bool _is_inplace>
            // [32, 8] ([warp_size, 8])
            __global__ void cu_v_block_exclusive_scan_fp32_2D(const float* __restrict src, float4* __restrict _status, float* __restrict dst,
                const uint32_t Wmat, const uint32_t Wstatus, const uint2 proc_dim);


            // [32, 32] ([warp_size, 32])
            template <bool _is_exclusive>
            __global__ void cu_v_scan_DLB_fp32_2D(float4* __restrict _status, float* __restrict dst, const uint Wmat_v4,
                const uint Wstatus, const uint2 proc_dim_v4);


            template <bool _is_exclusive>
            __global__ void cu_h_scan_DLB_fp32_2D(float4* __restrict block_status, float4* __restrict dst,
                const uint Wmat_v4, const uint Wstatus, const uint2 proc_dim_v4);


// ------------------------------------------------- fp16 ------------------------------------------------------------

            __global__ void cu_h_block_inclusive_scan_fp16_2D(const float4* src, float4* _status, float4* dst,
                const uint32_t Wsrc_v8, const uint32_t Wdst_v4, const uint32_t Wstatus, const uint2 proc_dim_v4);


            __global__ void cu_h_block_exclusive_scan_fp16_2D(const float4* src, float4* _status, float4* dst,
                const uint32_t Wsrc_v8, const uint32_t Wdst_v4, const uint32_t Wstatus, const uint2 proc_dim_v4);


            __global__ void cu_v_block_inclusive_scan_fp16_2D_v2(const float* src, float4* _status, float2* dst,
                const uint32_t Wsrc_v2, const uint32_t Wdst_v2, const uint32_t Wstatus, const uint2 proc_dim_v2);


            __global__ void cu_v_block_exclusive_scan_fp16_2D_v2(const float* src, float4* _status, float2* dst,
                const uint32_t Wsrc_v2, const uint32_t Wdst_v2, const uint32_t Wstatus, const uint2 proc_dim_v2);


            // [32, 8] ([warp_size, 8])
            template <bool _is_exclusive>
            /*
            * this
            * @param _status        : In-Out buffer, the pointer where scanning status are stored
            * @param dst            : In-Out buffer, the pointer of the output matrix of warp-scan process
            * @param Wmat_v4        : The width of the 'dst' matrix. Measured in scale of 4 elements
            * @param Wstatus        : The width of the matrix where scanning status are stored. Measured in scale of 1 element
            * @param proc_dims_v8   : ~.x -> The width of the process area, measured in scale of 8 elements
            *                         ~.y -> The height og the process area, measured in scale of 1 element
            */
            __global__ void cu_h_scan_DLB_fp32_2D_v8(float4* __restrict _status, float4* __restrict dst, const uint Wmat_v4,
                const uint Wstatus, const uint2 proc_dim_v4);

            // ------------------------------------------------- uint8 ------------------------------------------------------------
            
            /*
            * @param LDG_STG_bounds : ~.x -> height, measured in scale of 1 element
            *                         ~.y -> Wsrc_v16, measured in scale of 16 elements
            *                         ~.z -> Wdst_v8_fp16, measured in scale of 8 elements, (taking fp16 as referance)
            */
            __global__ void cu_h_block_inclusive_scan_u8_u16_2D(const float2* src, float4* _status, float4* dst,
                const uint32_t Wsrc_v16, const uint32_t Wdst_v8_fp16, const uint32_t Wstatus, const uint3 LDG_STG_bounds);


            __global__ void cu_h_block_exclusive_scan_u8_u16_2D(const float2* src, float4* _status, float4* dst,
                const uint32_t Wsrc_v16, const uint32_t Wdst_v8_fp16, const uint32_t Wstatus, const uint3 LDG_STG_bounds);


            // [32, 8] ([warp_size, 8])
            template <bool _is_exclusive>
            __global__ void cu_h_block_DLB_fp16_i32_2D_v8(const float4* __restrict src, float4* __restrict _status, int4* __restrict dst, const uint Wmat_v4_i32,
                const uint Wmat_v8_fp16, const uint Wstatus, const uint2 proc_dim_v4);


            // [32, 32] ([warp_size, 32])
            template <bool _is_exclusive>
            __global__ void cu_v_scan_DLB_int32_2D(float4* __restrict _status, int* __restrict dst, const uint Wmat_v4,
                const uint Wstatus, const uint2 proc_dim_v4);


            // [32, 8] ([warp_size, 8])
            __global__ void cu_v_block_inclusive_scan_int32_2D(float4* _status, int* dst,
                const uint32_t Wmat, const uint32_t Wstatus, const uint2 proc_dim);


            __global__ void cu_v_block_exclusive_scan_int32_2D(float4* _status, int* dst,
                const uint32_t Wmat, const uint32_t Wstatus, const uint2 proc_dim);


            __global__ void cu_v_block_inclusive_scan_u8_u16_2D_v4(const float* src, float4* _status, double* dst,
                const uint32_t Wsrc_v2, const uint32_t Wdst_v2, const uint32_t Wstatus, const uint2 proc_dim_v2);

            __global__ void cu_v_block_exclusive_scan_u8_u16_2D_v4(const float* src, float4* _status, double* dst,
                const uint32_t Wsrc_v2, const uint32_t Wdst_v2, const uint32_t Wstatus, const uint2 proc_dim_v2);

            // [32, 32] ([warp_size, 32])
            template <bool _is_exclusive>
            __global__ void cu_v_scan_DLB_u16_i32_2D(const ushort* __restrict src, float4* __restrict _status, int* __restrict dst, 
                const uint Wsrc, const uint Wdst, const uint Wstatus, const uint2 proc_dim_v4);
        }
    }
}





#endif