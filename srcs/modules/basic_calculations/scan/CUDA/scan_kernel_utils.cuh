/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _SCAN_KERNEL_UTILS_CUH_
#define _SCAN_KERNEL_UTILS_CUH_


#include "../../../core/basic.h"
#include "../../../core/utils/decx_cuda_vectypes_ops.cuh"
#include "../../../core/utils/cuda_int32_math_functions.cuh"


namespace decx
{
namespace scan {
    namespace GPUK {
        __device__ __inline__ float4 _exclusive_scan_float4(const float4 _in)
        {
            float4 res;
            res.x = 0;
            res.y = _in.x;
            res.z = __fadd_rn(_in.x, _in.y);
            res.w = __fadd_rn(res.z, _in.z);

            return res;
        }


        __device__ __inline__ void _exclusive_scan_uc8(const decx::utils::_cuda_vec64& _in, int* res)
        {
            ((int4*)res)[0] = decx::utils::vec4_set1_int32(0);
            ((int4*)res)[1] = decx::utils::vec4_set1_int32(0);

            res[1] = _in._v_uint8[0];

#pragma unroll 6
            for (int i = 2; i < 8; ++i) {
                res[i] = res[i - 1] + _in._v_uint8[i - 1];
            }
        }


        __device__ __inline__ float4 _inclusive_scan_float4(const float4 _in)
        {
            float4 res;
            res.x = _in.x;
            res.y = __fadd_rn(res.x, _in.y);
            res.z = __fadd_rn(res.y, _in.z);
            res.w = __fadd_rn(res.z, _in.w);

            return res;
        }


        __device__ __inline__ int4 _inclusive_scan_int4(const int4 _in)
        {
            int4 res;
            res.x = _in.x;
            res.y = res.x + _in.y;
            res.z = res.y + _in.z;
            res.w = res.z + _in.w;

            return res;
        }


        __device__ __inline__ int4 _exclusive_scan_int4(const int4 _in)
        {
            int4 res;
            res.x = 0;
            res.y = _in.x;
            res.z = res.y + _in.y;
            res.w = res.z + _in.z;

            return res;
        }


        /*
        * Inplace operation is supported
        */
        __device__ __inline__ void _inclusive_scan_half8(const decx::utils::_cuda_vec128& _in, decx::utils::_cuda_vec128* res)
        {
#if __ABOVE_SM_53
            res->_vf = _in._vf;

#pragma unroll 7
            for (int i = 1; i < 8; ++i) {
                res->_arrh[i] = __hadd(res->_arrh[i - 1], _in._arrh[i]);
            }
#endif
        }


        __device__ __inline__ float4 _inclusive_scan_half4_2way(const decx::utils::_cuda_vec128& _in)
        {
#if __ABOVE_SM_53
            decx::utils::_cuda_vec128 res;

            res._arrh2[0] = _in._arrh2[0];
            res._arrh2[1] = __hadd2(res._arrh2[0], _in._arrh2[1]);
            res._arrh2[2] = __hadd2(res._arrh2[1], _in._arrh2[2]);
            res._arrh2[3] = __hadd2(res._arrh2[2], _in._arrh2[3]);

            return res._vf;
#endif
        }


        __device__ __inline__ float4 _exclusive_scan_half4_2way(const decx::utils::_cuda_vec128& _in)
        {
#if __ABOVE_SM_53
            decx::utils::_cuda_vec128 res;

            res._vi.x = 0x00000000;
            res._arrh2[1] = _in._arrh2[0];
            res._arrh2[2] = __hadd2(res._arrh2[1], _in._arrh2[1]);
            res._arrh2[3] = __hadd2(res._arrh2[2], _in._arrh2[2]);

            return res._vf;
#endif
        }

        // _res -> int[8]
        __device__ __inline__ void _inclusive_scan_u8_4way(const decx::utils::_cuda_vec128& _in, int* _res)
        {
            int2 _tmp;
            _tmp.x = __byte_perm(_in._vi.x, 0, 0x5140);         _tmp.y = __byte_perm(_in._vi.x, 0, 0x5342);
            *((int2*)&_res[0]) = _tmp;

            _tmp.x = __byte_perm(_in._vi.y, 0, 0x5140);         _tmp.y = __byte_perm(_in._vi.y, 0, 0x5342);
            _res[2] = __vadd2(_tmp.x, _res[0]);                 _res[3] = __vadd2(_tmp.y, _res[1]);

            _tmp.x = __byte_perm(_in._vi.z, 0, 0x5140);         _tmp.y = __byte_perm(_in._vi.z, 0, 0x5342);
            _res[4] = __vadd2(_tmp.x, _res[2]);                 _res[5] = __vadd2(_tmp.y, _res[3]);

            _tmp.x = __byte_perm(_in._vi.w, 0, 0x5140);         _tmp.y = __byte_perm(_in._vi.w, 0, 0x5342);
            _res[6] = __vadd2(_tmp.x, _res[4]);                 _res[7] = __vadd2(_tmp.y, _res[5]);
        }


        __device__ __inline__ void _exclusive_scan_half8(const decx::utils::_cuda_vec128& _in, decx::utils::_cuda_vec128* res)
        {
#if __ABOVE_SM_53
            res->_arrh[0] = __float2half(0);
            res->_arrh[1] = _in._arrh[1];

#pragma unroll 6
            for (int i = 2; i < 8; ++i) {
                res->_arrh[i] = __hadd(res->_arrh[i - 1], _in._arrh[i]);
            }
#endif
        }


        __device__ __inline__ void _exclusive_scan_half8_inp(decx::utils::_cuda_vec128* _in)
        {
#if __ABOVE_SM_53
            __half tmp = _in->_arrh[1], tmp1;
            _in->_arrh[1] = _in->_arrh[0];
            _in->_arrh[0] = __float2half(0);

            tmp1 = _in->_arrh[2];       _in->_arrh[2] = __hadd(_in->_arrh[1], tmp);
            tmp = _in->_arrh[3];        _in->_arrh[3] = __hadd(_in->_arrh[2], tmp1);
            tmp1 = _in->_arrh[4];       _in->_arrh[4] = __hadd(_in->_arrh[3], tmp);
            tmp = _in->_arrh[5];        _in->_arrh[5] = __hadd(_in->_arrh[4], tmp1);
            tmp1 = _in->_arrh[6];       _in->_arrh[6] = __hadd(_in->_arrh[5], tmp);
            tmp1 = _in->_arrh[7];       _in->_arrh[7] = __hadd(_in->_arrh[6], tmp);
#endif
        }



        __device__ __inline__ void _inclusive_scan_uc8(const decx::utils::_cuda_vec64& _in, int* res)
        {
            ((int4*)res)[0] = decx::utils::vec4_set1_int32(0);
            ((int4*)res)[1] = decx::utils::vec4_set1_int32(0);

            res[0] = _in._v_uint8[0];

#pragma unroll 7
            for (int i = 1; i < 8; ++i) {
                res[i] = res[i - 1] + _in._v_uint8[i];
            }
        }


        __device__ __inline__ void _inclusive_scan_uc8_u16(const decx::utils::_cuda_vec64& _in, ushort* res)
        {
            ((int4*)res)[0] = decx::utils::vec4_set1_int32(0);

            res[0] = _in._v_uint8[0];

#pragma unroll 7
            for (int i = 1; i < 8; ++i) {
                res[i] = res[i - 1] + _in._v_uint8[i];
            }
        }


        __device__ __inline__ void _exclusive_scan_uc8_u16(const decx::utils::_cuda_vec64& _in, ushort* res)
        {
            // set every bit in the result registers to zero first
            ((int4*)res)[0] = decx::utils::vec4_set1_int32(0);

            res[1] = _in._v_uint8[0];

#pragma unroll 6
            for (int i = 2; i < 8; ++i) {
                res[i] = res[i - 1] + _in._v_uint8[i - 1];
            }
        }
    }
}
}


namespace decx
{
namespace scan
{
    namespace GPUK
    {
        template <typename TypeOP, uint32_t scan_length>
        __device__ __inline__ static void cu_warp_exclusive_scan_fp16(TypeOP& _op,
            const __half* _src, __half* _dst, const uint32_t lane_id)
        {
#if __ABOVE_SM_53
            float _accu = *_src, tmp;
#pragma unroll
            for (int i = 1; i < scan_length; i *= 2) {
                tmp = __shfl_up_sync(0xffffffff, _accu, i, warpSize);

                _accu = lane_id > (i - 1) ? (_op(_accu, tmp)) : _accu;
            }
            tmp = __shfl_up_sync(0xffffffff, _accu, 1, warpSize);
            tmp = (lane_id == 0) ? __float2half(0) : tmp;
            *_dst = tmp;
#endif
        }


        template <typename TypeOP, uint32_t scan_length>
        __device__ __inline__ static void cu_warp_exclusive_scan_u16(TypeOP& _op,
            const ushort* _src, ushort* _dst, const uint32_t lane_id)
        {
            float _accu = *_src, tmp;
#pragma unroll
            for (int i = 1; i < scan_length; i *= 2) {
                tmp = __shfl_up_sync(0xffffffff, _accu, i, warpSize);

                _accu = lane_id > (i - 1) ? (_op(_accu, tmp)) : _accu;
            }
            tmp = __shfl_up_sync(0xffffffff, _accu, 1, warpSize);
            tmp = (lane_id == 0) ? 0 : tmp;
            *_dst = tmp;
        }


        template <typename TypeOP, uint32_t scan_length>
        __device__ __inline__ static void cu_warp_inclusive_scan_fp16(TypeOP& _op,
            const __half* _src, __half* _dst, const uint32_t lane_id)
        {
#if __ABOVE_SM_53
            float _accu = *_src, tmp;
#pragma unroll
            for (int i = 1; i < scan_length; i *= 2) {
                tmp = __shfl_up_sync(0xffffffff, _accu, i, warpSize);

                _accu = lane_id > (i - 1) ? (_op(_accu, tmp)) : _accu;
            }
            *_dst = _accu;
#endif
        }


        template <typename TypeOP, uint32_t scan_length>
        __device__ __inline__ static void cu_warp_exclusive_scan_fp32(TypeOP& _op,
            const float* _src, float* _dst, const uint32_t lane_id)
        {
            float _accu = *_src, tmp;
#pragma unroll
            for (int i = 1; i < scan_length; i *= 2) {
                tmp = __shfl_up_sync(0xffffffff, _accu, i, warpSize);

                _accu = lane_id > (i - 1) ? (_op(_accu, tmp)) : _accu;
            }
            tmp = __shfl_up_sync(0xffffffff, _accu, 1, warpSize);
            tmp = (lane_id == 0) ? 0 : tmp;
            *_dst = tmp;
        }


        template <typename TypeOP, uint32_t scan_length>
        __device__ __inline__ static void cu_warp_inclusive_scan_fp32(TypeOP& _op,
            const float* _src, float* _dst, const uint32_t lane_id)
        {
            float _accu = *_src, tmp;
#pragma unroll
            for (int i = 1; i < scan_length; i *= 2) {
                tmp = __shfl_up_sync(0xffffffff, _accu, i, warpSize);

                _accu = lane_id > (i - 1) ? (_op(_accu, tmp)) : _accu;
            }
            *_dst = _accu;
        }


        template <typename TypeOP, uint32_t scan_length>
        __device__ __inline__ static void cu_warp_exclusive_scan_int32(TypeOP& _op,
            const int* _src, int* _dst, const uint32_t lane_id)
        {
            int _accu = *_src, tmp;
#pragma unroll
            for (int i = 1; i < scan_length; i *= 2) {
                tmp = __shfl_up_sync(0xffffffff, _accu, i, warpSize);

                _accu = lane_id > (i - 1) ? (_op(_accu, tmp)) : _accu;
            }
            tmp = __shfl_up_sync(0xffffffff, _accu, 1, warpSize);
            tmp = (lane_id == 0) ? 0 : tmp;
            *_dst = tmp;
        }


        template <typename TypeOP, uint32_t scan_length>
        __device__ __inline__ static void cu_warp_inclusive_scan_int32(TypeOP& _op,
            const int* _src, int* _dst, const uint32_t lane_id)
        {
            int _accu = *_src, tmp;
#pragma unroll
            for (int i = 1; i < scan_length; i *= 2) {
                tmp = __shfl_up_sync(0xffffffff, _accu, i, warpSize);

                _accu = lane_id > (i - 1) ? (_op(_accu, tmp)) : _accu;
            }
            *_dst = _accu;
        }
    }
}
}


#endif