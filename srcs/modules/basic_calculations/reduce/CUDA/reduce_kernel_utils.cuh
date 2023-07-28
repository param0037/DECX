/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _REDUCE_KERNEL_UTILS_CUH_
#define _REDUCE_KERNEL_UTILS_CUH_

#include "../../../core/basic.h"
#include "../../../core/utils/decx_cuda_vectypes_ops.cuh"
#include "../../../core/utils/cuda_int32_math_functions.cuh"


#define _REDUCE1D_BLOCK_DIM_ 32 * 8

namespace decx
{
namespace reduce
{
    namespace GPUK 
    {
        template <typename TypeOP, uint32_t engaged_thread>
        //the result is stored under the 0th thread
        __device__ __inline__ void cu_warp_reduce_fp32(TypeOP& _op, const float* _src, float* _dst)
        {
            float tmp;
            float accu = *_src;
#pragma unroll 
            for (int32_t i = engaged_thread / 2; i > 0; i >>= 1) {
                tmp = __shfl_down_sync(0xffffffff, accu, i, 32);
                accu = _op(tmp, accu);
            }
            *_dst = accu;
        }


        template <typename TypeOP, uint32_t engaged_thread>
        //the result is stored under the 0th thread
        __device__ __inline__ void cu_warp_reduce_fp16(TypeOP& _op, const __half* _src, __half* _dst)
        {
            __half tmp;
            __half accu = *_src;
#pragma unroll 
            for (int32_t i = engaged_thread / 2; i > 0; i >>= 1) {
                tmp = __shfl_down_sync(0xffffffff, accu, i, 32);
                accu = _op(tmp, accu);
            }
            *_dst = accu;
        }


        template <typename TypeOP, uint32_t engaged_thread>
        //the result is stored under the 0th thread
        __device__ __inline__ void cu_warp_reduce_int32(TypeOP& _op, const int32_t* _src, int32_t* _dst)
        {
            int32_t tmp;
            int32_t accu = *_src;
#pragma unroll 
            for (int32_t i = engaged_thread / 2; i > 0; i >>= 1) {
                tmp = __shfl_down_sync(0xffffffff, accu, i, 32);
                accu = _op(tmp, accu);
            }
            *_dst = accu;
        }


        __device__ __inline__ float float4_reduce_max(const float4 _in)
        {
            float res = _in.x;

            res = max(_in.y, res);
            res = max(_in.z, res);
            res = max(_in.w, res);

            return res;
        }


        __device__ __inline__ __half half8_max(const __half2 _in[4])
        {
#if __ABOVE_SM_53
            __half2 _max = _in[0];

            _max = decx::utils::cuda::__half2_max(_in[1], _max);
            _max = decx::utils::cuda::__half2_max(_in[2], _max);
            _max = decx::utils::cuda::__half2_max(_in[3], _max);

            return decx::utils::cuda::__half_max(_max.x, _max.y);
#endif
        }

        
        __device__ __inline__ __half half8_min(const __half2 _in[4])
        {
#if __ABOVE_SM_53
            __half2 _min = _in[0];

            _min = decx::utils::cuda::__half2_min(_in[1], _min);
            _min = decx::utils::cuda::__half2_min(_in[2], _min);
            _min = decx::utils::cuda::__half2_min(_in[3], _min);

            return decx::utils::cuda::__half_min(_min.x, _min.y);
#endif
        }


        __device__ __inline__ float float4_reduce_min(const float4 _in)
        {
            float res = _in.x;

            res = min(_in.y, res);
            res = min(_in.z, res);
            res = min(_in.w, res);

            return res;
        }


        __device__ __inline__ float float4_reduce_sum(const float4 _in)
        {
            float res = _in.x;
            res = __fadd_rn(_in.y, res);
            res = __fadd_rn(_in.z, res);
            res = __fadd_rn(_in.w, res);

            return res;
        }


        __device__ __inline__ float half8_reduce_sum(const float4 _in)
        {
            __half2 _accu = *((__half2*)&_in.x);
            _accu = __hadd2(*((__half2*)&_in.y), _accu);
            _accu = __hadd2(*((__half2*)&_in.z), _accu);
            _accu = __hadd2(*((__half2*)&_in.w), _accu);

            return __fadd_rn(__half2float(_accu.x), __half2float(_accu.y));
        }


        __device__ __inline__ int32_t int4_reduce_sum(const int4 _in)
        {
            int32_t res = _in.x;
            res = _in.y + res;
            res = _in.z + res;
            res = _in.w + res;

            return res;
        }


        __device__ __inline__ int32_t uchar16_reduce_sum(const int4 _in)
        {
            int32_t _accu = __vsadu4(_in.x, 0);
            _accu = _accu + __vsadu4(_in.y, 0);
            _accu = _accu + __vsadu4(_in.z, 0);
            _accu = _accu + __vsadu4(_in.w, 0);

            return _accu;
        }


        __device__ __inline__ uint8_t uchar16_max(const int4 _in)
        {
            int _tmp = _in.x;
            _tmp = __vmaxu4(_tmp, _in.y);
            _tmp = __vmaxu4(_tmp, _in.z);
            _tmp = __vmaxu4(_tmp, _in.w);

            _tmp = __vmaxu4(_tmp, __byte_perm(_tmp, 0, 0x1032));
            uint8_t _res = ((uchar4*)&_tmp)->x;

            return umax(_res, ((uchar4*)&_tmp)->y);
        }


        __device__ __inline__ uint8_t uchar16_min(const int4 _in)
        {
            int _tmp = _in.x;
            _tmp = __vminu4(_tmp, _in.y);
            _tmp = __vminu4(_tmp, _in.z);
            _tmp = __vminu4(_tmp, _in.w);

            _tmp = __vminu4(_tmp, __byte_perm(_tmp, 0, 0x1032));
            uint8_t _res = ((uchar4*)&_tmp)->x;

            return umin(_res, ((uchar4*)&_tmp)->y);
        }
    }
}
}


#endif