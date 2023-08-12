/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _CUDA_INT32_MATH_FUNCTION_CUH_
#define _CUDA_INT32_MATH_FUNCTION_CUH_

#include "../../core/basic.h"


namespace decx
{
    namespace utils
    {
        namespace cuda
        {
            template <typename _Ty>
            using cu_math_ops = _Ty(_Ty, _Ty);


            __device__ __inline__ int32_t
                __i32_add(const int32_t a, const int32_t b)
            {
                return a + b;
            }


            __device__ __inline__ double
                __i32_add2(const double a, const double b)
            {
                decx::utils::_cuda_vec64 _res;
                _res._vi2.x = ((int2*)&a)->x + ((int2*)&b)->x;
                _res._vi2.y = ((int2*)&a)->y + ((int2*)&b)->y;

                return _res._fp64;
            }


            __device__ __inline__ double
                __float_add2(const double a, const double b)
            {
                float2 res;
                res.x = __fadd_rn(((float2*)&a)->x, ((float2*)&b)->x);
                res.y = __fadd_rn(((float2*)&a)->y, ((float2*)&b)->y);
                return *((double*)&res);
            }


            __device__ __inline__ double
                __float_max2(const double a, const double b)
            {
                float2 res;
                res.x = max(((float2*)&a)->x, ((float2*)&b)->x);
                res.y = max(((float2*)&a)->y, ((float2*)&b)->y);
                return *((double*)&res);
            }


            __device__ __inline__ double
                __float_min2(const double a, const double b)
            {
                float2 res;
                res.x = min(((float2*)&a)->x, ((float2*)&b)->x);
                res.y = min(((float2*)&a)->y, ((float2*)&b)->y);
                return *((double*)&res);
            }


            __device__ __inline__ uint16_t
                __u16_add(const uint16_t a, const uint16_t b)
            {
                return a + b;
            }


            __device__ __inline__ double
                __u16_add4(const double a, const double b)
            {
                decx::utils::_cuda_vec64 _res;
                _res._vui2.x = __vadd2(((uint2*)&a)->x, ((uint2*)&b)->x);
                _res._vui2.y = __vadd2(((uint2*)&a)->y, ((uint2*)&b)->y);

                return _res._fp64;
            }


            __device__ __inline__ int32_t
                __i32_max(const int32_t a, const int32_t b)
            {
                return max(a, b);
            }


            __device__ __inline__ float
                __fp32_max(const float a, const float b)
            {
                return max(a, b);
            }

            __device__ __inline__ double
                __fp64_max(const double a, const double b)
            {
                return max(a, b);
            }

            __device__ __inline__ double
                __fp64_min(const double a, const double b)
            {
                return min(a, b);
            }


            __device__ __inline__ __half
                __half_max(const __half a, const __half b)
            {
#if __ABOVE_SM_53
                return __hge(a, b) ? a : b;
#endif
            }

            __device__ __inline__ __half
                __half_min(const __half a, const __half b)
            {
#if __ABOVE_SM_53
                return __hle(a, b) ? a : b;
#endif
            }

            __device__ __inline__ __half2
                __half2_max(const __half2 a, const __half2 b)
            {
#if __ABOVE_SM_53
                __half2 _crit = __hge2(a, b);
                int32_t _mask = *((int32_t*)&_crit) | (*((int32_t*)&_crit) << 2) | (*((int32_t*)&_crit) >> 2);
                _mask = __byte_perm(_mask, 0, 0x3311);
                int32_t res = _mask & *((int32_t*)&a) | (~_mask & *((int32_t*)&b));
                return *((__half2*)&res);
#endif
            }

            __device__ __inline__ __half2
                __half2_min(const __half2 a, const __half2 b)
            {
#if __ABOVE_SM_53
                __half2 _crit = __hle2(a, b);
                int32_t _mask = *((int32_t*)&_crit) | (*((int32_t*)&_crit) << 2) | (*((int32_t*)&_crit) >> 2);
                _mask = __byte_perm(_mask, 0, 0x3311);
                int32_t res = _mask & *((int32_t*)&a) | (~_mask & *((int32_t*)&b));
                return *((__half2*)&res);
#endif
            }


            __device__ __inline__ float
                __fp32_min(const float a, const float b)
            {
                return min(a, b);
            }


            __device__ __inline__ int32_t
                __i32_min(const int32_t a, const int32_t b)
            {
                return min(a, b);
            }



            //template <class _TyData>
            //using device_op_type = _TyData(_TyData, _TyData);
        }

        // -------------------------------------- typecasting ------------------------------------------
        namespace cuda
        {
            __device__ __inline__ double
                __cvt_uchar4_ushort4(const int32_t _in)
            {
                _cuda_vec64 _res;
                _res._vi2.x = __byte_perm(0, _in, 0x0504);
                _res._vi2.y = __byte_perm(0, _in, 0x0706);

                return _res._fp64;
            }
        }
    }
}


#endif