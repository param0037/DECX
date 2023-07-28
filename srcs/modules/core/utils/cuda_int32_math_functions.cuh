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
            __device__ __inline__ int
                __i32_add(const int32_t a, const int32_t b)
            {
                return a + b;
            }


            __device__ __inline__ uint16_t
                __u16_add(const uint16_t a, const uint16_t b)
            {
                return a + b;
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

            template <class _TyData>
            using device_op_type = _TyData(_TyData, _TyData);
        }
    }
}


#endif