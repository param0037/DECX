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
                __i32_add(const int a, const int b)
            {
                return a + b;
            }


            __device__ __inline__ ushort
                __u16_add(const ushort a, const ushort b)
            {
                return a + b;
            }


            __device__ __inline__ int
                __i32_max(const int a, const int b)
            {
                return max(a, b);
            }


            __device__ __inline__ int
                __i32_min(const int a, const int b)
            {
                return min(a, b);
            }

            template <class _TyData>
            using device_op_type = _TyData(_TyData, _TyData);
        }
    }
}


#endif