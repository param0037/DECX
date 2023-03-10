/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/


#ifndef _DECX_UTILS_DEVICE_FUNCTIONS_CUH_
#define _DECX_UTILS_DEVICE_FUNCTIONS_CUH_


#include "../basic.h"


namespace decx
{
    namespace utils
    {
        __host__ __device__ __inline__ float4 vec4_set1_fp32(const float __x);


        __host__ __device__ __inline__ double2 vec2_set1_fp64(const double __x);


        __device__ __inline__ void
            float4_vector_add(float4* __x, float4* __y, float4* _dst);
    }
}


__device__ __inline__ void
decx::utils::float4_vector_add(float4* __x, float4* __y, float4* _dst)
{
    _dst->x = __fadd_rn(__x->x, __y->x);
    _dst->y = __fadd_rn(__x->y, __y->y);
    _dst->z = __fadd_rn(__x->z, __y->z);
    _dst->w = __fadd_rn(__x->w, __y->w);
}


__host__ __device__ __inline__ float4 
decx::utils::vec4_set1_fp32(const float __x)
{
    return make_float4(__x, __x, __x, __x);
}



__host__ __device__ __inline__ double2
decx::utils::vec2_set1_fp64(const double __x)
{
    return make_double2(__x, __x);
}



namespace decx
{
    namespace utils {
        union __align__(16) _cuda_vec128
        {
            float4 _vf;
            double2 _vd;
            int4 _vi;
        };
    }
}


#endif