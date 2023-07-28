/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _DECX_UTILS_DEVICE_FUNCTIONS_CUH_
#define _DECX_UTILS_DEVICE_FUNCTIONS_CUH_


#include "../basic.h"



namespace decx
{
    namespace utils {
        union __align__(16) _cuda_vec128
        {
            float4 _vf;
            double2 _vd;
            int4 _vi;

            __half2 _arrh2[4];
            __half _arrh[8];

            uchar4 _arru8_4[4];

            uint16_t _arrs[8];

            float _arrf[4];
            double _arrd[2];
            int _arri[4];

            uint8_t _arru8[16];

            __host__ __device__ _cuda_vec128() {}
        };


        union __align__(8) _cuda_vec64
        {
            float2 _vf2;
            size_t _ull;
            uint8_t _v_uint8[8];
        };
    }
}



namespace decx
{
    namespace utils
    {
        __host__ __device__ __inline__ float4 vec4_set1_fp32(const float __x);


        __host__ __device__ __inline__ int4 vec4_set1_int32(const int __x);


        __host__ __device__ __inline__ double2 vec2_set1_fp64(const double __x);


        __device__ __inline__ float4
            add_vec4(const float4 __x, const float4 __y);


        __device__ __inline__ float4
            sub_vec4(const float4 __x, const float4 __y);


        __device__ __inline__ int4
            sub_vec4(const int4 __x, const uchar4 __y);


        __device__ __inline__ float4
            add_scalar_vec4(const float4 __x, const float __y);


        __device__ __inline__ float4
            add_scalar_vec4(const decx::utils::_cuda_vec128& __x, const __half2 __y);


        __device__ __inline__ int4
            add_scalar_vec4(const int4 __x, const int __y);


        __device__ __inline__ double4
            add_scalar_vec4(const double4 __x, const int2 __y);


        __device__ __inline__ float4
            add_scalar_vec4(const decx::utils::_cuda_vec128& __x, const __half __y);


        __device__ __inline__ int4
            add_scalar_vec4(const decx::utils::_cuda_vec128& __x, const ushort __y);
    }
}


__device__ __inline__ float4
decx::utils::add_vec4(const float4 __x, const float4 __y)
{
    float4 _dst;

    _dst.x = __fadd_rn(__x.x, __y.x);
    _dst.y = __fadd_rn(__x.y, __y.y);
    _dst.z = __fadd_rn(__x.z, __y.z);
    _dst.w = __fadd_rn(__x.w, __y.w);
    return _dst;
}



__device__ __inline__ float4
decx::utils::sub_vec4(const float4 __x, const float4 __y)
{
    float4 _dst;
    _dst.x = __fsub_rn(__x.x, __y.x);
    _dst.y = __fsub_rn(__x.y, __y.y);
    _dst.z = __fsub_rn(__x.z, __y.z);
    _dst.w = __fsub_rn(__x.w, __y.w);

    return _dst;
}


__device__ __inline__ float4
decx::utils::add_scalar_vec4(const decx::utils::_cuda_vec128& __x, const __half2 __y)
{
#if __ABOVE_SM_53
    float4 _dst;
    *((__half2*)&_dst.x) = __hadd2(__x._arrh2[0], __y);
    *((__half2*)&_dst.y) = __hadd2(__x._arrh2[1], __y);
    *((__half2*)&_dst.z) = __hadd2(__x._arrh2[2], __y);
    *((__half2*)&_dst.w) = __hadd2(__x._arrh2[3], __y);

    return _dst;
#endif
}


__device__ __inline__ int4
decx::utils::sub_vec4(const int4 __x, const uchar4 __y)
{
    int4 _dst;
    _dst.x = __x.x - __y.x;
    _dst.y = __x.y - __y.y;
    _dst.z = __x.z - __y.z;
    _dst.w = __x.w - __y.w;

    return _dst;
}


__device__ __inline__ float4
decx::utils::add_scalar_vec4(const float4 __x, const float __y)
{
    float4 _dst;
    _dst.x = __fadd_rn(__x.x, __y);
    _dst.y = __fadd_rn(__x.y, __y);
    _dst.z = __fadd_rn(__x.z, __y);
    _dst.w = __fadd_rn(__x.w, __y);
    return _dst;
}


__device__ __inline__ int4
decx::utils::add_scalar_vec4(const int4 __x, const int __y)
{
    int4 _dst;
    _dst.x = __x.x + __y;
    _dst.y = __x.y + __y;
    _dst.z = __x.z + __y;
    _dst.w = __x.w + __y;
    return _dst;
}


__device__ __inline__ float4
decx::utils::add_scalar_vec4(const decx::utils::_cuda_vec128& __x, const __half __y)
{
#if __ABOVE_SM_53
    decx::utils::_cuda_vec128 _dst;
    half2 _add;
    _add.x = __y;
    _add.y = __y;

    _dst._arrh2[0] = __hadd2(__x._arrh2[0], _add);
    _dst._arrh2[1] = __hadd2(__x._arrh2[1], _add);
    _dst._arrh2[2] = __hadd2(__x._arrh2[2], _add);
    _dst._arrh2[3] = __hadd2(__x._arrh2[3], _add);

    return _dst._vf;
#endif
}


__device__ __inline__ int4
decx::utils::add_scalar_vec4(const decx::utils::_cuda_vec128& __x, const ushort __y)
{
    decx::utils::_cuda_vec128 _dst;

    int _add = (((int)__y) << 16) | ((int)__y);

    _dst._arri[0] = __vadd2(__x._arri[0], _add);
    _dst._arri[1] = __vadd2(__x._arri[1], _add);
    _dst._arri[2] = __vadd2(__x._arri[2], _add);
    _dst._arri[3] = __vadd2(__x._arri[3], _add);

    return _dst._vi;
}


__host__ __device__ __inline__ float4 
decx::utils::vec4_set1_fp32(const float __x)
{
    return make_float4(__x, __x, __x, __x);
}


__host__ __device__ __inline__ int4
decx::utils::vec4_set1_int32(const int __x)
{
    return make_int4(__x, __x, __x, __x);
}




__host__ __device__ __inline__ double2
decx::utils::vec2_set1_fp64(const double __x)
{
    return make_double2(__x, __x);
}



__device__ __inline__ double4
decx::utils::add_scalar_vec4(const double4 __x, const int2 __y)
{
    double4 _res;

    *((int2*)&_res.x) = make_int2(__vadd2(((int2*)&__x.x)->x, __y.x),
                                  __vadd2(((int2*)&__x.x)->y, __y.y));
    *((int2*)&_res.y) = make_int2(__vadd2(((int2*)&__x.y)->x, __y.x),
                                  __vadd2(((int2*)&__x.y)->y, __y.y));
    *((int2*)&_res.z) = make_int2(__vadd2(((int2*)&__x.z)->x, __y.x),
                                  __vadd2(((int2*)&__x.z)->y, __y.y));
    *((int2*)&_res.w) = make_int2(__vadd2(((int2*)&__x.w)->x, __y.x),
                                  __vadd2(((int2*)&__x.w)->y, __y.y));

    return _res;
}


#endif