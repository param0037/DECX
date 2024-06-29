/**
*   ----------------------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ----------------------------------------------------------------------------------
* 
* This is a part of the open source project named "DECX", a high-performance scientific
* computational library. This project follows the MIT License. For more information 
* please visit https://github.com/param0037/DECX.
* 
* Copyright (c) 2021 Wayne Anderson
* 
* Permission is hereby granted, free of charge, to any person obtaining a copy of this 
* software and associated documentation files (the "Software"), to deal in the Software 
* without restriction, including without limitation the rights to use, copy, modify, 
* merge, publish, distribute, sublicense, and/or sell copies of the Software, and to 
* permit persons to whom the Software is furnished to do so, subject to the following 
* conditions:
* 
* The above copyright notice and this permission notice shall be included in all copies 
* or substantial portions of the Software.
* 
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
* INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR 
* PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE 
* FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR 
* OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
* DEALINGS IN THE SOFTWARE.
*/


#ifndef _DECX_CUDA_MATH_FUNCTIONS_CUH_
#define _DECX_CUDA_MATH_FUNCTIONS_CUH_

#include "../../core/basic.h"
#include "decx_cuda_vectypes_ops.cuh"


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


        __device__ __inline__ float4
        __float_div4_1(const float4& numer_vec4, const float deno)
        {
            return { __fdividef(numer_vec4.x, deno),
                     __fdividef(numer_vec4.y, deno),
                     __fdividef(numer_vec4.z, deno),
                     __fdividef(numer_vec4.w, deno) };
        }


        __device__ __inline__ float2
        __float_div2_1(const float2& numer_vec2, const float deno)
        {
            return { __fdividef(numer_vec2.x, deno),
                     __fdividef(numer_vec2.y, deno) };
        }

        __device__ __inline__ double2
        __double_div2_1(const double2& numer_vec2, const double deno)
        {
            return { __ddiv_rn(numer_vec2.x, deno),
                     __ddiv_rn(numer_vec2.y, deno) };
        }

        __device__ __inline__ float2
        __float_add2(const float2 a, const float2 b)
        {
            return { __fadd_rn(a.x, b.x),
                     __fadd_rn(a.y, b.y) };
        }


        __device__ __inline__ double2
        __double_add2(const double2 a, const double2 b)
        {
            return { __dadd_rn(a.x, b.x),
                     __dadd_rn(a.y, b.y) };
        }


        __device__ __inline__ double2
        __double_sub2(const double2 a, const double2 b)
        {
            return { __dsub_rn(a.x, b.x),
                     __dsub_rn(a.y, b.y) };
        }


        __device__ __inline__ float4
        __float_add4(const float4& a, const float4& b)
        {
            return { __fadd_rn(a.x, b.x), __fadd_rn(a.y, b.y),
                     __fadd_rn(a.z, b.z), __fadd_rn(a.w, b.w) };
        }


        __device__ __inline__ float4
        __float_sub4(const float4& a, const float4& b)
        {
            return { __fsub_rn(a.x, b.x), __fsub_rn(a.y, b.y),
                     __fsub_rn(a.z, b.z), __fsub_rn(a.w, b.w) };
        }


        __device__ __inline__ float2
        __float_sub2(const float2& a, const float2& b)
        {
            return { __fsub_rn(a.x, b.x), __fsub_rn(a.y, b.y) };
        }


        __device__ __inline__ float4
        __float_mul4(const float4& a, const float4& b)
        {
            return { __fmul_rn(a.x, b.x),
                     __fmul_rn(a.y, b.y),
                     __fmul_rn(a.z, b.z),
                     __fmul_rn(a.w, b.w) };
        }


        __device__ __inline__ float4
        __float_mul4_1(const float4& a, const float b)
        {
            /*decx::utils::_cuda_vec128 res;
            res._vf.x = __fmul_rn(a._vf.x, b);
            res._vf.y = __fmul_rn(a._vf.y, b);
            res._vf.z = __fmul_rn(a._vf.z, b);
            res._vf.w = __fmul_rn(a._vf.w, b);
            return res;*/

            return { __fmul_rn(a.x, b),
                     __fmul_rn(a.y, b),
                     __fmul_rn(a.z, b),
                     __fmul_rn(a.w, b) };
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


        __device__ __inline__ float4
        __fmaf_v4_v1_v4(const float4 a, const float b, const float4 c)
        {
            return { __fmaf_rn(a.x, b, c.x),
                    __fmaf_rn(a.y, b, c.y),
                    __fmaf_rn(a.z, b, c.z),
                    __fmaf_rn(a.w, b, c.w) };
        }


        __device__ __inline__ double2
        __fma_v2_v1_v2(const double2 a, const double b, const double2 c)
        {
            return { __fma_rn(a.x, b, c.x),
                     __fma_rn(a.y, b, c.y)};
        }


        __device__ __inline__ float4
        __fmaf_v4_v1_v4_u8(const uchar4 a, const float b, const float4 c)
        {
            return { __fmaf_rn(__int2float_rn(a.x), b, c.x),
                    __fmaf_rn(__int2float_rn(a.y), b, c.y),
                    __fmaf_rn(__int2float_rn(a.z), b, c.z),
                    __fmaf_rn(__int2float_rn(a.w), b, c.w) };
        }


        __device__ __inline__ float4
        __fmah_v8_v1_v8(const float4 a, const __half2 b2, const float4 c)
        {
            float4 res;
            ((__half2*)&res)[0] = __hfma2(((__half2*)&a)[0], b2, ((__half2*)&c)[0]);
            ((__half2*)&res)[1] = __hfma2(((__half2*)&a)[1], b2, ((__half2*)&c)[1]);
            ((__half2*)&res)[2] = __hfma2(((__half2*)&a)[2], b2, ((__half2*)&c)[2]);
            ((__half2*)&res)[3] = __hfma2(((__half2*)&a)[3], b2, ((__half2*)&c)[3]);
            return res;
        }


        __device__ __inline__ float4
        __fmaf_v4(const float4 a, const float4 b, const float4 c)
        {
            float4 res;
            res.x = __fmaf_rn(a.x, b.x, c.x);
            res.y = __fmaf_rn(a.y, b.y, c.y);
            res.z = __fmaf_rn(a.z, b.z, c.z);
            res.w = __fmaf_rn(a.w, b.w, c.w);
            return res;
        }


        __device__ __inline__ decx::utils::_cuda_vec128
        __hmul_v8_1(const decx::utils::_cuda_vec128& a, const __half b)
        {
            decx::utils::_cuda_vec128 res;
#if __ABOVE_SM_53
            __half2 _b2;
            _b2.x = b;
            _b2.y = b;
            res._arrh2[0] = __hmul2(a._arrh2[0], _b2);
            res._arrh2[1] = __hmul2(a._arrh2[1], _b2);
            res._arrh2[2] = __hmul2(a._arrh2[2], _b2);
            res._arrh2[3] = __hmul2(a._arrh2[3], _b2);
#endif
            return res;
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