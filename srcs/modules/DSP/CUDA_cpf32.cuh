/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _CUDA_CPF32_CUH_
#define _CUDA_CPF32_CUH_

#include "../core/basic.h"
#include "../classes/classes_util.h"
#include "../core/utils/decx_cuda_vectypes_ops.cuh"


#ifdef _DECX_CUDA_PARTS_
namespace decx
{
namespace dsp {
namespace fft {
    namespace GPUK {
        __device__ static de::CPf _complex_mul_fp32(const de::CPf a, const de::CPf b);

        __device__ static float4 _complex_mul2_fp32(const float4 a, const float4 b);
        __device__ static float4 _complex_2mul1_fp32(const float4 a, const de::CPf b);


        __device__ __inline__ de::CPf _complex_fma_fp32(const de::CPf a, const de::CPf b, const de::CPf c);


        __device__ __inline__ float _complex_fma_preserve_R(const de::CPf a, const de::CPf b, const de::CPf c);


        __device__ __inline__ float4 _complex_fma2_fp32(const float4 a, const float4 b, const float4 c);
        __device__ __inline__ float4 _complex_fma2(const float4 a, const float4 b, const float4 c);


        __device__ __inline__ float4 _complex_2fma1_fp32(const float4 a, const de::CPf b, const float4 c);
        __device__ __inline__ float4 _complex_2fma1(const float4 a, const de::CPf b, const float4 c);


        __device__ __inline__ de::CPf _complex_add_fp32(const de::CPf a, const de::CPf b);
        __device__ __inline__ de::CPf _complex_sub_fp32(const de::CPf a, const de::CPf b);


        __device__ __inline__ double _complex_add_warp_call(const double a, const double b);


        __device__ __inline__ float4 _complex_add2(const float4 a, const float4 b);


        __device__ __inline__ float4 _complex_sum2_4(const float4 a, const float4 b, const float4 c, const float4 d);


        __device__ __inline__ de::CPf _complex_conjugate_fp32(const de::CPf __x);


        __device__ __inline__ decx::utils::_cuda_vec128 _complex4_conjugate_fp32(const decx::utils::_cuda_vec128& __x);
    }
}
}
}

__device__ static de::CPf 
decx::dsp::fft::GPUK::_complex_mul_fp32(const de::CPf a, const de::CPf b)
{
    return { __fsub_rn(__fmul_rn(a.real, b.real), __fmul_rn(a.image, b.image)),
             __fadd_rn(__fmul_rn(a.real, b.image), __fmul_rn(a.image, b.real)) };
}


__device__ static float4 
decx::dsp::fft::GPUK::_complex_mul2_fp32(const float4 a, const float4 b)
{
    return { __fsub_rn(__fmul_rn(a.x, b.x), __fmul_rn(a.y, b.y)),
             __fadd_rn(__fmul_rn(a.x, b.y), __fmul_rn(a.y, b.x)),
             __fsub_rn(__fmul_rn(a.z, b.z), __fmul_rn(a.w, b.w)),
             __fadd_rn(__fmul_rn(a.z, b.w), __fmul_rn(a.w, b.z)) };
}


__device__ static float4
decx::dsp::fft::GPUK::_complex_2mul1_fp32(const float4 a, const de::CPf b)
{
    return { __fsub_rn(__fmul_rn(a.x, b.real), __fmul_rn(a.y, b.image)),
             __fadd_rn(__fmul_rn(a.x, b.image), __fmul_rn(a.y, b.real)),
             __fsub_rn(__fmul_rn(a.z, b.real), __fmul_rn(a.w, b.image)),
             __fadd_rn(__fmul_rn(a.z, b.image), __fmul_rn(a.w, b.real)) };
}


__device__ __inline__ de::CPf 
decx::dsp::fft::GPUK::_complex_fma_fp32(const de::CPf a, const de::CPf b, const de::CPf c)
{
    return { __fsub_rn(__fmaf_rn(a.real, b.real, c.real), __fmul_rn(a.image, b.image)),
             __fadd_rn(__fmaf_rn(a.real, b.image, c.image), __fmul_rn(a.image, b.real)) };
}


__device__ __inline__ float
decx::dsp::fft::GPUK::_complex_fma_preserve_R(const de::CPf a, const de::CPf b, const de::CPf c)
{
    return __fsub_rn(__fmaf_rn(a.real, b.real, c.real), __fmul_rn(a.image, b.image));
}


__device__ __inline__ float4 
decx::dsp::fft::GPUK::_complex_fma2_fp32(const float4 a, const float4 b, const float4 c)
{
    return { __fsub_rn(__fmaf_rn(a.x, b.x, c.x), __fmul_rn(a.y, b.y)),
             __fadd_rn(__fmaf_rn(a.x, b.y, c.y), __fmul_rn(a.y, b.x)),
             __fsub_rn(__fmaf_rn(a.z, b.z, c.z), __fmul_rn(a.w, b.w)),
             __fadd_rn(__fmaf_rn(a.z, b.w, c.w), __fmul_rn(a.w, b.z)) };
}


__device__ __inline__ float4
decx::dsp::fft::GPUK::_complex_fma2(const float4 a, const float4 b, const float4 c)
{
    return { __fsub_rn(__fmaf_rn(a.x, b.x, c.x), __fmul_rn(a.y, b.y)),
             __fadd_rn(__fmaf_rn(a.x, b.y, c.y), __fmul_rn(a.y, b.x)),
             __fsub_rn(__fmaf_rn(a.z, b.z, c.z), __fmul_rn(a.w, b.w)),
             __fadd_rn(__fmaf_rn(a.z, b.w, c.w), __fmul_rn(a.w, b.z)) };
}


__device__ __inline__ float4 
decx::dsp::fft::GPUK::_complex_2fma1(const float4 a, const de::CPf b, const float4 c)
{
    float4 res;
    res.x = __fsub_rn(__fmaf_rn(a.x, b.real, c.x), __fmul_rn(a.y, b.image));     // real 1
    res.y = __fadd_rn(__fmaf_rn(a.x, b.image, c.y), __fmul_rn(a.y, b.real));     // image 1
    res.z = __fsub_rn(__fmaf_rn(a.z, b.real, c.z), __fmul_rn(a.w, b.image));     // real 2
    res.w = __fadd_rn(__fmaf_rn(a.z, b.image, c.w), __fmul_rn(a.w, b.real));     // image 2
    return res;
}


__device__ __inline__ float4
decx::dsp::fft::GPUK::_complex_2fma1_fp32(const float4 a, const de::CPf b, const float4 c)
{
    return make_float4(__fsub_rn(__fmaf_rn(a.x, b.real, c.x), __fmul_rn(a.y, b.image)),
                       __fadd_rn(__fmaf_rn(a.x, b.image, c.y), __fmul_rn(a.y, b.real)),
                       __fsub_rn(__fmaf_rn(a.z, b.real, c.z), __fmul_rn(a.w, b.image)),
                       __fadd_rn(__fmaf_rn(a.z, b.image, c.w), __fmul_rn(a.w, b.real)));
}



__device__ __inline__ de::CPf 
decx::dsp::fft::GPUK::_complex_add_fp32(const de::CPf a, const de::CPf b)
{
    de::CPf res;
    res.real = __fadd_rn(a.real, b.real);
    res.image = __fadd_rn(a.image, b.image);
    return res;
}


__device__ __inline__ de::CPf
decx::dsp::fft::GPUK::_complex_sub_fp32(const de::CPf a, const de::CPf b)
{
    de::CPf res;
    res.real = __fsub_rn(a.real, b.real);
    res.image = __fsub_rn(a.image, b.image);
    return res;
}



__device__ __inline__ double
decx::dsp::fft::GPUK::_complex_add_warp_call(const double a, const double b)
{
    de::CPf res;
    res.real = __fadd_rn(((de::CPf*)&a)->real, ((de::CPf*)&b)->real);
    res.image = __fadd_rn(((de::CPf*)&a)->image, ((de::CPf*)&b)->image);
    return *((double*)&res);
}


__device__ __inline__ float4
decx::dsp::fft::GPUK::_complex_add2(const float4 a, const float4 b)
{
    float4 res;
    res.x = __fadd_rn(a.x, b.x);
    res.y = __fadd_rn(a.y, b.y);
    res.z = __fadd_rn(a.z, b.z);
    res.w = __fadd_rn(a.w, b.w);
    return res;
}


__device__ __inline__ float4 
decx::dsp::fft::GPUK::_complex_sum2_4(const float4 a, const float4 b, const float4 c, const float4 d)
{
    float4 res;
    res.x = __fadd_rn(__fadd_rn(a.x, b.x), __fadd_rn(c.x, d.x));
    res.y = __fadd_rn(__fadd_rn(a.y, b.y), __fadd_rn(c.y, d.y));
    res.z = __fadd_rn(__fadd_rn(a.z, b.z), __fadd_rn(c.z, d.z));
    res.w = __fadd_rn(__fadd_rn(a.w, b.w), __fadd_rn(c.w, d.w));
    return res;
}


__device__ __inline__ de::CPf 
decx::dsp::fft::GPUK::_complex_conjugate_fp32(const de::CPf __x)
{
    de::CPf res = __x;
    *((uint32_t*)&res.image) ^= 0x80000000;
    return res;
}


__device__ __inline__ decx::utils::_cuda_vec128 
decx::dsp::fft::GPUK::_complex4_conjugate_fp32(const decx::utils::_cuda_vec128& __x)
{
    decx::utils::_cuda_vec128 res = __x;
    res._arrui[1] ^= 0x80000000;
    res._arrui[3] ^= 0x80000000;
    return res;
}


#endif

#endif