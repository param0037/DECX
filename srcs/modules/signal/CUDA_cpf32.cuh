/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/

#ifndef _CUDA_CPF32_CUH_
#define _CUDA_CPF32_CUH_

#include "../core/basic.h"
#include "../classes/classes_util.h"


#ifdef _DECX_CUDA_CODES_
namespace decx
{
    namespace signal {
        namespace cuda {
            namespace dev {
                __device__ static de::CPf _complex_mul(const de::CPf a, const de::CPf b);


                __device__ __inline__ de::CPf _complex_fma(const de::CPf a, const de::CPf b, const de::CPf c);


                __device__ __inline__ float _complex_fma_preserve_R(const de::CPf a, const de::CPf b, const de::CPf c);


                __device__ __inline__ float4 _complex_fma2(const float4 a, const float4 b, const float4 c);


                __device__ __inline__ float4 _complex_2fma1(const float4 a, const de::CPf b, const float4 c);


                __device__ __inline__ de::CPf _complex_add(const de::CPf a, const de::CPf b);


                __device__ __inline__ float4 _complex_add2(const float4 a, const float4 b);


                __device__ __inline__ float4 _complex_sum2_4(const float4 a, const float4 b, const float4 c, const float4 d);
            }
        }
    }
}
__device__ static de::CPf 
decx::signal::cuda::dev::_complex_mul(const de::CPf a, const de::CPf b)
{
    de::CPf res;
    res.real = __fsub_rn(__fmul_rn(a.real, b.real), __fmul_rn(a.image, b.image));
    res.image = __fadd_rn(__fmul_rn(a.real, b.image), __fmul_rn(a.image, b.real));
    return res;
}



__device__ __inline__ de::CPf 
decx::signal::cuda::dev::_complex_fma(const de::CPf a, const de::CPf b, const de::CPf c)
{
    de::CPf res;
    res.real = __fsub_rn(fmaf(a.real, b.real, c.real), __fmul_rn(a.image, b.image));
    res.image = __fadd_rn(fmaf(a.real, b.image, c.image), __fmul_rn(a.image, b.real));
    return res;
}



__device__ __inline__ float
decx::signal::cuda::dev::_complex_fma_preserve_R(const de::CPf a, const de::CPf b, const de::CPf c)
{
    float res;
    res = __fsub_rn(fmaf(a.real, b.real, c.real), __fmul_rn(a.image, b.image));
    return res;
}



__device__ __inline__ float4 
decx::signal::cuda::dev::_complex_fma2(const float4 a, const float4 b, const float4 c)
{
    float4 res;
    res.x = __fsub_rn(fmaf(a.x, b.x, c.x), __fmul_rn(a.y, b.y));
    res.y = __fadd_rn(fmaf(a.x, b.y, c.y), __fmul_rn(a.y, b.x));
    res.z = __fsub_rn(fmaf(a.z, b.z, c.z), __fmul_rn(a.w, b.w));
    res.w = __fadd_rn(fmaf(a.z, b.w, c.w), __fmul_rn(a.w, b.z));
    return res;
}


__device__ __inline__ float4 
decx::signal::cuda::dev::_complex_2fma1(const float4 a, const de::CPf b, const float4 c)
{
    float4 res;
    res.x = __fsub_rn(fmaf(a.x, b.real, c.x), __fmul_rn(a.y, b.image));
    res.y = __fadd_rn(fmaf(a.x, b.image, c.y), __fmul_rn(a.y, b.real));
    res.z = __fsub_rn(fmaf(a.z, b.real, c.z), __fmul_rn(a.w, b.image));
    res.w = __fadd_rn(fmaf(a.z, b.image, c.w), __fmul_rn(a.w, b.real));
    return res;
}


__device__ __inline__ de::CPf 
decx::signal::cuda::dev::_complex_add(const de::CPf a, const de::CPf b)
{
    de::CPf res;
    res.real = __fadd_rn(a.real, b.real);
    res.image = __fadd_rn(a.image, b.image);
    return res;
}


__device__ __inline__ float4
decx::signal::cuda::dev::_complex_add2(const float4 a, const float4 b)
{
    float4 res;
    res.x = __fadd_rn(a.x, b.x);
    res.y = __fadd_rn(a.y, b.y);
    res.z = __fadd_rn(a.z, b.z);
    res.w = __fadd_rn(a.w, b.w);
    return res;
}


__device__ __inline__ float4 
decx::signal::cuda::dev::_complex_sum2_4(const float4 a, const float4 b, const float4 c, const float4 d)
{
    float4 res;
    res.x = __fadd_rn(__fadd_rn(a.x, b.x), __fadd_rn(c.x, d.x));
    res.y = __fadd_rn(__fadd_rn(a.y, b.y), __fadd_rn(c.y, d.y));
    res.z = __fadd_rn(__fadd_rn(a.z, b.z), __fadd_rn(c.z, d.z));
    res.w = __fadd_rn(__fadd_rn(a.w, b.w), __fadd_rn(c.w, d.w));
    return res;
}
#endif

#endif