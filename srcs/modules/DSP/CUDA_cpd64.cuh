/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _CUDA_CPD64_CUH_
#define _CUDA_CPD64_CUH_

#include "../core/basic.h"
#include "../classes/classes_util.h"
#include "../core/utils/decx_cuda_vectypes_ops.cuh"


#ifdef _DECX_CUDA_PARTS_
namespace decx
{
namespace dsp {
namespace fft {
    namespace GPUK {
        __device__ static de::CPd _complex_mul_fp64(const de::CPd a, const de::CPd b);

        __device__ __inline__ de::CPd _complex_fma_fp64(const de::CPd a, const de::CPd b, const de::CPd c);


        __device__ __inline__ de::CPd _complex_conjugate_fp64(const de::CPd __x);
    }
}
}
}

__device__ static de::CPd
decx::dsp::fft::GPUK::_complex_mul_fp64(const de::CPd a, const de::CPd b)
{
    return { __dsub_rn(__dmul_rn(a.real, b.real), __dmul_rn(a.image, b.image)),
             __dadd_rn(__dmul_rn(a.real, b.image), __dmul_rn(a.image, b.real)) };
}


__device__ __inline__ de::CPd
decx::dsp::fft::GPUK::_complex_fma_fp64(const de::CPd a, const de::CPd b, const de::CPd c)
{
    return { __dsub_rn(__fma_rn(a.real, b.real, c.real), __dmul_rn(a.image, b.image)),
             __dadd_rn(__fma_rn(a.real, b.image, c.image), __dmul_rn(a.image, b.real)) };
}


__device__ __inline__ de::CPd
decx::dsp::fft::GPUK::_complex_conjugate_fp64(const de::CPd __x)
{
    de::CPd res = __x;
    *((uint64_t*)&res.image) ^= 0x8000000000000000;
    return res;
}



#endif      // #ifdef _DECX_CUDA_PARTS_


#endif      // #ifndef _CUDA_CPD64_CUH_
