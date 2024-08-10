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


#ifndef _CUDA_CPD64_CUH_
#define _CUDA_CPD64_CUH_

#include "../basic.h"
#include "../Classes/classes_util.h"
#include "decx_cuda_vectypes_ops.cuh"


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
