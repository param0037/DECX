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

#include "fill_kernels.h"

#define _RADN_LCG_B_ 1664525
#define _RADN_LCG_A_ 0x77673563


_THREAD_FUNCTION_ void 
decx::CPUK::fill1D_rand_int32(int32_t* dst, const uint64_t proc_len_v1, const int32_t min, const int32_t max)
{
    if (max == min) return;

    decx::utils::simd::xmm256_reg seed_v8, mask, res;
    for (int i = 0; i < 8; ++i){
        seed_v8._arrui[i] = rand();
    }

    __m256i LCG_A = _mm256_set1_epi32(_RADN_LCG_A_);
    __m256i LCG_B = _mm256_set1_epi32(_RADN_LCG_B_);

    const uint32_t range = (uint32_t)abs(max - min);
    const float scale_rcp = (float)range / (float)(0x7fffffff);

    const uint8_t _L = proc_len_v1 % 8;
    
    mask._vf = _mm256_setzero_ps();
    for (uint8_t i = 0; i < _L; ++i) mask._arrui[i] = 0xFFFFFFFF;

    res._vf = seed_v8._vf;

    for (uint64_t i = 0; i < proc_len_v1 / 8; ++i){
        res._vi = _mm256_mullo_epi32(res._vi, LCG_A);
        res._vi = _mm256_add_epi32(res._vi, LCG_B);
        
        res._vi = _mm256_and_si256(res._vi, _mm256_set1_epi32(0x7fffffff));

        res._vf = _mm256_cvtepi32_ps(res._vi);
        res._vf = _mm256_mul_ps(res._vf, _mm256_set1_ps(scale_rcp));
        res._vi = _mm256_cvtps_epi32(res._vf);
        
        res._vi = _mm256_add_epi32(res._vi, _mm256_set1_epi32(min));
        _mm256_store_ps((float*)dst + i * 8, res._vf);
    }
    if (_L) {
        decx::utils::simd::xmm256_reg masked_vals;
        res._vi = _mm256_mullo_epi32(res._vi, LCG_A);
        res._vi = _mm256_add_epi32(res._vi, LCG_B);
        
        res._vi = _mm256_and_si256(res._vi, _mm256_set1_epi32(0x7fffffff));

        res._vf = _mm256_cvtepi32_ps(res._vi);
        res._vf = _mm256_mul_ps(res._vf, _mm256_set1_ps(scale_rcp));
        res._vi = _mm256_cvtps_epi32(res._vf);

        res._vi = _mm256_add_epi32(res._vi, _mm256_set1_epi32(min));

        masked_vals._vf = _mm256_and_ps(mask._vf, res._vf);
        _mm256_store_ps((float*)dst + (proc_len_v1 / 8) * 8, masked_vals._vf);
    }
}



_THREAD_FUNCTION_ void 
decx::CPUK::fill1D_rand_fp32(float* dst, const uint64_t proc_len_v1, const float min, const float max)
{
    if (max == min) return;

    decx::utils::simd::xmm256_reg seed_v8, mask, res;
    for (int i = 0; i < 8; ++i){
        seed_v8._arrui[i] = rand();
    }

    __m256i LCG_A = _mm256_set1_epi32(_RADN_LCG_A_);
    __m256i LCG_B = _mm256_set1_epi32(_RADN_LCG_B_);

    const uint32_t range = (uint32_t)abs(max - min);
    const float scale_rcp = (float)range / (float)(0x7fffffff);

    const uint8_t _L = proc_len_v1 % 8;
    
    mask._vf = _mm256_setzero_ps();
    for (uint8_t i = 0; i < _L; ++i) mask._arrui[i] = 0xFFFFFFFF;

    res._vf = seed_v8._vf;

    for (uint64_t i = 0; i < proc_len_v1 / 8; ++i){
        res._vi = _mm256_mullo_epi32(res._vi, LCG_A);
        res._vi = _mm256_add_epi32(res._vi, LCG_B);
        
        res._vi = _mm256_and_si256(res._vi, _mm256_set1_epi32(0x7fffffff));

        res._vf = _mm256_cvtepi32_ps(res._vi);
        res._vf = _mm256_mul_ps(res._vf, _mm256_set1_ps(scale_rcp));
        
        res._vf = _mm256_add_ps(res._vf, _mm256_set1_ps(min));
        _mm256_store_ps(dst + i * 8, res._vf);
    }
    if (_L) {
        decx::utils::simd::xmm256_reg masked_vals;
        res._vi = _mm256_mullo_epi32(res._vi, LCG_A);
        res._vi = _mm256_add_epi32(res._vi, LCG_B);
        
        res._vi = _mm256_and_si256(res._vi, _mm256_set1_epi32(0x7fffffff));

        res._vf = _mm256_cvtepi32_ps(res._vi);
        res._vf = _mm256_mul_ps(res._vf, _mm256_set1_ps(scale_rcp));

        res._vf = _mm256_add_ps(res._vf, _mm256_set1_ps(min));

        masked_vals._vf = _mm256_and_ps(mask._vf, res._vf);
        _mm256_store_ps(dst + (proc_len_v1 / 8) * 8, masked_vals._vf);
    }
}



_THREAD_FUNCTION_ void decx::CPUK::
fill2D_rand_int32(int32_t* __restrict dst,
                  const uint2 proc_dims_v1,
                  const uint32_t pitch_v1,
                  const int32_t min, 
                  const int32_t max)
{
    if (max == min) return;

    decx::utils::simd::xmm256_reg seed_v8, mask, res;
    for (int i = 0; i < 8; ++i){
        seed_v8._arrui[i] = rand();
    }

    __m256i LCG_A = _mm256_set1_epi32(_RADN_LCG_A_);
    __m256i LCG_B = _mm256_set1_epi32(_RADN_LCG_B_);

    const uint32_t range = (uint32_t)abs(max - min);
    const float scale_rcp = (float)range / (float)(0x7fffffff);
    
    uint64_t dex = 0;
    const uint8_t _L = proc_dims_v1.x % 8;
    
    mask._vf = _mm256_setzero_ps();
    res._vf = seed_v8._vf;

    for (uint8_t i = 0; i < _L; ++i) mask._arrui[i] = 0xFFFFFFFF;

    for (uint32_t i = 0; i < proc_dims_v1.y; ++i){
        dex = i * pitch_v1;
        for (uint32_t j = 0; j < proc_dims_v1.x / 8; ++j){
            res._vi = _mm256_mullo_epi32(res._vi, LCG_A);
            res._vi = _mm256_add_epi32(res._vi, LCG_B);

            res._vi = _mm256_and_si256(res._vi, _mm256_set1_epi32(0x7fffffff));

            res._vf = _mm256_cvtepi32_ps(res._vi);
            res._vf = _mm256_mul_ps(res._vf, _mm256_set1_ps(scale_rcp));
            res._vi = _mm256_cvtps_epi32(res._vf);

            res._vi = _mm256_add_epi32(res._vi, _mm256_set1_epi32(min));
            _mm256_store_ps((float*)dst + dex, res._vf);
            dex += 8;
        }
        if (_L) {
            decx::utils::simd::xmm256_reg masked_vals;
            res._vi = _mm256_mullo_epi32(res._vi, LCG_A);
            res._vi = _mm256_add_epi32(res._vi, LCG_B);

            res._vi = _mm256_and_si256(res._vi, _mm256_set1_epi32(0x7fffffff));

            res._vf = _mm256_cvtepi32_ps(res._vi);
            res._vf = _mm256_mul_ps(res._vf, _mm256_set1_ps(scale_rcp));
            res._vi = _mm256_cvtps_epi32(res._vf);

            res._vi = _mm256_add_epi32(res._vi, _mm256_set1_epi32(min));
            masked_vals._vf = _mm256_and_ps(mask._vf, res._vf);
            _mm256_store_ps((float*)dst + dex, masked_vals._vf);
        }
    }
}



_THREAD_FUNCTION_ void decx::CPUK::
fill2D_rand_fp32(float* __restrict dst,
                  const uint2 proc_dims_v1,
                  const uint32_t pitch_v1,
                  const float min, 
                  const float max)
{
    if (max == min) return;

    decx::utils::simd::xmm256_reg seed_v8, mask, res;
    for (int i = 0; i < 8; ++i){
        seed_v8._arrui[i] = rand();
    }

    __m256i LCG_A = _mm256_set1_epi32(_RADN_LCG_A_);
    __m256i LCG_B = _mm256_set1_epi32(_RADN_LCG_B_);

    const uint32_t range = (uint32_t)abs(max - min);
    const float scale_rcp = (float)range / (float)(0x7fffffff);
    
    uint64_t dex = 0;
    const uint8_t _L = proc_dims_v1.x % 8;
    
    mask._vf = _mm256_setzero_ps();
    res._vf = seed_v8._vf;

    for (uint8_t i = 0; i < _L; ++i) mask._arrui[i] = 0xFFFFFFFF;

    for (uint32_t i = 0; i < proc_dims_v1.y; ++i){
        dex = i * pitch_v1;
        for (uint32_t j = 0; j < proc_dims_v1.x / 8; ++j){
            res._vi = _mm256_mullo_epi32(res._vi, LCG_A);
            res._vi = _mm256_add_epi32(res._vi, LCG_B);

            res._vi = _mm256_and_si256(res._vi, _mm256_set1_epi32(0x7fffffff));

            res._vf = _mm256_cvtepi32_ps(res._vi);
            res._vf = _mm256_mul_ps(res._vf, _mm256_set1_ps(scale_rcp));

            res._vf = _mm256_add_ps(res._vf, _mm256_set1_ps(min));
            _mm256_store_ps(dst + dex, res._vf);
            dex += 8;
        }
        if (_L) {
            decx::utils::simd::xmm256_reg masked_vals;
            res._vi = _mm256_mullo_epi32(res._vi, LCG_A);
            res._vi = _mm256_add_epi32(res._vi, LCG_B);

            res._vi = _mm256_and_si256(res._vi, _mm256_set1_epi32(0x7fffffff));

            res._vf = _mm256_cvtepi32_ps(res._vi);
            res._vf = _mm256_mul_ps(res._vf, _mm256_set1_ps(scale_rcp));

            res._vf = _mm256_add_ps(res._vf, _mm256_set1_ps(min));
            masked_vals._vf = _mm256_and_ps(mask._vf, res._vf);
            _mm256_store_ps(dst + dex, masked_vals._vf);
        }
    }
}
