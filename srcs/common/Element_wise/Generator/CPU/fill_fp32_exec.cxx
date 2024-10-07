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


_THREAD_FUNCTION_ void decx::CPUK::                 
fill1D_constant_fp32(float* __restrict dst,
                     const uint64_t proc_len_v1,
                     const float _constant)
{
    decx::utils::simd::xmm256_reg _constant_v;
    decx::utils::simd::xmm256_reg mask;
    const uint8_t _L = proc_len_v1 % 8;
    
    mask._vf = _mm256_setzero_ps();
    for (uint8_t i = 0; i < _L; ++i) mask._arrui[i] = 0xFFFFFFFF;

    _constant_v._vf = _mm256_broadcast_ss(&_constant);

    for (uint64_t i = 0; i < proc_len_v1 / 8; ++i){
        _mm256_store_ps(dst + i * 8, _constant_v._vf);
    }                                               
    if (_L) {                                       
        decx::utils::simd::xmm256_reg masked_vals;  
        masked_vals._vf = _mm256_and_ps(mask._vf, _constant_v._vf);
        _mm256_store_ps(dst + (proc_len_v1 / 8) * 8, masked_vals._vf);
    }
}


_THREAD_FUNCTION_ void decx::CPUK::
fill1D_constant_fp64(double* __restrict dst,
                     const uint64_t proc_len_v1,
                     const double _constant)
{
    decx::utils::simd::xmm256_reg _constant_v;
    decx::utils::simd::xmm256_reg mask;
    const uint8_t _L = proc_len_v1 % 4;
    
    mask._vf = _mm256_setzero_ps();
    for (uint8_t i = 0; i < _L; ++i) mask._arrull[i] = 0xFFFFFFFFFFFFFFFF;

    _constant_v._vd = _mm256_broadcast_sd(&_constant);

    for (uint64_t i = 0; i < proc_len_v1 / 4; ++i){
        _mm256_store_pd(dst + i * 4, _constant_v._vd);
    }
    if (_L) {
        decx::utils::simd::xmm256_reg masked_vals;
        masked_vals._vf = _mm256_and_ps(mask._vf, _constant_v._vf);
        _mm256_store_pd(dst + (proc_len_v1 / 4) * 4, masked_vals._vd);
    }
}


_THREAD_FUNCTION_ void decx::CPUK::                 
fill1D_constant_int32(int32_t* __restrict dst,
                      const uint64_t proc_len_v1,
                      const int32_t _constant)
{
    decx::utils::simd::xmm256_reg _constant_v;
    decx::utils::simd::xmm256_reg mask;
    const uint8_t _L = proc_len_v1 % 8;
    
    mask._vf = _mm256_setzero_ps();
    for (uint8_t i = 0; i < _L; ++i) mask._arrui[i] = 0xFFFFFFFF;

    _constant_v._vf = _mm256_broadcast_ss((float*)(&_constant));

    for (uint64_t i = 0; i < proc_len_v1 / 8; ++i){
        _mm256_store_ps((float*)dst + i * 8, _constant_v._vf);
    }
    if (_L) {                                       
        decx::utils::simd::xmm256_reg masked_vals;  
        masked_vals._vf = _mm256_and_ps(mask._vf, _constant_v._vf);
        _mm256_store_ps((float*)dst + (proc_len_v1 / 8) * 8, masked_vals._vf);
    }
}

// ------------------------------------------ 2D ------------------------------------------

_THREAD_FUNCTION_ void decx::CPUK::
fill2D_constant_fp32(float* __restrict dst,
                     const uint2 proc_dims_v1,
                     const uint32_t pitch_v1,
                     const float _constant)
{
    decx::utils::simd::xmm256_reg _constant_v;
    _constant_v._vf = _mm256_broadcast_ss(&_constant);
    
    uint64_t dex = 0;
    decx::utils::simd::xmm256_reg mask;
    const uint8_t _L = proc_dims_v1.x % 8;
    
    mask._vf = _mm256_setzero_ps();
    for (uint8_t i = 0; i < _L; ++i) mask._arrui[i] = 0xFFFFFFFF;

    for (uint32_t i = 0; i < proc_dims_v1.y; ++i){
        dex = i * pitch_v1;
        for (uint32_t j = 0; j < proc_dims_v1.x / 8; ++j){
            _mm256_store_ps(dst + dex, _constant_v._vf);
            dex += 8;
        }
        if (_L) {
            decx::utils::simd::xmm256_reg masked_vals;
            masked_vals._vf = _mm256_and_ps(mask._vf, _constant_v._vf);
            _mm256_store_ps(dst + dex, masked_vals._vf);
        }
    }
}


_THREAD_FUNCTION_ void decx::CPUK::
fill2D_constant_fp64(double* __restrict dst,
                     const uint2 proc_dims_v1,
                     const uint32_t pitch_v1,
                     const double _constant)
{
    decx::utils::simd::xmm256_reg _constant_v;
    
    _constant_v._vd = _mm256_broadcast_sd(&_constant);
    
    uint64_t dex = 0;
    decx::utils::simd::xmm256_reg mask;
    const uint8_t _L = proc_dims_v1.x % 4;
    
    mask._vf = _mm256_setzero_ps();
    for (uint8_t i = 0; i < _L; ++i) mask._arrull[i] = 0xFFFFFFFFFFFFFFFF;

    for (uint32_t i = 0; i < proc_dims_v1.y; ++i){
        dex = i * pitch_v1;
        for (uint32_t j = 0; j < proc_dims_v1.x / 4; ++j){
            _mm256_store_pd(dst + dex, _constant_v._vd);
            dex += 4;
        }
        if (_L) {
            decx::utils::simd::xmm256_reg masked_vals;
            masked_vals._vf = _mm256_and_ps(mask._vf, _constant_v._vf);
            _mm256_store_pd(dst + dex, masked_vals._vd);
        }
    }
}


_THREAD_FUNCTION_ void decx::CPUK::
fill2D_constant_int32(int32_t* __restrict dst,
                     const uint2 proc_dims_v1,
                     const uint32_t pitch_v1,
                     const int32_t _constant)
{
    decx::utils::simd::xmm256_reg _constant_v;
    
    _constant_v._vf = _mm256_broadcast_ss((float*)(&_constant));
    
    uint64_t dex = 0;
    decx::utils::simd::xmm256_reg mask;
    const uint8_t _L = proc_dims_v1.x % 8;
    
    mask._vf = _mm256_setzero_ps();
    for (uint8_t i = 0; i < _L; ++i) mask._arrui[i] = 0xFFFFFFFF;

    for (uint32_t i = 0; i < proc_dims_v1.y; ++i){
        dex = i * pitch_v1;
        for (uint32_t j = 0; j < proc_dims_v1.x / 8; ++j){
            _mm256_store_ps((float*)dst + dex, _constant_v._vf);
            dex += 8;
        }
        if (_L) {
            decx::utils::simd::xmm256_reg masked_vals;
            masked_vals._vf = _mm256_and_ps(mask._vf, _constant_v._vf);
            _mm256_store_ps((float*)dst + dex, masked_vals._vf);
        }
    }
}

