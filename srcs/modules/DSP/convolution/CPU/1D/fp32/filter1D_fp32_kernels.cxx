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

#include "../common/filter1D_kernels.h"
#include "../../../../../../common/SIMD/x86_64/shf_mm256_fp32.h"


namespace decx
{
namespace dsp{
namespace CPUK
{
    /**
     * @brief Calculates 8-point parallel sliding window convolution. The 256 bit results are stored 
    */
    _THREAD_CALL_ static inline void conv1_fp32_spot_v8(const float* __restrict src, const float* kernel,
        float* __restrict dst, const uint32_t kernel_len);


    _THREAD_CALL_ static inline void conv1_fp32_BC_spot_v8(const float* __restrict src, const float* kernel,
        float* __restrict dst, const uint2 kernel_dim, const uint32_t pitchsrc_v1, const uint32_t pitchkernel_v1,
        const uint32_t row_id, const uint32_t Hsrc);


    _THREAD_CALL_ static inline void conv1_fp32_BR_spot_v8(const float* __restrict src, const float* kernel,
        float* __restrict dst, const uint2 kernel_dim, const uint32_t pitchsrc_v1, const uint32_t pitchkernel_v1,
        const uint32_t row_id, const uint32_t Hsrc);
}
}
}


_THREAD_CALL_ static inline void 
decx::dsp::CPUK::conv1_fp32_spot_v8(const float* __restrict src, 
                                    const float* kernel,
                                    float* __restrict dst, 
                                    const uint32_t kernel_len)
{
    uint32_t dex_ker = 0;
    __m256 _moving, _static, _k_v8;
    __m256 _accu = _mm256_setzero_ps();

    const uint32_t _workspace_len = kernel_len - 1 + 8;
    const uint32_t _k_loop_W_v8 = kernel_len / 8;
    const uint32_t _L_KW_v8 = _workspace_len & 7;

    __m256 _tmp;
    _moving = _mm256_load_ps(src);

    for (uint32_t j = 0; j < _k_loop_W_v8; ++j)
    {
        // Update the two registers
        if (j > 0)  _moving = _static;
        _static = _mm256_load_ps(src + (j + 1) * 8);

        _k_v8 = _mm256_broadcast_ss(kernel + dex_ker);
        _accu = _mm256_fmadd_ps(_moving, _k_v8, _accu);     // 0-th

        _SHF_MM256_FP32_SHF_1_(_moving, _static, _tmp);     // 1-st
        _k_v8 = _mm256_broadcast_ss(kernel + dex_ker + 1);
        _accu = _mm256_fmadd_ps(_moving, _k_v8, _accu);

        _SHF_MM256_FP32_SHF_2_(_moving, _static, _tmp);     // 2-nd
        _k_v8 = _mm256_broadcast_ss(kernel + dex_ker + 2);
        _accu = _mm256_fmadd_ps(_moving, _k_v8, _accu);

        _SHF_MM256_FP32_SHF_3_(_moving, _static, _tmp);     // 3-rd
        _k_v8 = _mm256_broadcast_ss(kernel + dex_ker + 3);
        _accu = _mm256_fmadd_ps(_moving, _k_v8, _accu);

        _SHF_MM256_FP32_SHF_4_(_moving, _static, _tmp);     // 4-th
        _k_v8 = _mm256_broadcast_ss(kernel + dex_ker + 4);
        _accu = _mm256_fmadd_ps(_moving, _k_v8, _accu);

        _SHF_MM256_FP32_SHF_5_(_moving, _static, _tmp);     // 5-th
        _k_v8 = _mm256_broadcast_ss(kernel + dex_ker + 5);
        _accu = _mm256_fmadd_ps(_moving, _k_v8, _accu);

        _SHF_MM256_FP32_SHF_6_(_moving, _static, _tmp);     // 6-th
        _k_v8 = _mm256_broadcast_ss(kernel + dex_ker + 6);
        _accu = _mm256_fmadd_ps(_moving, _k_v8, _accu);

        _SHF_MM256_FP32_SHF_7_(_moving, _static, _tmp);     // 7-th
        _k_v8 = _mm256_broadcast_ss(kernel + dex_ker + 7);
        _accu = _mm256_fmadd_ps(_moving, _k_v8, _accu);

        dex_ker += 8;
    }

    if (_k_loop_W_v8) _moving = _static;
    _static = _mm256_load_ps(src + (_k_loop_W_v8 + 1) * 8);
    _k_v8 = _mm256_broadcast_ss(kernel + dex_ker);
    _accu = _mm256_fmadd_ps(_moving, _k_v8, _accu);
    ++dex_ker;

    for (uint32_t j = 0; j < _L_KW_v8; ++j) {
        _SHF_MM256_FP32_GENERAL_(j, _moving, _static, _tmp);
        _k_v8 = _mm256_broadcast_ss(kernel + dex_ker);
        _accu = _mm256_fmadd_ps(_moving, _k_v8, _accu);

        ++dex_ker;
    }

    _mm256_store_ps(dst, _accu);
}


_THREAD_FUNCTION_ void decx::dsp::CPUK::
conv1_fp32_kernel(const float* __restrict   src,  
                  const float*              kernel, 
                  float* __restrict         dst,
                  const uint32_t            kernel_len, 
                  const uint32_t            proc_len_v1)
{
    uint64_t dex_src = 0, dex_dst = 0;
    uint32_t dex_ker = 0;

    for (uint32_t j = 0; j < decx::utils::ceil<uint32_t>(proc_len_v1, 8); ++j)
    {
        decx::dsp::CPUK::conv1_fp32_spot_v8(src + dex_src, kernel, dst + dex_dst, kernel_len);

        dex_src += 8;
        dex_dst += 8;
    }
}
