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

#include "../common/filter2D_kernels.h"
#include "../../../../../../common/SIMD/x86_64/shf_mm256_fp32.h"


namespace decx
{
namespace dsp{
namespace CPUK
{
    /**
     * @brief Calculates 8-point parallel sliding window convolution. The 256 bit results are stored 
    */
    _THREAD_CALL_ static inline void conv2_fp32_spot_v8(const float* __restrict src, const float* kernel,
        float* __restrict dst, const uint2 kernel_dim, const uint32_t pitchsrc_v1, const uint32_t pitchkernel_v1);


    _THREAD_CALL_ static inline void conv2_fp32_BC_spot_v8(const float* __restrict src, const float* kernel,
        float* __restrict dst, const uint2 kernel_dim, const uint32_t pitchsrc_v1, const uint32_t pitchkernel_v1,
        const uint32_t row_id, const uint32_t Hsrc);


    _THREAD_CALL_ static inline void conv2_fp32_BR_spot_v8(const float* __restrict src, const float* kernel,
        float* __restrict dst, const uint2 kernel_dim, const uint32_t pitchsrc_v1, const uint32_t pitchkernel_v1,
        const uint32_t row_id, const uint32_t Hsrc);


    typedef void _conv2_spot_B_kernel (const float* __restrict, const float*,
        float* __restrict, const uint2, const uint32_t, const uint32_t, const uint32_t, const uint32_t);


    _THREAD_CALL_ static void conv2_fp32_block(const float* __restrict src, const float* kernel, float* __restrict dst,
        const uint2 ker_dim, const uint2 proc_dim_v8, const uint32_t pitchsrc_v1, const uint32_t pitchkernel_v1,
        const uint32_t pitchdst_v1);


    template <_conv2_spot_B_kernel _kernel>
    _THREAD_CALL_ static void conv2_fp32_B_block(const float* __restrict src, const float* kernel, float* __restrict dst,
        const uint2 ker_dim, const uint2 proc_dim_v8, const uint32_t pitchsrc_v1, const uint32_t pitchkernel_v1,
        const uint32_t pitchdst_v1, const uint32_t row_id, const uint32_t Hsrc);
}
}
}


_THREAD_CALL_ static void decx::dsp::CPUK::
conv2_fp32_spot_v8(const float* __restrict src,     const float* kernel,
                   float* __restrict dst,           const uint2 kernel_dim, 
                   const uint32_t pitchsrc_v1,      const uint32_t pitchkernel_v1)
{
    uint64_t dex_src = 0;
    uint32_t dex_ker = 0;
    __m256 _moving, _static, _k_v8;
    __m256 _accu = _mm256_setzero_ps();

    const uint32_t _workspace_len = kernel_dim.x - 1 + 8;
    const uint32_t _k_loop_W_v8 = kernel_dim.x / 8;
    const uint32_t _L_KW_v8 = _workspace_len & 7;

    for (uint32_t i = 0; i < kernel_dim.y; ++i)
    {
        dex_src = i * pitchsrc_v1;
        dex_ker = i * pitchkernel_v1;

        __m256 _tmp;
        _moving = _mm256_load_ps(src + dex_src);

        for (uint32_t j = 0; j < _k_loop_W_v8; ++j)
        {
            // Update the two registers
            if (j > 0)  _moving = _static;
            _static = _mm256_load_ps(src + dex_src + 8);

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

            dex_src += 8;
            dex_ker += 8;
        }

        if (_k_loop_W_v8) _moving = _static;
        _static = _mm256_load_ps(src + dex_src + 8);
        _k_v8 = _mm256_broadcast_ss(kernel + dex_ker);
        _accu = _mm256_fmadd_ps(_moving, _k_v8, _accu);
        ++dex_ker;

        for (uint32_t j = 0; j < _L_KW_v8; ++j) {
            _SHF_MM256_FP32_GENERAL_(j, _moving, _static, _tmp);
            _k_v8 = _mm256_broadcast_ss(kernel + dex_ker);
            _accu = _mm256_fmadd_ps(_moving, _k_v8, _accu);

            ++dex_ker;
        }
    }

    _mm256_store_ps(dst, _accu);
}



_THREAD_CALL_ static void decx::dsp::CPUK::
conv2_fp32_BC_spot_v8(const float* __restrict src,     const float* kernel,
                      float* __restrict dst,           const uint2 kernel_dim, 
                      const uint32_t pitchsrc_v1,      const uint32_t pitchkernel_v1,
                      const uint32_t row_id,           const uint32_t Hsrc)
{
    uint64_t dex_src = 0;
    uint32_t dex_ker = 0;
    __m256 _moving, _static, _k_v8;
    __m256 _accu = _mm256_setzero_ps();

    const uint32_t _workspace_len = kernel_dim.x - 1 + 8;
    const uint32_t _k_loop_W_v8 = kernel_dim.x / 8;
    const uint32_t _L_KW_v8 = _workspace_len & 7;

    const uint32_t _halv_Hker = kernel_dim.y >> 1;

    for (uint32_t i = 0; i < kernel_dim.y; ++i)
    {
        dex_ker = i * pitchkernel_v1;

        int32_t this_row = (int32_t)(row_id + i) - (int32_t)_halv_Hker;
        if (this_row > -1 && this_row < Hsrc) 
        {
            dex_src = this_row * (uint64_t)pitchsrc_v1;
            __m256 _tmp;
            _moving = _mm256_load_ps(src + dex_src);

            for (uint32_t j = 0; j < _k_loop_W_v8; ++j)
            {
                // Update the two registers
                if (j > 0)  _moving = _static;
                _static = _mm256_load_ps(src + dex_src + 8);

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

                dex_src += 8;
                dex_ker += 8;
            }

            if (_k_loop_W_v8) _moving = _static;
            _static = _mm256_load_ps(src + dex_src + 8);
            _k_v8 = _mm256_broadcast_ss(kernel + dex_ker);
            _accu = _mm256_fmadd_ps(_moving, _k_v8, _accu);
            ++dex_ker;

            for (uint32_t j = 0; j < _L_KW_v8; ++j) {
                _SHF_MM256_FP32_GENERAL_(j, _moving, _static, _tmp);
                _k_v8 = _mm256_broadcast_ss(kernel + dex_ker);
                _accu = _mm256_fmadd_ps(_moving, _k_v8, _accu);

                ++dex_ker;
            }
        }
    }

    _mm256_store_ps(dst, _accu);
}




_THREAD_CALL_ static void decx::dsp::CPUK::
conv2_fp32_BR_spot_v8(const float* __restrict src,     const float* kernel,
                      float* __restrict dst,           const uint2 kernel_dim, 
                      const uint32_t pitchsrc_v1,      const uint32_t pitchkernel_v1,
                      const uint32_t row_id,           const uint32_t Hsrc)
{
    uint64_t dex_src = 0;
    uint32_t dex_ker = 0;
    __m256 _moving, _static, _k_v8;
    __m256 _accu = _mm256_setzero_ps();

    const uint32_t _workspace_len = kernel_dim.x - 1 + 8;
    const uint32_t _k_loop_W_v8 = kernel_dim.x / 8;
    const uint32_t _L_KW_v8 = _workspace_len & 7;

    const uint32_t _halv_Hker = kernel_dim.y >> 1;

    for (uint32_t i = 0; i < kernel_dim.y; ++i)
    {
        dex_ker = i * pitchkernel_v1;

        int32_t this_row = (int32_t)(row_id + i) - (int32_t)_halv_Hker;
        uint32_t _targeted_row = 0;

        if (this_row < 0) {
            _targeted_row = 0 - this_row;
        }
        else if (this_row > Hsrc - 1) {
            _targeted_row = (Hsrc << 1) - this_row - 2;
        }
        else {
            _targeted_row = this_row;
        }
        dex_src = _targeted_row * (uint64_t)pitchsrc_v1;
        __m256 _tmp;
        _moving = _mm256_load_ps(src + dex_src);

        for (uint32_t j = 0; j < _k_loop_W_v8; ++j)
        {
            // Update the two registers
            if (j > 0)  _moving = _static;
            _static = _mm256_load_ps(src + dex_src + 8);

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

            dex_src += 8;
            dex_ker += 8;
        }

        if (_k_loop_W_v8) _moving = _static;
        _static = _mm256_load_ps(src + dex_src + 8);
        _k_v8 = _mm256_broadcast_ss(kernel + dex_ker);
        _accu = _mm256_fmadd_ps(_moving, _k_v8, _accu);
        ++dex_ker;

        for (uint32_t j = 0; j < _L_KW_v8; ++j) {
            _SHF_MM256_FP32_GENERAL_(j, _moving, _static, _tmp);
            _k_v8 = _mm256_broadcast_ss(kernel + dex_ker);
            _accu = _mm256_fmadd_ps(_moving, _k_v8, _accu);

            ++dex_ker;
        }
    }

    _mm256_store_ps(dst, _accu);
}




_THREAD_CALL_ static void decx::dsp::CPUK::
conv2_fp32_block(const float* __restrict src,       const float* kernel, 
                 float* __restrict dst,             const uint2 ker_dim, 
                 const uint2 proc_dim_v8,           const uint32_t pitchsrc_v1, 
                 const uint32_t pitchkernel_v1,     const uint32_t pitchdst_v1)
{
    uint64_t dex_src = 0, dex_dst = 0;
    uint32_t dex_ker = 0;

    for (uint32_t i = 0; i < proc_dim_v8.y; ++i)
    {
        dex_src = i * pitchsrc_v1;
        dex_dst = i * pitchdst_v1;

        for (uint32_t j = 0; j < proc_dim_v8.x; ++j)
        {
            decx::dsp::CPUK::conv2_fp32_spot_v8(src + dex_src, kernel, dst + dex_dst,
                ker_dim, pitchsrc_v1, pitchkernel_v1);

            dex_src += 8;
            dex_dst += 8;
        }
    }
}



template <decx::dsp::CPUK::_conv2_spot_B_kernel _kernel>
_THREAD_CALL_ static void decx::dsp::CPUK::
conv2_fp32_B_block(const float* __restrict src,       const float* kernel, 
                   float* __restrict dst,             const uint2 ker_dim, 
                   const uint2 proc_dim_v8,           const uint32_t pitchsrc_v1, 
                   const uint32_t pitchkernel_v1,     const uint32_t pitchdst_v1,
                   const uint32_t start_row_id,       const uint32_t Hsrc)
{
    uint64_t dex_src = 0, dex_dst = 0;
    uint32_t dex_ker = 0;

    for (uint32_t i = 0; i < proc_dim_v8.y; ++i)
    {
        dex_src = 0;
        dex_dst = i * pitchdst_v1;

        for (uint32_t j = 0; j < proc_dim_v8.x; ++j)
        {
            _kernel(src + dex_src, kernel, dst + dex_dst,
                ker_dim, pitchsrc_v1, pitchkernel_v1, start_row_id + i, Hsrc);

            dex_src += 8;
            dex_dst += 8;
        }
    }
}


_THREAD_FUNCTION_ void decx::dsp::CPUK::
conv2_fp32_kernel(const float* __restrict src,                          const float* kernel, 
                  float* __restrict dst,                                const uint2 kernel_dim, 
                  const decx::utils::_blocking2D_fmgrs* _block_conf,    const uint32_t pitchsrc_v1,   
                  const uint32_t pitchkernel_v1,                        const uint32_t pitchdst_v1)
{
    uint64_t dex_src = 0, dex_dst = 0;
    uint32_t dex_kernel = 0;

    const uint32_t& blockH = _block_conf->_fmgrH.frag_len;
    const uint32_t& blockW = _block_conf->_fmgrW.frag_len;

    for (int32_t i = 0; i < (int32_t)_block_conf->_fmgrH.frag_num; ++i)
    {
        dex_src = i * blockH * pitchsrc_v1;
        dex_dst = i * blockH * pitchdst_v1;

        uint2 block_dims_v8 = make_uint2(blockW, blockH);

        for (int32_t j = 0; j < (int32_t)_block_conf->_fmgrW.frag_num - 1; ++j)
        {
            decx::dsp::CPUK::conv2_fp32_block(src + dex_src, kernel, dst + dex_dst, 
                kernel_dim, block_dims_v8, pitchsrc_v1, pitchkernel_v1, pitchdst_v1);

            dex_src += blockW * 8;
            dex_dst += blockW * 8;
        }
        block_dims_v8.x = _block_conf->_fmgrW.last_frag_len;
        decx::dsp::CPUK::conv2_fp32_block(src + dex_src, kernel, dst + dex_dst, 
                kernel_dim, block_dims_v8, pitchsrc_v1, pitchkernel_v1, pitchdst_v1);
    }
}



_THREAD_FUNCTION_ void decx::dsp::CPUK::
conv2_BC_fp32_kernel(const float* __restrict src,                          const float* kernel, 
                     float* __restrict dst,                                const uint2 kernel_dim, 
                     const decx::utils::_blocking2D_fmgrs* _block_conf,    const uint32_t pitchsrc_v1,   
                     const uint32_t pitchkernel_v1,                        const uint32_t pitchdst_v1,
                     const uint32_t start_row_id,                          const uint32_t Hsrc)
{
    uint64_t dex_src = 0, dex_dst = 0;
    uint32_t dex_kernel = 0;

    const uint32_t& blockH = _block_conf->_fmgrH.frag_len;
    const uint32_t& blockW = _block_conf->_fmgrW.frag_len;

    for (int32_t i = 0; i < (int32_t)_block_conf->_fmgrH.frag_num; ++i)
    {
        dex_src = 0;
        dex_dst = i * blockH * pitchdst_v1;

        uint2 block_dims_v8 = make_uint2(blockW, blockH);

        for (int32_t j = 0; j < (int32_t)_block_conf->_fmgrW.frag_num - 1; ++j)
        {
            decx::dsp::CPUK::conv2_fp32_B_block<decx::dsp::CPUK::conv2_fp32_BC_spot_v8>(
                src + dex_src,      kernel,         dst + dex_dst,
                kernel_dim,         block_dims_v8,  pitchsrc_v1, 
                pitchkernel_v1,     pitchdst_v1,    start_row_id + i * blockH, Hsrc);

            dex_src += blockW * 8;
            dex_dst += blockW * 8;
        }
        block_dims_v8.x = _block_conf->_fmgrW.last_frag_len;
        decx::dsp::CPUK::conv2_fp32_B_block<decx::dsp::CPUK::conv2_fp32_BC_spot_v8>(
            src + dex_src,      kernel,         dst + dex_dst,
            kernel_dim,         block_dims_v8,  pitchsrc_v1, 
            pitchkernel_v1,     pitchdst_v1,    start_row_id + i * blockH, Hsrc);
    }
}



_THREAD_FUNCTION_ void decx::dsp::CPUK::
conv2_BR_fp32_kernel(const float* __restrict src,                          const float* kernel, 
                     float* __restrict dst,                                const uint2 kernel_dim, 
                     const decx::utils::_blocking2D_fmgrs* _block_conf,    const uint32_t pitchsrc_v1,   
                     const uint32_t pitchkernel_v1,                        const uint32_t pitchdst_v1,
                     const uint32_t start_row_id,                          const uint32_t Hsrc)
{
    uint64_t dex_src = 0, dex_dst = 0;
    uint32_t dex_kernel = 0;

    const uint32_t& blockH = _block_conf->_fmgrH.frag_len;
    const uint32_t& blockW = _block_conf->_fmgrW.frag_len;

    for (int32_t i = 0; i < (int32_t)_block_conf->_fmgrH.frag_num; ++i)
    {
        dex_src = 0;
        dex_dst = i * blockH * pitchdst_v1;

        uint2 block_dims_v8 = make_uint2(blockW, blockH);

        for (int32_t j = 0; j < (int32_t)_block_conf->_fmgrW.frag_num - 1; ++j)
        {
            decx::dsp::CPUK::conv2_fp32_B_block<decx::dsp::CPUK::conv2_fp32_BR_spot_v8>(
                src + dex_src,      kernel,         dst + dex_dst,
                kernel_dim,         block_dims_v8,  pitchsrc_v1, 
                pitchkernel_v1,     pitchdst_v1,    start_row_id + i * blockH, Hsrc);

            dex_src += blockW * 8;
            dex_dst += blockW * 8;
        }
        block_dims_v8.x = _block_conf->_fmgrW.last_frag_len;
        decx::dsp::CPUK::conv2_fp32_B_block<decx::dsp::CPUK::conv2_fp32_BR_spot_v8>(
            src + dex_src,      kernel,         dst + dex_dst,
            kernel_dim,         block_dims_v8,  pitchsrc_v1, 
            pitchkernel_v1,     pitchdst_v1,    start_row_id + i * blockH, Hsrc);
    }
}
