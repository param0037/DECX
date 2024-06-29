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


#include "../matrix_B_arrange.h"


namespace decx
{
namespace blas {
    namespace CPUK 
    {
        template <uint32_t block_W_v16, uint32_t block_H>
        _THREAD_CALL_ static void _matrix_B_arrange_fp32_block(const float* __restrict, float* __restrict, 
            const uint32_t, const uint32_t);


        _THREAD_CALL_ static void _matrix_B_arrange_fp32_block_var(const float* __restrict, float* __restrict,
            const uint32_t, const uint32_t, const uint2);


        template <uint32_t block_W_v8, uint32_t block_H>
        _THREAD_FUNCTION_ static void _matrix_B_arrange_fp32_exec(const float* __restrict, float* __restrict,
            const uint2, const uint32_t, const uint32_t);
    }
}
}


template <uint32_t block_W_v16, uint32_t block_H>
_THREAD_CALL_ static void decx::blas::CPUK::
_matrix_B_arrange_fp32_block(const float* __restrict    src, 
                             float* __restrict          dst, 
                             const uint32_t             pitchsrc_v1,
                             const uint32_t             Llen)
{
    uint64_t dex_src = 0, dex_dst = 0;
    for (uint32_t i = 0; i < block_H; ++i) 
    {
        dex_src = i * pitchsrc_v1;
        dex_dst = i * 16;
        for (uint32_t j = 0; j < block_W_v16; ++j) {
            _mm256_store_ps(dst + dex_dst, _mm256_load_ps(src + dex_src));
            _mm256_store_ps(dst + dex_dst + 8, _mm256_load_ps(src + dex_src + 8));

            dex_src += 16;
            dex_dst += Llen * 16;
        }
    }
}


_THREAD_CALL_ static void decx::blas::CPUK::
_matrix_B_arrange_fp32_block_var(const float* __restrict    src, 
                             float* __restrict              dst, 
                             const uint32_t                 pitchsrc_v1,
                             const uint32_t                 Llen,
                             const uint2                    proc_dims_v8)
{
    uint64_t dex_src = 0, dex_dst = 0;
    for (uint32_t i = 0; i < proc_dims_v8.y; ++i)
    {
        dex_src = i * pitchsrc_v1;
        dex_dst = i * 16;
        for (uint32_t j = 0; j < proc_dims_v8.x / 2; ++j) {
            _mm256_store_ps(dst + dex_dst, _mm256_load_ps(src + dex_src));
            _mm256_store_ps(dst + dex_dst + 8, _mm256_load_ps(src + dex_src + 8));

            dex_src += 16;
            dex_dst += Llen * 16;
        }
        if (proc_dims_v8.x & 1) {
            _mm256_store_ps(dst + dex_dst, _mm256_load_ps(src + dex_src));
            _mm256_store_ps(dst + dex_dst + 8, _mm256_setzero_ps());
        }
    }
}


// e.g.
// block_W = 16 (minimal) floats = 64 byte = cache line size
template <uint32_t block_W_v16, uint32_t block_H>
_THREAD_FUNCTION_ static void decx::blas::CPUK::
_matrix_B_arrange_fp32_exec(const float* __restrict src,            // pointer of original matrix B (input)
                            float* __restrict       dst,            // pointer of arranged matrix B (output)
                            const uint2             proc_dims_v8,   // processing area for a single thread task
                            const uint32_t          pitchsrc_v1,    // pitch of original matrix B
                            const uint32_t          Llen)           // Length of L dimension = # of lanes in the width of arranged matrix B
{
    uint64_t dex_src = 0, dex_dst = 0;
    constexpr uint32_t block_W_v8 = block_W_v16 * 2;

    const uint32_t ptime_w = proc_dims_v8.x / block_W_v8;
    const uint32_t _LW = proc_dims_v8.x % block_W_v8;

    const uint32_t ptime_h = decx::utils::ceil<uint32_t>(proc_dims_v8.y, block_H);
    const uint32_t _LH = proc_dims_v8.y % block_H;

    for (uint32_t i = 0; i < ptime_h; ++i) 
    {
        dex_src = i * pitchsrc_v1 * block_H;
        dex_dst = i * block_H * 16;
        if (i < ptime_h - 1 || _LH == 0) 
        {
            for (uint32_t j = 0; j < ptime_w; ++j) {
                decx::blas::CPUK::_matrix_B_arrange_fp32_block<block_W_v16, block_H>(src + dex_src,
                    dst + dex_dst, pitchsrc_v1, Llen);
                dex_src += block_W_v8 * 8;
                dex_dst += Llen * block_W_v16 * 16;
            }
            if (_LW) {
                decx::blas::CPUK::_matrix_B_arrange_fp32_block_var(src + dex_src,
                    dst + dex_dst, pitchsrc_v1, Llen, make_uint2(_LW, block_H));
            }
        }
        else {
            for (uint32_t j = 0; j < ptime_w; ++j) {
                decx::blas::CPUK::_matrix_B_arrange_fp32_block_var(src + dex_src,
                    dst + dex_dst, pitchsrc_v1, Llen, make_uint2(block_W_v8, _LH));
                dex_src += block_W_v8 * 8;
                dex_dst += Llen * block_W_v16 * 16;
            }
            if (_LW) {
                decx::blas::CPUK::_matrix_B_arrange_fp32_block_var(src + dex_src,
                    dst + dex_dst, pitchsrc_v1, Llen, make_uint2(_LW, _LH));
            }
        }
    }
}


void decx::blas::matrix_B_arrange_fp32(const float*                     src, 
                                       float*                           dst, 
                                       const uint32_t                   pitchsrc_v1,
                                       const uint32_t                   Llen, 
                                       const decx::utils::frag_manager* _fmgr_WH,   // Aligned to 8 on width
                                       decx::utils::_thr_2D*            t2D)
{
    const float* loc_src = NULL;
    float* loc_dst = NULL;

    for (uint32_t i = 0; i < t2D->thread_h; ++i) 
    {
        uint2 proc_dims;
        proc_dims = make_uint2(_fmgr_WH[0].frag_len, 
                               i < t2D->thread_h - 1 ? _fmgr_WH[1].frag_len
                                                     : _fmgr_WH[1].last_frag_len);

        loc_src = src + i * _fmgr_WH[1].frag_len * pitchsrc_v1;
        loc_dst = dst + i * _fmgr_WH[1].frag_len * 16;
        for (uint32_t j = 0; j < t2D->thread_w - 1; ++j) 
        {
            t2D->_async_thread[i * t2D->thread_w + j] = decx::cpu::register_task_default(
                decx::blas::CPUK::_matrix_B_arrange_fp32_exec<2, 16>,
                loc_src, loc_dst, proc_dims, pitchsrc_v1, Llen);
            loc_src += _fmgr_WH[0].frag_len * 8;
            loc_dst += _fmgr_WH[0].frag_len * Llen * 8;
        }
        const uint32_t _LW = _fmgr_WH[0].is_left ? _fmgr_WH[0].frag_left_over : _fmgr_WH[0].frag_len;

        proc_dims.x = _LW;
        t2D->_async_thread[(i+1)*t2D->thread_w - 1] = decx::cpu::register_task_default(
            decx::blas::CPUK::_matrix_B_arrange_fp32_exec<2, 16>,
            loc_src, loc_dst, proc_dims, pitchsrc_v1, Llen);
    }

    t2D->__sync_all_threads();
}
