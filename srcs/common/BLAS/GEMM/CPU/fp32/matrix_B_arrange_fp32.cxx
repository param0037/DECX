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

#if defined(__x86_64__) || defined(__i386__)

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
#endif


#if defined(__aarch64__) || defined(__arm__)

template <uint32_t block_W_v8, uint32_t block_H>
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
        dex_dst = i * 8;
        for (uint32_t j = 0; j < block_W_v8; ++j) {
            vst1q_f32_x2(dst + dex_dst, vld1q_f32_x2(src + dex_src));

            dex_src += 8;
            dex_dst += Llen * 8;
        }
    }
}


_THREAD_CALL_ static void decx::blas::CPUK::
_matrix_B_arrange_fp32_block_var(const float* __restrict    src, 
                             float* __restrict              dst, 
                             const uint32_t                 pitchsrc_v1,
                             const uint32_t                 Llen,
                             const uint2                    proc_dims_v4)
{
    uint64_t dex_src = 0, dex_dst = 0;
    for (uint32_t i = 0; i < proc_dims_v4.y; ++i)
    {
        dex_src = i * pitchsrc_v1;
        dex_dst = i * 8;
        for (uint32_t j = 0; j < proc_dims_v4.x / 2; ++j) {
            vst1q_f32_x2(dst + dex_dst, vld1q_f32_x2(src + dex_src));

            dex_src += 8;
            dex_dst += Llen * 8;
        }
        if (proc_dims_v4.x & 1) {
            vst1q_f32(dst + dex_dst, vld1q_f32(src + dex_src));
            uint32x4_t vzeros;
            vzeros = veorq_u32(vzeros, vzeros);
            vst1q_f32(dst + dex_dst + 4, vreinterpretq_f32_u32(vzeros));
        }
    }
}
#endif


// e.g.
// block_W = 16 (minimal) floats = 64 byte = cache line size
template <uint32_t block_W_v2x, uint32_t block_H>
_THREAD_FUNCTION_ static void decx::blas::CPUK::
_matrix_B_arrange_fp32_exec(const float* __restrict src,            // pointer of original matrix B (input)
                            float* __restrict       dst,            // pointer of arranged matrix B (output)
                            const uint2             proc_dims_v,   // processing area for a single thread task
                            const uint32_t          pitchsrc_v1,    // pitch of original matrix B
                            const uint32_t          Llen)           // Length of L dimension = # of lanes in the width of arranged matrix B
{
#if defined(__x86_64__) || defined(__i386__)
    constexpr uint32_t _alignment = 8;
#endif
#if defined(__aarch64__) || defined(__arm__)
    constexpr uint32_t _alignment = 4;
#endif
    uint64_t dex_src = 0, dex_dst = 0;
    constexpr uint32_t block_W_v = block_W_v2x * 2;

    const uint32_t ptime_w = proc_dims_v.x / block_W_v;
    const uint32_t _LW = proc_dims_v.x % block_W_v;

    const uint32_t ptime_h = decx::utils::idiv_ceil<uint32_t>(proc_dims_v.y, block_H);
    const uint32_t _LH = proc_dims_v.y % block_H;

    for (uint32_t i = 0; i < ptime_h; ++i) 
    {
        dex_src = i * pitchsrc_v1 * block_H;
        dex_dst = i * block_H * _alignment * 2;
        if (i < ptime_h - 1 || _LH == 0) 
        {
            for (uint32_t j = 0; j < ptime_w; ++j) {
                decx::blas::CPUK::_matrix_B_arrange_fp32_block<block_W_v2x, block_H>(src + dex_src,
                    dst + dex_dst, pitchsrc_v1, Llen);
                dex_src += block_W_v * _alignment;
                dex_dst += Llen * block_W_v2x * _alignment * 2;
            }
            if (_LW) {
                decx::blas::CPUK::_matrix_B_arrange_fp32_block_var(src + dex_src,
                    dst + dex_dst, pitchsrc_v1, Llen, make_uint2(_LW, block_H));
            }
        }
        else {
            for (uint32_t j = 0; j < ptime_w; ++j) {
                decx::blas::CPUK::_matrix_B_arrange_fp32_block_var(src + dex_src,
                    dst + dex_dst, pitchsrc_v1, Llen, make_uint2(block_W_v, _LH));
                dex_src += block_W_v * _alignment;
                dex_dst += Llen * block_W_v * _alignment;
            }
            if (_LW) {
                decx::blas::CPUK::_matrix_B_arrange_fp32_block_var(src + dex_src,
                    dst + dex_dst, pitchsrc_v1, Llen, make_uint2(_LW, _LH));
            }
        }
    }
}


int32_t decx::blas::
matrix_B_arrange_fp32(const float*                     src, 
                      float*                           dst, 
                      const uint32_t                   pitchsrc_v1,
                      const uint32_t                   Llen, 
                      const decx::utils::frag_manager* _fmgr_WH,   // Aligned to 8 on width
                      decx::utils::ComputeLoadsMgr2D*  t2D)
{
    int32_t rval = 0;
    constexpr uint32_t _alignment = decx::utils::simd::GetCPUSimdAlignBytes() / sizeof(float);

    const float* loc_src = NULL;
    float* loc_dst = NULL;
    
    uint32_t task_id = 0;
    for (uint32_t i = 0; i < t2D->GetDist().y; ++i) 
    {
        loc_src = src + i * _fmgr_WH[1].GetFragLen() * pitchsrc_v1;
        loc_dst = dst + i * _fmgr_WH[1].GetFragLen() * _alignment * 2;
        for (uint32_t j = 0; j < t2D->GetDist().x; ++j) 
        {
            const uint2 proc_dims = make_uint2(_fmgr_WH[0].GetFragLenById(j), 
                                               _fmgr_WH[1].GetFragLenById(i));

            rval |= t2D->AppendTask(task_id, decx::blas::CPUK::_matrix_B_arrange_fp32_exec<2, 16>,
                PACK_CPY(loc_src), PACK_CPY(loc_dst), PACK_CPY(proc_dims), PACK_REF(pitchsrc_v1), PACK_REF(Llen));

            loc_src += _fmgr_WH[0].GetFragLen() * _alignment;
            loc_dst += _fmgr_WH[0].GetFragLen() * Llen * _alignment;
            ++task_id;
        }
    }

    rval |= t2D->RunAll();
    rval |= t2D->SynchronizeAll();
    rval |= t2D->ClearAll();
    return rval;
}