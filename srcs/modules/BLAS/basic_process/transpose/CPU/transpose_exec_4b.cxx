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


#include "../../../../core/basic.h"
#include "transpose2D_config.h"
#include "transpose_exec.h"

decx::ResourceHandle decx::blas::g_cpu_transpose_4b_config;

namespace decx
{
namespace blas {
    namespace CPUK 
    {
        /**
        * <_LH> is in perspective of matrix 'src'. The height is not aligned (truncated instead), hence
        * load the data considering a not-4 ending. But storing is still aligned since it's related to the
        * width of matrix 'dst' and it's aligned to 4.
        */
        _THREAD_CALL_ static void transpose_block_4b_LH(const float* src, float* dst, const uint32_t _Wv4, const uint32_t _LW,
            const uint32_t pitchsrc_v1, const uint32_t pitchdst_v1, const uint32_t _LH);


        _THREAD_CALL_ static void transpose_block_4b(const float* src, float* dst, const uint2 proc_dims_v4, const uint32_t pitchsrc_v1,
            const uint32_t pitchdst_v1);

        
        _THREAD_FUNCTION_ static void transpose_4b_kernel(const float* src, float* dst, const decx::utils::_blocking2D_fmgrs* proc_dims_v4, 
            const uint32_t Wsrc_v8, const uint32_t Wdst_v1);
    }
}
}


_THREAD_CALL_ static void decx::blas::CPUK::
transpose_block_4b_LH(const float* __restrict src, 
                      float* __restrict dst, 
                      const uint32_t _Wv4, 
                      const uint32_t _LW, 
                      const uint32_t pitchsrc_v1,
                      const uint32_t pitchdst_v1, 
                      const uint32_t _LH)
{
    uint32_t dex_src = 0, dex_dst = 0;

    for (uint32_t i = 0; i < _Wv4; ++i) {
        __m128 regs[4] = { _mm_setzero_ps(), _mm_setzero_ps(),
                           _mm_setzero_ps(), _mm_setzero_ps()}, tmp[4];

        for (uint32_t j = 0; j < _LH; ++j) {
            regs[j] = _mm_load_ps(src + dex_src + pitchsrc_v1 * j);
        }

        _AVX_MM128_TRANSPOSE_4X4_(regs, tmp);

        if (i < _Wv4 - 1 || _LW == 0) {
            _mm_store_ps(dst + dex_dst, regs[0]);
            _mm_store_ps(dst + dex_dst + pitchdst_v1, regs[1]);
            _mm_store_ps(dst + dex_dst + (pitchdst_v1 << 1), regs[2]);
            _mm_store_ps(dst + dex_dst + (pitchdst_v1 << 1) + pitchdst_v1, regs[3]);
        }
        else {
            for (uint32_t j = 0; j < _LW; ++j) {
                _mm_store_ps(dst + dex_dst + pitchdst_v1 * j, regs[j]);
            }
        }

        dex_src += 4;
        dex_dst += (pitchdst_v1 << 2);
    }
}


_THREAD_CALL_ static void decx::blas::CPUK::
transpose_block_4b(const float* __restrict   src, 
                   float* __restrict         dst, 
                   const uint2               proc_dims_v1, 
                   const uint32_t            pitchsrc_v1,
                   const uint32_t            pitchdst_v1)
{
    uint32_t dex_src = 0, dex_dst = 0;

    const uint32_t _Hv4 = proc_dims_v1.y / 4;
    const uint32_t _LH = proc_dims_v1.y % 4;
    const uint32_t _Wv4 = decx::utils::ceil<uint32_t>(proc_dims_v1.x, 4);
    const uint32_t _LW = proc_dims_v1.x % 4;

    for (uint32_t i = 0; i < _Hv4; ++i) 
    {
        dex_src = i * (pitchsrc_v1 << 2);
        dex_dst = (i << 2);
        for (uint32_t j = 0; j < _Wv4; ++j) {
            __m128 regs[4], tmp[4];

            regs[0] = _mm_load_ps(src + dex_src);
            regs[1] = _mm_load_ps(src + dex_src + pitchsrc_v1);
            regs[2] = _mm_load_ps(src + dex_src + (pitchsrc_v1 << 1));
            regs[3] = _mm_load_ps(src + dex_src + (pitchsrc_v1 << 1) + pitchsrc_v1);

            _AVX_MM128_TRANSPOSE_4X4_(regs, tmp);

            if (j < _Wv4 - 1 || _LW == 0) {
                _mm_store_ps(dst + dex_dst, regs[0]);
                _mm_store_ps(dst + dex_dst + pitchdst_v1, regs[1]);
                _mm_store_ps(dst + dex_dst + (pitchdst_v1 << 1), regs[2]);
                _mm_store_ps(dst + dex_dst + (pitchdst_v1 << 1) + pitchdst_v1, regs[3]);
            }
            else {
                for (uint32_t k = 0; k < _LW; ++k) {
                    _mm_store_ps(dst + dex_dst + pitchdst_v1 * k, regs[k]);
                }
            }

            dex_src += 4;
            dex_dst += (pitchdst_v1 << 2);
        }
    }
    
    if (_LH) {
        dex_src = _Hv4 * (pitchsrc_v1 << 2);
        dex_dst = (_Hv4 << 2);
        decx::blas::CPUK::transpose_block_4b_LH(src + dex_src, dst + dex_dst, _Wv4, _LW,
            pitchsrc_v1, pitchdst_v1, _LH);
    }
}


_THREAD_FUNCTION_ static void decx::blas::CPUK::
transpose_4b_kernel(const float* __restrict                 src, 
                     float* __restrict                      dst, 
                     const decx::utils::_blocking2D_fmgrs*  proc_dims_config,
                     const uint32_t                         pitchsrc_v1, 
                     const uint32_t                         pitchdst_v1)
{
    uint64_t dex_src = 0, dex_dst = 0;

    for (uint32_t i = 0; i < proc_dims_config->_fmgrH.frag_num; ++i) 
    {
        dex_src = i * (proc_dims_config->_fmgrH.frag_len) * pitchsrc_v1;
        dex_dst = i * (proc_dims_config->_fmgrH.frag_len);

        uint2 proc_dims = make_uint2(proc_dims_config->_fmgrW.frag_len,
                                        i < proc_dims_config->_fmgrH.frag_num - 1 ?
                                        proc_dims_config->_fmgrH.frag_len : 
                                        proc_dims_config->_fmgrH.last_frag_len);

        for (uint32_t j = 0; j < proc_dims_config->_fmgrW.frag_num - 1; ++j) {
            decx::blas::CPUK::transpose_block_4b(src + dex_src, dst + dex_dst,
                proc_dims, pitchsrc_v1, pitchdst_v1);

            dex_src += proc_dims_config->_fmgrW.frag_len;
            dex_dst += (proc_dims_config->_fmgrW.frag_len * pitchdst_v1);
        }
        proc_dims.x = proc_dims_config->_fmgrW.last_frag_len;
        decx::blas::CPUK::transpose_block_4b(src + dex_src, dst + dex_dst,
            proc_dims, pitchsrc_v1, pitchdst_v1);
    }
}


void decx::blas::_cpu_transpose_config::transpose_4b_caller(const float* src, float* dst, 
    const uint32_t pitchsrc_v1, const uint32_t pitchdst_v1, decx::utils::_thread_arrange_1D* t1D) const
{
    const float* src_loc = src;
    float* dst_loc = dst;
    
    for (uint32_t i = 0; i < this->_thread_dist2D.y; ++i)
    {
        src_loc = src + i * this->_fmgr_H.frag_len * pitchsrc_v1;
        dst_loc = dst + i * this->_fmgr_H.frag_len;

        for (uint32_t j = 0; j < this->_thread_dist2D.x; ++j) 
        {
            t1D->_async_thread[i * this->_thread_dist2D.x + j] = decx::cpu::register_task_default(
                decx::blas::CPUK::transpose_4b_kernel, src_loc, dst_loc,
                &this->_blocking_configs.ptr[this->_thread_dist2D.x * i + j], pitchsrc_v1, pitchdst_v1);

            src_loc += this->_fmgr_W.frag_len;
            dst_loc += this->_fmgr_W.frag_len * pitchdst_v1;
        }
    }
    t1D->__sync_all_threads();
}
