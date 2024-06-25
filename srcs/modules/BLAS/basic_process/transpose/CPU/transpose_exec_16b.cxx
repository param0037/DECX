/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#include "../../../../core/basic.h"
#include "transpose2D_config.h"
#include "transpose_exec.h"


decx::ResourceHandle decx::blas::g_cpu_transpose_16b_config;


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
        _THREAD_CALL_ static void transpose_block_16b_LH(const double* src, double* dst, const uint32_t _Wv4, const uint32_t _LW,
            const uint32_t pitchsrc_v1, const uint32_t pitchdst_v1, const uint32_t _LH);


        _THREAD_CALL_ static void transpose_block_16b(const double* src, double* dst, const uint2 proc_dims_v4, const uint32_t pitchsrc_v1,
            const uint32_t pitchdst_v1);

        
        _THREAD_FUNCTION_ static void transpose_16b_kernel(const double* src, double* dst, const decx::utils::_blocking2D_fmgrs* proc_dims_v4, 
            const uint32_t Wsrc_v8, const uint32_t Wdst_v1);
    }
}
}



_THREAD_CALL_ static void decx::blas::CPUK::
transpose_block_16b_LH(const double* __restrict src, 
                      double* __restrict dst, 
                      const uint32_t _Wv2, 
                      const uint32_t _LW, 
                      const uint32_t pitchsrc_v1,
                      const uint32_t pitchdst_v1, 
                      const uint32_t _LH)
{
    uint32_t dex_src = 0, dex_dst = 0;

    for (uint32_t i = 0; i < _Wv2; ++i) {
        __m256d recv[2] = { _mm256_setzero_pd(), _mm256_setzero_pd() }, stg[2];

        for (uint32_t j = 0; j < _LH; ++j) {
            recv[j] = _mm256_load_pd(src + dex_src + pitchsrc_v1 * j * 2);
        }

        _AVX_MM256_TRANSPOSE_2X2_(recv, stg);

        if (i < _Wv2 - 1 || _LW == 0) {
            _mm256_store_pd(dst + dex_dst, stg[0]);
            _mm256_store_pd(dst + dex_dst + pitchdst_v1 * 2, stg[1]);
        }
        else {
            for (uint32_t j = 0; j < _LW; ++j) {
                _mm256_store_pd(dst + dex_dst + pitchdst_v1 * j * 2, recv[j]);
            }
        }

        dex_src += 4;
        dex_dst += (pitchdst_v1 << 2);
    }
}



_THREAD_CALL_ static void decx::blas::CPUK::
transpose_block_16b(const double* __restrict   src, 
                   double* __restrict         dst, 
                   const uint2               proc_dims_v1, 
                   const uint32_t            pitchsrc_v1,
                   const uint32_t            pitchdst_v1)
{
    uint32_t dex_src = 0, dex_dst = 0;

    const uint32_t _Hv2 = proc_dims_v1.y / 2;
    const uint32_t _LH = proc_dims_v1.y % 2;
    const uint32_t _Wv2 = decx::utils::ceil<uint32_t>(proc_dims_v1.x, 2);
    const uint32_t _LW = proc_dims_v1.x % 2;

    for (uint32_t i = 0; i < _Hv2; ++i) 
    {
        dex_src = i * (pitchsrc_v1 << 2);
        dex_dst = (i << 2);
        for (uint32_t j = 0; j < _Wv2; ++j) {
            __m256d recv[4], stg[4];

            recv[0] = _mm256_load_pd(src + dex_src);
            recv[1] = _mm256_load_pd(src + dex_src + pitchsrc_v1 * 2);

            _AVX_MM256_TRANSPOSE_2X2_(recv, stg);

            if (j < _Wv2 - 1 || _LW == 0) {
                _mm256_store_pd(dst + dex_dst, stg[0]);
                _mm256_store_pd(dst + dex_dst + pitchdst_v1 * 2, stg[1]);
            }
            else {
                for (uint32_t k = 0; k < _LW; ++k) {
                    _mm256_store_pd(dst + dex_dst + pitchdst_v1 * k * 2, stg[k]);
                }
            }

            dex_src += 4;
            dex_dst += (pitchdst_v1 << 2);
        }
    }
    
    if (_LH) {
        dex_src = _Hv2 * (pitchsrc_v1 << 2);
        dex_dst = (_Hv2 << 2);
        decx::blas::CPUK::transpose_block_16b_LH(src + dex_src, dst + dex_dst, _Wv2, _LW,
            pitchsrc_v1, pitchdst_v1, _LH);
    }
}



_THREAD_FUNCTION_ static void decx::blas::CPUK::
transpose_16b_kernel(const double* __restrict                src, 
                     double* __restrict                      dst, 
                     const decx::utils::_blocking2D_fmgrs*  proc_dims_config,
                     const uint32_t                         pitchsrc_v1, 
                     const uint32_t                         pitchdst_v1)
{
    uint64_t dex_src = 0, dex_dst = 0;

    for (uint32_t i = 0; i < proc_dims_config->_fmgrH.frag_num; ++i) 
    {
        dex_src = i * (proc_dims_config->_fmgrH.frag_len) * pitchsrc_v1 * 2;
        dex_dst = i * (proc_dims_config->_fmgrH.frag_len * 2);

        uint2 proc_dims = make_uint2(proc_dims_config->_fmgrW.frag_len,
                                        i < proc_dims_config->_fmgrH.frag_num - 1 ?
                                        proc_dims_config->_fmgrH.frag_len : 
                                        proc_dims_config->_fmgrH.last_frag_len);

        for (uint32_t j = 0; j < proc_dims_config->_fmgrW.frag_num - 1; ++j) {
            decx::blas::CPUK::transpose_block_16b(src + dex_src, dst + dex_dst,
                proc_dims, pitchsrc_v1, pitchdst_v1);

            dex_src += proc_dims_config->_fmgrW.frag_len * 2;
            dex_dst += (proc_dims_config->_fmgrW.frag_len * pitchdst_v1 * 2);
        }
        proc_dims.x = proc_dims_config->_fmgrW.last_frag_len;
        decx::blas::CPUK::transpose_block_16b(src + dex_src, dst + dex_dst,
            proc_dims, pitchsrc_v1, pitchdst_v1);
    }
}



void decx::blas::_cpu_transpose_config::transpose_16b_caller(const double* src, double* dst,
    const uint32_t pitchsrc_v1, const uint32_t pitchdst_v1, decx::utils::_thread_arrange_1D* t1D) const
{
    const double* src_loc = src;
    double* dst_loc = dst;

    for (uint32_t i = 0; i < this->_thread_dist2D.y; ++i)
    {
        src_loc = src + i * this->_fmgr_H.frag_len * pitchsrc_v1 * 2;
        dst_loc = dst + i * this->_fmgr_H.frag_len * 2;

        for (uint32_t j = 0; j < this->_thread_dist2D.x; ++j)
        {
            t1D->_async_thread[i * this->_thread_dist2D.x + j] = decx::cpu::register_task_default(
                decx::blas::CPUK::transpose_16b_kernel, src_loc, dst_loc,
                &this->_blocking_configs.ptr[this->_thread_dist2D.x * i + j], pitchsrc_v1, pitchdst_v1);

            src_loc += this->_fmgr_W.frag_len * 2;
            dst_loc += this->_fmgr_W.frag_len * pitchdst_v1 * 2;
        }
    }
    t1D->__sync_all_threads();
}
