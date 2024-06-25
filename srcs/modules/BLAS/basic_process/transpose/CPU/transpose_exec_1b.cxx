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


decx::ResourceHandle decx::blas::g_cpu_transpose_1b_config;


namespace decx
{
namespace blas
{
    namespace CPUK 
    {
        /**
        * <_LH> is in perspective of matrix 'src'. The height is not aligned (truncated instead), hence
        * load the data considering a not-4 ending. But storing is still aligned since it's related to the
        * width of matrix 'dst' and it's aligned to 8.
        */
        _THREAD_CALL_ static void transpose_block_1b_LH(const uint64_t* src, uint64_t* dst, const uint32_t _Wv8, const uint32_t _LW,
            const uint32_t pitchsrc_v8, const uint32_t pitchdst_v8, const uint32_t _LH);


        _THREAD_CALL_ static void transpose_block_1b(const uint64_t* src, uint64_t* dst, const uint2 proc_dims_v1, const uint32_t pitchsrc_v8,
            const uint32_t pitchdst_v8);


        _THREAD_FUNCTION_ static void transpose_1b_kernel(const uint64_t* src, uint64_t* dst, const decx::utils::_blocking2D_fmgrs* proc_dims_v1,
            const uint32_t pitchsrc_v8, const uint32_t pitchdst_v8);
    }
}
}



_THREAD_CALL_ static void decx::blas::CPUK::
transpose_block_1b_LH(const uint64_t* __restrict src, 
                      uint64_t* __restrict dst, 
                      const uint32_t _Wv8, 
                      const uint32_t _LW, 
                      const uint32_t pitchsrc_v8,
                      const uint32_t pitchdst_v8, 
                      const uint32_t _LH)
{
    uint32_t dex_src = 0, dex_dst = 0;

    for (uint32_t i = 0; i < _Wv8; ++i) {
        uint64_t recv[8] = { 0, 0, 0, 0, 0, 0, 0, 0 }, stg[8];

        for (uint32_t j = 0; j < _LH; ++j) {
            recv[j] = src[dex_src + pitchsrc_v8 * j];
        }

        decx::blas::CPUK::block8x8_transpose_u8(recv, stg);

        if (i < _Wv8 - 1 || _LW == 0) {
            dst[dex_dst] = stg[0];
            dst[dex_dst + pitchdst_v8] = stg[1];
            dst[dex_dst + (pitchdst_v8 << 1)] = stg[2];
            dst[dex_dst + pitchdst_v8 * 3] = stg[3];
            dst[dex_dst + (pitchdst_v8 << 2)] = stg[4];
            dst[dex_dst + pitchdst_v8 * 5] = stg[5];
            dst[dex_dst + pitchdst_v8 * 6] = stg[6];
            dst[dex_dst + pitchdst_v8 * 7] = stg[7];
        }
        else {
            for (uint32_t j = 0; j < _LW; ++j) {
                dst[dex_dst + pitchdst_v8 * j] = stg[j];
            }
        }

        ++dex_src;
        dex_dst += (pitchdst_v8 << 3);
    }
}



_THREAD_CALL_ static void decx::blas::CPUK::
transpose_block_1b(const uint64_t* __restrict   src, 
                   uint64_t* __restrict         dst, 
                   const uint2                  proc_dims_v1, 
                   const uint32_t               pitchsrc_v8,
                   const uint32_t               pitchdst_v8)
{
    uint32_t dex_src = 0, dex_dst = 0;

    const uint32_t _Hv8 = proc_dims_v1.y / 8;
    const uint32_t _LH = proc_dims_v1.y % 8;
    const uint32_t _Wv8 = decx::utils::ceil<uint32_t>(proc_dims_v1.x, 8);
    const uint32_t _LW = proc_dims_v1.x % 8;

    for (uint32_t i = 0; i < _Hv8; ++i) 
    {
        dex_src = i * (pitchsrc_v8 << 3);
        dex_dst = i;
        for (uint32_t j = 0; j < _Wv8; ++j) {
            uint64_t recv[8], stg[8];

            recv[0] = src[dex_src];
            recv[1] = src[dex_src + pitchsrc_v8];
            recv[2] = src[dex_src + (pitchsrc_v8 << 1)];
            recv[3] = src[dex_src + pitchsrc_v8 * 3];
            recv[4] = src[dex_src + (pitchsrc_v8 << 2)];
            recv[5] = src[dex_src + pitchsrc_v8 * 5];
            recv[6] = src[dex_src + pitchsrc_v8 * 6];
            recv[7] = src[dex_src + pitchsrc_v8 * 7];

            decx::blas::CPUK::block8x8_transpose_u8(recv, stg);

            if (j < _Wv8 - 1 || _LW == 0) {
                dst[dex_dst] = stg[0];
                dst[dex_dst + pitchdst_v8] = stg[1];
                dst[dex_dst + (pitchdst_v8 << 1)] = stg[2];
                dst[dex_dst + pitchdst_v8 * 3] = stg[3];
                dst[dex_dst + (pitchdst_v8 << 2)] = stg[4];
                dst[dex_dst + pitchdst_v8 * 5] = stg[5];
                dst[dex_dst + pitchdst_v8 * 6] = stg[6];
                dst[dex_dst + pitchdst_v8 * 7] = stg[7];
            }
            else {
                for (uint32_t k = 0; k < _LW; ++k) {
                    dst[dex_dst + pitchdst_v8 * k] = stg[k];
                }
            }

            ++dex_src;
            dex_dst += (pitchdst_v8 << 3);
        }
    }
    
    if (_LH) {
        dex_src = _Hv8 * (pitchsrc_v8 << 3);
        dex_dst = _Hv8;
        decx::blas::CPUK::transpose_block_1b_LH(src + dex_src, dst + dex_dst, _Wv8, _LW,
            pitchsrc_v8, pitchdst_v8, _LH);
    }
}



_THREAD_FUNCTION_ static void decx::blas::CPUK::
transpose_1b_kernel(const uint64_t* __restrict                 src, 
                    uint64_t* __restrict                       dst, 
                    const decx::utils::_blocking2D_fmgrs*      proc_dims_config,
                    const uint32_t                             pitchsrc_v8, 
                    const uint32_t                             pitchdst_v8)
{
    uint64_t dex_src = 0, dex_dst = 0;

    for (uint32_t i = 0; i < proc_dims_config->_fmgrH.frag_num; ++i) 
    {
        dex_src = i * (proc_dims_config->_fmgrH.frag_len) * pitchsrc_v8;
        dex_dst = i * (proc_dims_config->_fmgrH.frag_len / 8);

        uint2 proc_dims = make_uint2(proc_dims_config->_fmgrW.frag_len,
                                        i < proc_dims_config->_fmgrH.frag_num - 1 ?
                                        proc_dims_config->_fmgrH.frag_len : 
                                        proc_dims_config->_fmgrH.last_frag_len);
        
        //printf("proc_dims.x : %d; .y : %d\n", proc_dims.x, proc_dims.y);

        for (uint32_t j = 0; j < proc_dims_config->_fmgrW.frag_num - 1; ++j) {
            decx::blas::CPUK::transpose_block_1b(src + dex_src, dst + dex_dst,
                proc_dims, pitchsrc_v8, pitchdst_v8);

            dex_src += proc_dims_config->_fmgrW.frag_len / 8;
            dex_dst += (proc_dims_config->_fmgrW.frag_len * pitchdst_v8);
        }
        proc_dims.x = proc_dims_config->_fmgrW.last_frag_len;
        decx::blas::CPUK::transpose_block_1b(src + dex_src, dst + dex_dst,
            proc_dims, pitchsrc_v8, pitchdst_v8);
    }
}



void decx::blas::_cpu_transpose_config::
transpose_1b_caller(const uint64_t* src, 
                    uint64_t* dst, 
                    const uint32_t pitchsrc_v8, 
                    const uint32_t pitchdst_v8, 
                    decx::utils::_thread_arrange_1D* t1D) const
{
    const uint64_t* src_loc = src;
    uint64_t* dst_loc = dst;
    
    for (uint32_t i = 0; i < this->_thread_dist2D.y; ++i)
    {
        src_loc = src + i * this->_fmgr_H.frag_len * pitchsrc_v8;
        dst_loc = dst + i * this->_fmgr_H.frag_len / 8;

        for (uint32_t j = 0; j < this->_thread_dist2D.x; ++j)
        {
            t1D->_async_thread[i * this->_thread_dist2D.x + j] = decx::cpu::register_task_default(
                decx::blas::CPUK::transpose_1b_kernel, src_loc, dst_loc,
                &this->_blocking_configs.ptr[this->_thread_dist2D.x * i + j], pitchsrc_v8, pitchdst_v8);

            src_loc += this->_fmgr_W.frag_len / 8;
            dst_loc += this->_fmgr_W.frag_len * pitchdst_v8;
        }
    }
    t1D->__sync_all_threads();
}
