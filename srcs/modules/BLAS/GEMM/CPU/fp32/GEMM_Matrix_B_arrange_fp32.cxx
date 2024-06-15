/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#include "GEMM_Matrix_B_arrange_fp32.h"


void decx::gemm::CPUK::
arrange_MatB_fp32_caller(const float*                srcB, 
                         float*                      dstB, 
                         const uint32_t              WsrcB,
                         const uint32_t              WtmpB,
                         const uint32_t              _eff_L_len,
                         const bool                  is_L8,
                         decx::utils::_thr_2D*       t2D,
                         decx::utils::frag_manager*  f_mgr)
{
    const float* local_ptr_src = srcB;
    float* local_ptr_dst = dstB;
    if (f_mgr->is_left) {
        for (int i = 0; i < t2D->total_thread - 1; ++i) {
            t2D->_async_thread[i] = decx::cpu::register_task_default(decx::gemm::CPUK::_sort_ST_MatB_fp32,
                local_ptr_src, local_ptr_dst, _eff_L_len, f_mgr->frag_len, WsrcB, WtmpB);

            local_ptr_src += 16 * f_mgr->frag_len;
            local_ptr_dst += WtmpB * f_mgr->frag_len;
        }
        if (is_L8) {
            // Fullfilling-method is used, hence, the lane that is 16 has to be decresed by one
            t2D->_async_thread[t2D->total_thread - 1] = decx::cpu::register_task_default(decx::gemm::CPUK::_sort_ST_MatB_fp32_L8,
                local_ptr_src, local_ptr_dst, _eff_L_len, f_mgr->frag_left_over - 1, WsrcB, WtmpB);
        }
        else {
            t2D->_async_thread[t2D->total_thread - 1] = decx::cpu::register_task_default(decx::gemm::CPUK::_sort_ST_MatB_fp32,
                local_ptr_src, local_ptr_dst, _eff_L_len, f_mgr->frag_left_over, WsrcB, WtmpB);
        }
    }
    else {
        for (int i = 0; i < t2D->total_thread - 1; ++i) {
            t2D->_async_thread[i] = decx::cpu::register_task_default(decx::gemm::CPUK::_sort_ST_MatB_fp32,
                local_ptr_src, local_ptr_dst, _eff_L_len, f_mgr->frag_len, WsrcB, WtmpB);

            local_ptr_src += 16 * f_mgr->frag_len;
            local_ptr_dst += WtmpB * f_mgr->frag_len;
        }
        if (is_L8) {
            t2D->_async_thread[t2D->total_thread - 1] = decx::cpu::register_task_default(decx::gemm::CPUK::_sort_ST_MatB_fp32_L8,
                local_ptr_src, local_ptr_dst, _eff_L_len, f_mgr->frag_len - 1, WsrcB, WtmpB);
        }
        else {
            // Fullfilling-method is used, hence, the lane that is 16 has to be decresed by one
            t2D->_async_thread[t2D->total_thread - 1] = decx::cpu::register_task_default(decx::gemm::CPUK::_sort_ST_MatB_fp32,
                local_ptr_src, local_ptr_dst, _eff_L_len, f_mgr->frag_len, WsrcB, WtmpB);
        }
    }

    t2D->__sync_all_threads();
}

