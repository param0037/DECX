/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/


#include "conv2_fp32_BC_MK_exec.h"



// r9, r12
_THREAD_CALL_ 
void decx::conv::CPUK::_conv2_rN_BC_MK_fp32_ST_top(float* __restrict src, float* __restrict tmp_src, float* __restrict kernel, float* __restrict dst, const uint2 proc_dim, 
                                            const decx::_C2_MK32* conv2_mk_props)
{
    const uint2 start = make_uint2(conv2_mk_props->ker_dims.y / 2, conv2_mk_props->ker_dims.x / 2);
    const uint2 end = make_uint2(proc_dim.y + conv2_mk_props->ker_dims.y - 1, (proc_dim.x << 3) + start.y);

    decx::utils::frag_manager f_mgrH, f_mgrW;
    decx::utils::frag_manager_gen_from_fragLen(&f_mgrH, proc_dim.y, _BLOCKED_CONV2_FP32_H_);
    decx::utils::frag_manager_gen_from_fragLen(&f_mgrW, proc_dim.x, _BLOCKED_CONV2_FP32_W_);

    for (int i = 0; i < conv2_mk_props->channel_size; ++i) {
        decx::_general_copy2D_BC<float>(src + i * conv2_mk_props->page_size_dst, tmp_src, start, end, conv2_mk_props->Wdst << 3, conv2_mk_props->Wsrc << 3);

        decx::conv::CPUK::_conv2_rN_fp32_ST_unconfigured(tmp_src, kernel + i * conv2_mk_props->page_size_ker,
            dst + i * conv2_mk_props->page_size_dst,
            proc_dim, conv2_mk_props->ker_dims, conv2_mk_props->Wsrc, conv2_mk_props->Wdst, conv2_mk_props->reg_WL, 
            &f_mgrH, &f_mgrW, conv2_mk_props->_loop);
    }
}



_THREAD_CALL_ 
void decx::conv::CPUK::_conv2_rN_BC_MK_fp32_ST_mid(float* __restrict src, float* __restrict tmp_src, float* __restrict kernel, float* __restrict dst, 
                                            const uint2 proc_dim, const decx::_C2_MK32* conv2_mk_props)
{
    const uint2 start = make_uint2(0, conv2_mk_props->ker_dims.x / 2);
    const uint2 end = make_uint2(proc_dim.y + conv2_mk_props->ker_dims.y - 1, (proc_dim.x << 3) + start.y);

    decx::utils::frag_manager f_mgrH, f_mgrW;
    decx::utils::frag_manager_gen_from_fragLen(&f_mgrH, proc_dim.y, _BLOCKED_CONV2_FP32_H_);
    decx::utils::frag_manager_gen_from_fragLen(&f_mgrW, proc_dim.x, _BLOCKED_CONV2_FP32_W_);

    for (int i = 0; i < conv2_mk_props->channel_size; ++i) {
        decx::_general_copy2D_BC<float>(src + i * conv2_mk_props->page_size_dst, tmp_src, start, end, conv2_mk_props->Wdst << 3, conv2_mk_props->Wsrc << 3);

        decx::conv::CPUK::_conv2_rN_fp32_ST_unconfigured(tmp_src, kernel + i * conv2_mk_props->page_size_ker,
            dst + i * conv2_mk_props->page_size_dst,
            proc_dim, conv2_mk_props->ker_dims, conv2_mk_props->Wsrc, conv2_mk_props->Wdst, conv2_mk_props->reg_WL, 
            &f_mgrH, &f_mgrW, conv2_mk_props->_loop);
    }
}



_THREAD_CALL_ 
void decx::conv::CPUK::_conv2_rN_BC_MK_fp32_ST_bottom(float* __restrict src, float* __restrict tmp_src, float* __restrict kernel, float* __restrict dst, 
                                            const uint2 proc_dim, const decx::_C2_MK32* conv2_mk_props)
{
    const uint2 start = make_uint2(0, conv2_mk_props->ker_dims.x / 2);
    const uint2 end = make_uint2(proc_dim.y + conv2_mk_props->ker_dims.y / 2, (proc_dim.x << 3) + start.y);

    decx::utils::frag_manager f_mgrH, f_mgrW;
    decx::utils::frag_manager_gen_from_fragLen(&f_mgrH, proc_dim.y, _BLOCKED_CONV2_FP32_H_);
    decx::utils::frag_manager_gen_from_fragLen(&f_mgrW, proc_dim.x, _BLOCKED_CONV2_FP32_W_);

    for (int i = 0; i < conv2_mk_props->channel_size; ++i) {
        decx::_general_copy2D_BC<float>(src + i * conv2_mk_props->page_size_dst, tmp_src, start, end, conv2_mk_props->Wdst << 3, conv2_mk_props->Wsrc << 3);

        decx::conv::CPUK::_conv2_rN_fp32_ST_unconfigured(tmp_src, kernel + i * conv2_mk_props->page_size_ker,
            dst + i * conv2_mk_props->page_size_dst,
            proc_dim, conv2_mk_props->ker_dims, conv2_mk_props->Wsrc, conv2_mk_props->Wdst, conv2_mk_props->reg_WL, 
            &f_mgrH, &f_mgrW, conv2_mk_props->_loop);
    }
}




// r1, r4
_THREAD_CALL_ 
void decx::conv::CPUK::_conv2_r1_r4_BC_MK_fp32_ST_top(float* __restrict src, float* __restrict tmp_src, float* __restrict kernel, float* __restrict dst, 
                                            const uint2 proc_dim, const decx::_C2_MK32* conv2_mk_props)
{
    const uint2 start = make_uint2(conv2_mk_props->ker_dims.y / 2, conv2_mk_props->ker_dims.x / 2);
    const uint2 end = make_uint2(proc_dim.y + conv2_mk_props->ker_dims.y - 1, (proc_dim.x << 3) + start.y);

    decx::utils::frag_manager f_mgrH, f_mgrW;
    decx::utils::frag_manager_gen_from_fragLen(&f_mgrH, proc_dim.y, _BLOCKED_CONV2_FP32_H_);
    decx::utils::frag_manager_gen_from_fragLen(&f_mgrW, proc_dim.x, _BLOCKED_CONV2_FP32_W_);

    for (int i = 0; i < conv2_mk_props->channel_size; ++i) {
        decx::_general_copy2D_BC<float>(src + i * conv2_mk_props->page_size_dst, 
            tmp_src, start, end, conv2_mk_props->Wdst << 3, conv2_mk_props->Wsrc << 3);

        decx::conv::CPUK::_conv2_r1_r4_fp32_ST_unconfigured(tmp_src, kernel + i * conv2_mk_props->page_size_ker,
        dst + i * conv2_mk_props->page_size_dst,
            proc_dim, conv2_mk_props->ker_dims, conv2_mk_props->Wsrc, conv2_mk_props->Wdst, &f_mgrH, &f_mgrW);
    }
}



_THREAD_CALL_ 
void decx::conv::CPUK::_conv2_r1_r4_BC_MK_fp32_ST_mid(float* __restrict src, float* __restrict tmp_src, float* __restrict kernel, float* __restrict dst, 
                                            const uint2 proc_dim, const decx::_C2_MK32* conv2_mk_props)
{
    const uint2 start = make_uint2(0, conv2_mk_props->ker_dims.x / 2);
    const uint2 end = make_uint2(proc_dim.y + conv2_mk_props->ker_dims.y - 1, (proc_dim.x << 3) + start.y);

    decx::utils::frag_manager f_mgrH, f_mgrW;
    decx::utils::frag_manager_gen_from_fragLen(&f_mgrH, proc_dim.y, _BLOCKED_CONV2_FP32_H_);
    decx::utils::frag_manager_gen_from_fragLen(&f_mgrW, proc_dim.x, _BLOCKED_CONV2_FP32_W_);

    for (int i = 0; i < conv2_mk_props->channel_size; ++i) {
        decx::_general_copy2D_BC<float>(src + i * conv2_mk_props->page_size_dst, 
            tmp_src, start, end, conv2_mk_props->Wdst << 3, conv2_mk_props->Wsrc << 3);

        decx::conv::CPUK::_conv2_r1_r4_fp32_ST_unconfigured(tmp_src, kernel + i * conv2_mk_props->page_size_ker,
        dst + i * conv2_mk_props->page_size_dst,
            proc_dim, conv2_mk_props->ker_dims, conv2_mk_props->Wsrc, conv2_mk_props->Wdst, &f_mgrH, &f_mgrW);
    }
}



_THREAD_CALL_ 
void decx::conv::CPUK::_conv2_r1_r4_BC_MK_fp32_ST_bottom(float* __restrict src, float* __restrict tmp_src, float* __restrict kernel, float* __restrict dst, 
                                            const uint2 proc_dim, const decx::_C2_MK32* conv2_mk_props)
{
    const uint2 start = make_uint2(0, conv2_mk_props->ker_dims.x / 2);
    const uint2 end = make_uint2(proc_dim.y + conv2_mk_props->ker_dims.y / 2, (proc_dim.x << 3) + start.y);

    decx::utils::frag_manager f_mgrH, f_mgrW;
    decx::utils::frag_manager_gen_from_fragLen(&f_mgrH, proc_dim.y, _BLOCKED_CONV2_FP32_H_);
    decx::utils::frag_manager_gen_from_fragLen(&f_mgrW, proc_dim.x, _BLOCKED_CONV2_FP32_W_);

    for (int i = 0; i < conv2_mk_props->channel_size; ++i) {
        decx::_general_copy2D_BC<float>(src + i * conv2_mk_props->page_size_dst, 
            tmp_src, start, end, conv2_mk_props->Wdst << 3, conv2_mk_props->Wsrc << 3);

        decx::conv::CPUK::_conv2_r1_r4_fp32_ST_unconfigured(tmp_src, kernel + i * conv2_mk_props->page_size_ker,
        dst + i * conv2_mk_props->page_size_dst,
            proc_dim, conv2_mk_props->ker_dims, conv2_mk_props->Wsrc, conv2_mk_props->Wdst, &f_mgrH, &f_mgrW);
    }
}




void decx::conv::_conv2_rN_BC_MK_fp32_caller(float* src,                          float* tmp_src, 
                                       float* kernel,                       float* dst, 
                                       const uint2 proc_dim,                decx::utils::_thr_1D* t1D,
                                       decx::_C2_MK32* conv2_props)
{
    float* tmp_src_ptr = src, * tmp_dst_ptr = dst, *tmp_cache_src = tmp_src;
    const size_t frag_cache_src = (conv2_props->f_mgr->frag_len + conv2_props->ker_dims.y - 1) * (conv2_props->Wsrc << 3);
    const size_t frag_dst_size = conv2_props->f_mgr->frag_len * (conv2_props->Wdst << 3);
    
    t1D->_async_thread[0] = decx::cpu::register_task(&decx::thread_pool, decx::conv::CPUK::_conv2_rN_BC_MK_fp32_ST_top,
        tmp_src_ptr, tmp_cache_src, kernel, tmp_dst_ptr,
        make_uint2(proc_dim.x, conv2_props->f_mgr->frag_len), conv2_props);

    tmp_src_ptr += (conv2_props->f_mgr->frag_len - conv2_props->ker_dims.y / 2) * (conv2_props->Wdst << 3);
    tmp_cache_src += frag_cache_src;
    tmp_dst_ptr += frag_dst_size;

    for (int i = 1; i < t1D->total_thread - 1; ++i)
    {
        t1D->_async_thread[i] = decx::cpu::register_task(&decx::thread_pool, decx::conv::CPUK::_conv2_rN_BC_MK_fp32_ST_mid,
            tmp_src_ptr, tmp_cache_src, kernel, tmp_dst_ptr,
            make_uint2(proc_dim.x, conv2_props->f_mgr->frag_len), conv2_props);

        tmp_src_ptr += conv2_props->f_mgr->frag_len * (conv2_props->Wdst << 3);
        tmp_cache_src += frag_cache_src;
        tmp_dst_ptr += frag_dst_size;
    }

    if (conv2_props->f_mgr->frag_left_over != 0) {
        t1D->_async_thread[t1D->total_thread - 1] = decx::cpu::register_task(&decx::thread_pool, decx::conv::CPUK::_conv2_rN_BC_MK_fp32_ST_bottom,
            tmp_src_ptr, tmp_cache_src, kernel, tmp_dst_ptr, make_uint2(proc_dim.x, conv2_props->f_mgr->frag_left_over),
            conv2_props);
    }
    else {
        t1D->_async_thread[t1D->total_thread - 1] = decx::cpu::register_task(&decx::thread_pool, decx::conv::CPUK::_conv2_rN_BC_MK_fp32_ST_bottom,
            tmp_src_ptr, tmp_cache_src, kernel, tmp_dst_ptr, make_uint2(proc_dim.x, conv2_props->f_mgr->frag_len),
            conv2_props);
    }
    
    t1D->__sync_all_threads();
}




void decx::conv::_conv2_r1_r4_BC_MK_fp32_caller(float* src,                          float* tmp_src, 
                                          float* kernel,                       float* dst, 
                                          const uint2 proc_dim,                decx::utils::_thr_1D* t1D,
                                          decx::_C2_MK32* conv2_props)
{
    float* tmp_src_ptr = src, * tmp_dst_ptr = dst, *tmp_cache_src = tmp_src;
    const size_t frag_cache_src = (conv2_props->f_mgr->frag_len + conv2_props->ker_dims.y - 1) * (conv2_props->Wsrc << 3);
    const size_t frag_dst_size = conv2_props->f_mgr->frag_len * (conv2_props->Wdst << 3);
    
    t1D->_async_thread[0] = decx::cpu::register_task(&decx::thread_pool, decx::conv::CPUK::_conv2_r1_r4_BC_MK_fp32_ST_top,
        tmp_src_ptr, tmp_cache_src, kernel, tmp_dst_ptr,
        make_uint2(proc_dim.x, conv2_props->f_mgr->frag_len), conv2_props);

    tmp_src_ptr += (conv2_props->f_mgr->frag_len - conv2_props->ker_dims.y / 2) * (conv2_props->Wdst << 3);
    tmp_cache_src += frag_cache_src;
    tmp_dst_ptr += frag_dst_size;

    for (int i = 1; i < t1D->total_thread - 1; ++i)
    {
        t1D->_async_thread[i] = decx::cpu::register_task(&decx::thread_pool, decx::conv::CPUK::_conv2_r1_r4_BC_MK_fp32_ST_mid,
            tmp_src_ptr, tmp_cache_src, kernel, tmp_dst_ptr,
            make_uint2(proc_dim.x, conv2_props->f_mgr->frag_len), conv2_props);

        tmp_src_ptr += conv2_props->f_mgr->frag_len * (conv2_props->Wdst << 3);
        tmp_cache_src += frag_cache_src;
        tmp_dst_ptr += frag_dst_size;
    }

    if (conv2_props->f_mgr->frag_left_over != 0) {
        t1D->_async_thread[t1D->total_thread - 1] = decx::cpu::register_task(&decx::thread_pool, decx::conv::CPUK::_conv2_r1_r4_BC_MK_fp32_ST_bottom,
            tmp_src_ptr, tmp_cache_src, kernel, tmp_dst_ptr, make_uint2(proc_dim.x, conv2_props->f_mgr->frag_left_over),
            conv2_props);
    }
    else {
        t1D->_async_thread[t1D->total_thread - 1] = decx::cpu::register_task(&decx::thread_pool, decx::conv::CPUK::_conv2_r1_r4_BC_MK_fp32_ST_bottom,
            tmp_src_ptr, tmp_cache_src, kernel, tmp_dst_ptr, make_uint2(proc_dim.x, conv2_props->f_mgr->frag_len),
            conv2_props);
    }
    
    t1D->__sync_all_threads();
}