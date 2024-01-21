/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#include "conv2_fp64_BC_MK_exec.h"



// r9, r12
_THREAD_CALL_
void decx::conv::CPUK::_conv2_rN_BC_MK_fp32_ST_top_rw2(double* __restrict src, double* __restrict tmp_src, double* __restrict kernel, double* __restrict dst, const uint2 proc_dim,
    const decx::_C2_MK64* conv2_mk_props)
{
    const uint2 start = make_uint2(conv2_mk_props->ker_dims.y / 2, conv2_mk_props->ker_dims.x / 2);
    const uint2 end = make_uint2(proc_dim.y + conv2_mk_props->ker_dims.y - 1, (proc_dim.x << 2) + start.y);

    decx::utils::frag_manager f_mgrH, f_mgrW;
    decx::utils::frag_manager_gen_from_fragLen(&f_mgrH, proc_dim.y, _BLOCKED_CONV2_FP64_H_);
    decx::utils::frag_manager_gen_from_fragLen(&f_mgrW, proc_dim.x, _BLOCKED_CONV2_FP64_W_);

    for (int i = 0; i < conv2_mk_props->channel_size; ++i) {
        decx::_general_copy2D_BC<double>(src + i * conv2_mk_props->page_size_dst, tmp_src, start, end, conv2_mk_props->Wdst << 2, conv2_mk_props->Wsrc << 2);

        decx::conv::CPUK::_conv2_rN_fp64_ST_rw2_unconfigured(tmp_src, kernel + i * conv2_mk_props->page_size_ker,
            dst + i * conv2_mk_props->page_size_dst, proc_dim, conv2_mk_props->ker_dims, conv2_mk_props->Wsrc, 
            conv2_mk_props->Wdst, &f_mgrH, &f_mgrW, conv2_mk_props->_loop);
    }
}



_THREAD_CALL_
void decx::conv::CPUK::_conv2_rN_BC_MK_fp32_ST_mid_rw2(double* __restrict src, double* __restrict tmp_src, double* __restrict kernel, double* __restrict dst,
    const uint2 proc_dim, const decx::_C2_MK64* conv2_mk_props)
{
    const uint2 start = make_uint2(0, conv2_mk_props->ker_dims.x / 2);
    const uint2 end = make_uint2(proc_dim.y + conv2_mk_props->ker_dims.y - 1, (proc_dim.x << 2) + start.y);

    decx::utils::frag_manager f_mgrH, f_mgrW;
    decx::utils::frag_manager_gen_from_fragLen(&f_mgrH, proc_dim.y, _BLOCKED_CONV2_FP64_H_);
    decx::utils::frag_manager_gen_from_fragLen(&f_mgrW, proc_dim.x, _BLOCKED_CONV2_FP64_W_);

    for (int i = 0; i < conv2_mk_props->channel_size; ++i) {
        decx::_general_copy2D_BC<double>(src + i * conv2_mk_props->page_size_dst, tmp_src, start, end, conv2_mk_props->Wdst << 2, conv2_mk_props->Wsrc << 2);

        decx::conv::CPUK::_conv2_rN_fp64_ST_rw2_unconfigured(tmp_src, kernel + i * conv2_mk_props->page_size_ker,
            dst + i * conv2_mk_props->page_size_dst, proc_dim, conv2_mk_props->ker_dims, conv2_mk_props->Wsrc, 
            conv2_mk_props->Wdst, &f_mgrH, &f_mgrW, conv2_mk_props->_loop);
    }
}



_THREAD_CALL_
void decx::conv::CPUK::_conv2_rN_BC_MK_fp32_ST_bottom_rw2(double* __restrict src, double* __restrict tmp_src, double* __restrict kernel, double* __restrict dst,
    const uint2 proc_dim, const decx::_C2_MK64* conv2_mk_props)
{
    const uint2 start = make_uint2(0, conv2_mk_props->ker_dims.x / 2);
    const uint2 end = make_uint2(proc_dim.y + conv2_mk_props->ker_dims.y / 2, (proc_dim.x << 2) + start.y);

    decx::utils::frag_manager f_mgrH, f_mgrW;
    decx::utils::frag_manager_gen_from_fragLen(&f_mgrH, proc_dim.y, _BLOCKED_CONV2_FP64_H_);
    decx::utils::frag_manager_gen_from_fragLen(&f_mgrW, proc_dim.x, _BLOCKED_CONV2_FP64_W_);

    for (int i = 0; i < conv2_mk_props->channel_size; ++i) {
        decx::_general_copy2D_BC<double>(src + i * conv2_mk_props->page_size_dst, tmp_src, start, end, conv2_mk_props->Wdst << 2, conv2_mk_props->Wsrc << 2);

        decx::conv::CPUK::_conv2_rN_fp64_ST_rw2_unconfigured(tmp_src, kernel + i * conv2_mk_props->page_size_ker,
            dst + i * conv2_mk_props->page_size_dst, proc_dim, conv2_mk_props->ker_dims, conv2_mk_props->Wsrc, 
            conv2_mk_props->Wdst, &f_mgrH, &f_mgrW, conv2_mk_props->_loop);
    }
}



_THREAD_CALL_
void decx::conv::CPUK::_conv2_rN_BC_MK_fp32_ST_top_rw4(double* __restrict src, double* __restrict tmp_src, double* __restrict kernel, double* __restrict dst, const uint2 proc_dim,
    const decx::_C2_MK64* conv2_mk_props)
{
    const uint2 start = make_uint2(conv2_mk_props->ker_dims.y / 2, conv2_mk_props->ker_dims.x / 2);
    const uint2 end = make_uint2(proc_dim.y + conv2_mk_props->ker_dims.y - 1, (proc_dim.x << 2) + start.y);

    decx::utils::frag_manager f_mgrH, f_mgrW;
    decx::utils::frag_manager_gen_from_fragLen(&f_mgrH, proc_dim.y, _BLOCKED_CONV2_FP64_H_);
    decx::utils::frag_manager_gen_from_fragLen(&f_mgrW, proc_dim.x, _BLOCKED_CONV2_FP64_W_);

    for (int i = 0; i < conv2_mk_props->channel_size; ++i) {
        decx::_general_copy2D_BC<double>(src + i * conv2_mk_props->page_size_dst, tmp_src, start, end, conv2_mk_props->Wdst << 2, conv2_mk_props->Wsrc << 2);

        decx::conv::CPUK::_conv2_rN_fp64_ST_rw4_unconfigured(tmp_src, kernel + i * conv2_mk_props->page_size_ker,
            dst + i * conv2_mk_props->page_size_dst, proc_dim, conv2_mk_props->ker_dims, conv2_mk_props->Wsrc,
            conv2_mk_props->Wdst, &f_mgrH, &f_mgrW, conv2_mk_props->_loop);
    }
}



_THREAD_CALL_
void decx::conv::CPUK::_conv2_rN_BC_MK_fp32_ST_mid_rw4(double* __restrict src, double* __restrict tmp_src, double* __restrict kernel, double* __restrict dst,
    const uint2 proc_dim, const decx::_C2_MK64* conv2_mk_props)
{
    const uint2 start = make_uint2(0, conv2_mk_props->ker_dims.x / 2);
    const uint2 end = make_uint2(proc_dim.y + conv2_mk_props->ker_dims.y - 1, (proc_dim.x << 2) + start.y);

    decx::utils::frag_manager f_mgrH, f_mgrW;
    decx::utils::frag_manager_gen_from_fragLen(&f_mgrH, proc_dim.y, _BLOCKED_CONV2_FP64_H_);
    decx::utils::frag_manager_gen_from_fragLen(&f_mgrW, proc_dim.x, _BLOCKED_CONV2_FP64_W_);

    for (int i = 0; i < conv2_mk_props->channel_size; ++i) {
        decx::_general_copy2D_BC<double>(src + i * conv2_mk_props->page_size_dst, tmp_src, start, end, conv2_mk_props->Wdst << 2, conv2_mk_props->Wsrc << 2);

        decx::conv::CPUK::_conv2_rN_fp64_ST_rw4_unconfigured(tmp_src, kernel + i * conv2_mk_props->page_size_ker,
            dst + i * conv2_mk_props->page_size_dst, proc_dim, conv2_mk_props->ker_dims, conv2_mk_props->Wsrc,
            conv2_mk_props->Wdst, &f_mgrH, &f_mgrW, conv2_mk_props->_loop);
    }
}



_THREAD_CALL_
void decx::conv::CPUK::_conv2_rN_BC_MK_fp32_ST_bottom_rw4(double* __restrict src, double* __restrict tmp_src, double* __restrict kernel, double* __restrict dst,
    const uint2 proc_dim, const decx::_C2_MK64* conv2_mk_props)
{
    const uint2 start = make_uint2(0, conv2_mk_props->ker_dims.x / 2);
    const uint2 end = make_uint2(proc_dim.y + conv2_mk_props->ker_dims.y / 2, (proc_dim.x << 2) + start.y);

    decx::utils::frag_manager f_mgrH, f_mgrW;
    decx::utils::frag_manager_gen_from_fragLen(&f_mgrH, proc_dim.y, _BLOCKED_CONV2_FP64_H_);
    decx::utils::frag_manager_gen_from_fragLen(&f_mgrW, proc_dim.x, _BLOCKED_CONV2_FP64_W_);

    for (int i = 0; i < conv2_mk_props->channel_size; ++i) {
        decx::_general_copy2D_BC<double>(src + i * conv2_mk_props->page_size_dst, tmp_src, start, end, conv2_mk_props->Wdst << 2, conv2_mk_props->Wsrc << 2);

        decx::conv::CPUK::_conv2_rN_fp64_ST_rw4_unconfigured(tmp_src, kernel + i * conv2_mk_props->page_size_ker,
            dst + i * conv2_mk_props->page_size_dst, proc_dim, conv2_mk_props->ker_dims, conv2_mk_props->Wsrc,
            conv2_mk_props->Wdst, &f_mgrH, &f_mgrW, conv2_mk_props->_loop);
    }
}



// ---------------------------------------------- level 2 ------------------------------------------------------------




_THREAD_CALL_
void decx::conv::CPUK::_conv2_r1_BC_MK_fp64_ST_top(double* __restrict src, double* __restrict tmp_src, double* __restrict kernel, double* __restrict dst, const uint2 proc_dim,
    const decx::_C2_MK64* conv2_mk_props)
{
    const uint2 start = make_uint2(conv2_mk_props->ker_dims.y / 2, conv2_mk_props->ker_dims.x / 2);
    const uint2 end = make_uint2(proc_dim.y + conv2_mk_props->ker_dims.y - 1, (proc_dim.x << 2) + start.y);

    decx::utils::frag_manager f_mgrH, f_mgrW;
    decx::utils::frag_manager_gen_from_fragLen(&f_mgrH, proc_dim.y, _BLOCKED_CONV2_FP64_H_);
    decx::utils::frag_manager_gen_from_fragLen(&f_mgrW, proc_dim.x, _BLOCKED_CONV2_FP64_W_);

    for (int i = 0; i < conv2_mk_props->channel_size; ++i) {
        decx::_general_copy2D_BC<double>(src + i * conv2_mk_props->page_size_dst, tmp_src, start, end, conv2_mk_props->Wdst << 2, conv2_mk_props->Wsrc << 2);

        decx::conv::CPUK::_conv2_r1_fp64_ST_unconfigured(tmp_src, kernel + i * conv2_mk_props->page_size_ker,
            dst + i * conv2_mk_props->page_size_dst, proc_dim, conv2_mk_props->ker_dims, conv2_mk_props->Wsrc,
            conv2_mk_props->Wdst, &f_mgrH, &f_mgrW);
    }
}



_THREAD_CALL_
void decx::conv::CPUK::_conv2_r1_BC_MK_fp64_ST_mid(double* __restrict src, double* __restrict tmp_src, double* __restrict kernel, double* __restrict dst,
    const uint2 proc_dim, const decx::_C2_MK64* conv2_mk_props)
{
    const uint2 start = make_uint2(0, conv2_mk_props->ker_dims.x / 2);
    const uint2 end = make_uint2(proc_dim.y + conv2_mk_props->ker_dims.y - 1, (proc_dim.x << 2) + start.y);

    decx::utils::frag_manager f_mgrH, f_mgrW;
    decx::utils::frag_manager_gen_from_fragLen(&f_mgrH, proc_dim.y, _BLOCKED_CONV2_FP64_H_);
    decx::utils::frag_manager_gen_from_fragLen(&f_mgrW, proc_dim.x, _BLOCKED_CONV2_FP64_W_);

    for (int i = 0; i < conv2_mk_props->channel_size; ++i) {
        decx::_general_copy2D_BC<double>(src + i * conv2_mk_props->page_size_dst, tmp_src, start, end, conv2_mk_props->Wdst << 2, conv2_mk_props->Wsrc << 2);

        decx::conv::CPUK::_conv2_r1_fp64_ST_unconfigured(tmp_src, kernel + i * conv2_mk_props->page_size_ker,
            dst + i * conv2_mk_props->page_size_dst, proc_dim, conv2_mk_props->ker_dims, conv2_mk_props->Wsrc,
            conv2_mk_props->Wdst, &f_mgrH, &f_mgrW);
    }
}



_THREAD_CALL_
void decx::conv::CPUK::_conv2_r1_BC_MK_fp64_ST_bottom(double* __restrict src, double* __restrict tmp_src, double* __restrict kernel, double* __restrict dst,
    const uint2 proc_dim, const decx::_C2_MK64* conv2_mk_props)
{
    const uint2 start = make_uint2(0, conv2_mk_props->ker_dims.x / 2);
    const uint2 end = make_uint2(proc_dim.y + conv2_mk_props->ker_dims.y / 2, (proc_dim.x << 2) + start.y);

    decx::utils::frag_manager f_mgrH, f_mgrW;
    decx::utils::frag_manager_gen_from_fragLen(&f_mgrH, proc_dim.y, _BLOCKED_CONV2_FP64_H_);
    decx::utils::frag_manager_gen_from_fragLen(&f_mgrW, proc_dim.x, _BLOCKED_CONV2_FP64_W_);

    for (int i = 0; i < conv2_mk_props->channel_size; ++i) {
        decx::_general_copy2D_BC<double>(src + i * conv2_mk_props->page_size_dst, tmp_src, start, end, conv2_mk_props->Wdst << 2, conv2_mk_props->Wsrc << 2);

        decx::conv::CPUK::_conv2_r2_fp64_ST_unconfigured(tmp_src, kernel + i * conv2_mk_props->page_size_ker,
            dst + i * conv2_mk_props->page_size_dst, proc_dim, conv2_mk_props->ker_dims, conv2_mk_props->Wsrc,
            conv2_mk_props->Wdst, &f_mgrH, &f_mgrW);
    }
}



_THREAD_CALL_
void decx::conv::CPUK::_conv2_r2_BC_MK_fp64_ST_top(double* __restrict src, double* __restrict tmp_src, double* __restrict kernel, double* __restrict dst, const uint2 proc_dim,
    const decx::_C2_MK64* conv2_mk_props)
{
    const uint2 start = make_uint2(conv2_mk_props->ker_dims.y / 2, conv2_mk_props->ker_dims.x / 2);
    const uint2 end = make_uint2(proc_dim.y + conv2_mk_props->ker_dims.y - 1, (proc_dim.x << 2) + start.y);

    decx::utils::frag_manager f_mgrH, f_mgrW;
    decx::utils::frag_manager_gen_from_fragLen(&f_mgrH, proc_dim.y, _BLOCKED_CONV2_FP64_H_);
    decx::utils::frag_manager_gen_from_fragLen(&f_mgrW, proc_dim.x, _BLOCKED_CONV2_FP64_W_);

    for (int i = 0; i < conv2_mk_props->channel_size; ++i) {
        decx::_general_copy2D_BC<double>(src + i * conv2_mk_props->page_size_dst, tmp_src, start, end, conv2_mk_props->Wdst << 2, conv2_mk_props->Wsrc << 2);

        decx::conv::CPUK::_conv2_r2_fp64_ST_unconfigured(tmp_src, kernel + i * conv2_mk_props->page_size_ker,
            dst + i * conv2_mk_props->page_size_dst, proc_dim, conv2_mk_props->ker_dims, conv2_mk_props->Wsrc,
            conv2_mk_props->Wdst, &f_mgrH, &f_mgrW);
    }
}



_THREAD_CALL_
void decx::conv::CPUK::_conv2_r2_BC_MK_fp64_ST_mid(double* __restrict src, double* __restrict tmp_src, double* __restrict kernel, double* __restrict dst,
    const uint2 proc_dim, const decx::_C2_MK64* conv2_mk_props)
{
    const uint2 start = make_uint2(0, conv2_mk_props->ker_dims.x / 2);
    const uint2 end = make_uint2(proc_dim.y + conv2_mk_props->ker_dims.y - 1, (proc_dim.x << 2) + start.y);

    decx::utils::frag_manager f_mgrH, f_mgrW;
    decx::utils::frag_manager_gen_from_fragLen(&f_mgrH, proc_dim.y, _BLOCKED_CONV2_FP64_H_);
    decx::utils::frag_manager_gen_from_fragLen(&f_mgrW, proc_dim.x, _BLOCKED_CONV2_FP64_W_);

    for (int i = 0; i < conv2_mk_props->channel_size; ++i) {
        decx::_general_copy2D_BC<double>(src + i * conv2_mk_props->page_size_dst, tmp_src, start, end, conv2_mk_props->Wdst << 2, conv2_mk_props->Wsrc << 2);

        decx::conv::CPUK::_conv2_r2_fp64_ST_unconfigured(tmp_src, kernel + i * conv2_mk_props->page_size_ker,
            dst + i * conv2_mk_props->page_size_dst, proc_dim, conv2_mk_props->ker_dims, conv2_mk_props->Wsrc,
            conv2_mk_props->Wdst, &f_mgrH, &f_mgrW);
    }
}



_THREAD_CALL_
void decx::conv::CPUK::_conv2_r2_BC_MK_fp64_ST_bottom(double* __restrict src, double* __restrict tmp_src, double* __restrict kernel, double* __restrict dst,
    const uint2 proc_dim, const decx::_C2_MK64* conv2_mk_props)
{
    const uint2 start = make_uint2(0, conv2_mk_props->ker_dims.x / 2);
    const uint2 end = make_uint2(proc_dim.y + conv2_mk_props->ker_dims.y / 2, (proc_dim.x << 2) + start.y);

    decx::utils::frag_manager f_mgrH, f_mgrW;
    decx::utils::frag_manager_gen_from_fragLen(&f_mgrH, proc_dim.y, _BLOCKED_CONV2_FP64_H_);
    decx::utils::frag_manager_gen_from_fragLen(&f_mgrW, proc_dim.x, _BLOCKED_CONV2_FP64_W_);

    for (int i = 0; i < conv2_mk_props->channel_size; ++i) {
        decx::_general_copy2D_BC<double>(src + i * conv2_mk_props->page_size_dst, tmp_src, start, end, conv2_mk_props->Wdst << 2, conv2_mk_props->Wsrc << 2);

        decx::conv::CPUK::_conv2_r1_fp64_ST_unconfigured(tmp_src, kernel + i * conv2_mk_props->page_size_ker,
            dst + i * conv2_mk_props->page_size_dst, proc_dim, conv2_mk_props->ker_dims, conv2_mk_props->Wsrc,
            conv2_mk_props->Wdst, &f_mgrH, &f_mgrW);
    }
}



void decx::conv::_conv2_rN_BC_MK_fp64_caller(double* src,                          double* tmp_src,
                                       double* kernel,                       double* dst, 
                                       const uint2 proc_dim,                 decx::utils::_thr_1D* t1D,
                                       decx::_C2_MK64* conv2_props)
{
    typedef void (*__called_ST_func) (double*, double*, double*, double*, const uint2, const decx::_C2_MK64*);

    __called_ST_func __func_top = NULL,
        __func_mid = NULL, 
        __func_bottom = NULL;
    if (conv2_props->reg_WL == 2) {
        __func_top = decx::conv::CPUK::_conv2_rN_BC_MK_fp32_ST_top_rw2;
        __func_mid = decx::conv::CPUK::_conv2_rN_BC_MK_fp32_ST_mid_rw2;
        __func_bottom = decx::conv::CPUK::_conv2_rN_BC_MK_fp32_ST_bottom_rw2;
    }
    else {
        __func_top = decx::conv::CPUK::_conv2_rN_BC_MK_fp32_ST_top_rw4;
        __func_mid = decx::conv::CPUK::_conv2_rN_BC_MK_fp32_ST_mid_rw4;
        __func_bottom = decx::conv::CPUK::_conv2_rN_BC_MK_fp32_ST_bottom_rw4;
    }

    double* tmp_src_ptr = src, * tmp_dst_ptr = dst, *tmp_cache_src = tmp_src;
    const size_t frag_cache_src = (conv2_props->f_mgr->frag_len + conv2_props->ker_dims.y - 1) * (conv2_props->Wsrc << 2);
    const size_t frag_dst_size = conv2_props->f_mgr->frag_len * (conv2_props->Wdst << 2);
    
    t1D->_async_thread[0] = decx::cpu::register_task_default( __func_top,
        tmp_src_ptr, tmp_cache_src, kernel, tmp_dst_ptr,
        make_uint2(proc_dim.x, conv2_props->f_mgr->frag_len), conv2_props);

    tmp_src_ptr += (conv2_props->f_mgr->frag_len - conv2_props->ker_dims.y / 2) * (conv2_props->Wdst << 2);
    tmp_cache_src += frag_cache_src;
    tmp_dst_ptr += frag_dst_size;

    for (int i = 1; i < t1D->total_thread - 1; ++i)
    {
        t1D->_async_thread[i] = decx::cpu::register_task_default( __func_mid,
            tmp_src_ptr, tmp_cache_src, kernel, tmp_dst_ptr,
            make_uint2(proc_dim.x, conv2_props->f_mgr->frag_len), conv2_props);

        tmp_src_ptr += conv2_props->f_mgr->frag_len * (conv2_props->Wdst << 2);
        tmp_cache_src += frag_cache_src;
        tmp_dst_ptr += frag_dst_size;
    }

    if (conv2_props->f_mgr->frag_left_over != 0) {
        t1D->_async_thread[t1D->total_thread - 1] = decx::cpu::register_task_default( __func_bottom,
            tmp_src_ptr, tmp_cache_src, kernel, tmp_dst_ptr, make_uint2(proc_dim.x, conv2_props->f_mgr->frag_left_over),
            conv2_props);
    }
    else {
        t1D->_async_thread[t1D->total_thread - 1] = decx::cpu::register_task_default( __func_bottom,
            tmp_src_ptr, tmp_cache_src, kernel, tmp_dst_ptr, make_uint2(proc_dim.x, conv2_props->f_mgr->frag_len),
            conv2_props);
    }
    t1D->__sync_all_threads();
}



void decx::conv::_conv2_r1_r2_BC_MK_fp64_caller(double* src,                          double* tmp_src,
                                          double* kernel,                       double* dst, 
                                          const uint2 proc_dim,                decx::utils::_thr_1D* t1D,
                                          decx::_C2_MK64* conv2_props)
{
    typedef void (*__called_ST_func) (double*, double*, double*, double*, const uint2, const decx::_C2_MK64*);

    __called_ST_func __func_top = NULL,
        __func_mid = NULL, 
        __func_bottom = NULL;
    if (conv2_props->reg_WL == 2) {
        __func_top = decx::conv::CPUK::_conv2_r1_BC_MK_fp64_ST_top;
        __func_mid = decx::conv::CPUK::_conv2_r1_BC_MK_fp64_ST_mid;
        __func_bottom = decx::conv::CPUK::_conv2_r1_BC_MK_fp64_ST_bottom;
    }
    else {
        __func_top = decx::conv::CPUK::_conv2_r2_BC_MK_fp64_ST_top;
        __func_mid = decx::conv::CPUK::_conv2_r2_BC_MK_fp64_ST_mid;
        __func_bottom = decx::conv::CPUK::_conv2_r2_BC_MK_fp64_ST_bottom;
    }

    double* tmp_src_ptr = src, * tmp_dst_ptr = dst, *tmp_cache_src = tmp_src;
    const size_t frag_cache_src = (conv2_props->f_mgr->frag_len + conv2_props->ker_dims.y - 1) * (conv2_props->Wsrc << 2);
    const size_t frag_dst_size = conv2_props->f_mgr->frag_len * (conv2_props->Wdst << 2);
    
    t1D->_async_thread[0] = decx::cpu::register_task_default( __func_top,
        tmp_src_ptr, tmp_cache_src, kernel, tmp_dst_ptr,
        make_uint2(proc_dim.x, conv2_props->f_mgr->frag_len), conv2_props);

    tmp_src_ptr += (conv2_props->f_mgr->frag_len - conv2_props->ker_dims.y / 2) * (conv2_props->Wdst << 2);
    tmp_cache_src += frag_cache_src;
    tmp_dst_ptr += frag_dst_size;

    for (int i = 1; i < t1D->total_thread - 1; ++i)
    {
        t1D->_async_thread[i] = decx::cpu::register_task_default( __func_mid,
            tmp_src_ptr, tmp_cache_src, kernel, tmp_dst_ptr,
            make_uint2(proc_dim.x, conv2_props->f_mgr->frag_len), conv2_props);

        tmp_src_ptr += conv2_props->f_mgr->frag_len * (conv2_props->Wdst << 2);
        tmp_cache_src += frag_cache_src;
        tmp_dst_ptr += frag_dst_size;
    }

    if (conv2_props->f_mgr->frag_left_over != 0) {
        t1D->_async_thread[t1D->total_thread - 1] = decx::cpu::register_task_default( __func_bottom,
            tmp_src_ptr, tmp_cache_src, kernel, tmp_dst_ptr, make_uint2(proc_dim.x, conv2_props->f_mgr->frag_left_over),
            conv2_props);
    }
    else {
        t1D->_async_thread[t1D->total_thread - 1] = decx::cpu::register_task_default( __func_bottom,
            tmp_src_ptr, tmp_cache_src, kernel, tmp_dst_ptr, make_uint2(proc_dim.x, conv2_props->f_mgr->frag_len),
            conv2_props);
    }
    t1D->__sync_all_threads();
}
