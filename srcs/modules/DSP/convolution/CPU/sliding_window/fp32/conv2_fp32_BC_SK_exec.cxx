/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#include "conv2_fp32_BC_SK_exec.h"


_THREAD_CALL_
void decx::conv::CPUK::_conv2_rN_fp32_ST_unconfigured(float* __restrict     src,
                                          float* __restrict     kernel,
                                          float* __restrict     dst, 
                                          const uint2           proc_dim, 
                                          const uint2           ker_dims,
                                          const uint            Wsrc,
                                          const uint            Wdst,
                                          const ushort          reg_WL,
                                          const _fmgr*          f_mgrH,    // fragment info on height
                                          const _fmgr*          f_mgrW,
                                          const uint            _loop)    // fragment info on width
{
    __m256 res_vec8;
    size_t dex_src = 0, dex_dst = 0;

    const uint _loopH = f_mgrH->is_left ? f_mgrH->frag_num - 1 : f_mgrH->frag_num;
    
    for (int i = 0; i < _loopH; ++i) 
    {
        const uint _loopW = f_mgrW->is_left ? f_mgrW->frag_num - 1 : f_mgrW->frag_num;

        for (int k = 0; k < _loopW; ++k) {
            decx::conv::CPUK::_conv2_rN_rect_fixed_fp32_ST(
                DECX_PTR_SHF_XY_SAME_TYPE(src, i * _BLOCKED_CONV2_FP32_H_, (k << 3) * _BLOCKED_CONV2_FP32_W_, Wsrc << 3),
                kernel,
                DECX_PTR_SHF_XY_SAME_TYPE(dst, i * _BLOCKED_CONV2_FP32_H_, (k << 3) * _BLOCKED_CONV2_FP32_W_, Wdst << 3),
                ker_dims, Wsrc, Wdst, reg_WL, _loop);
        }
        if (f_mgrW->is_left) {
            const uint _sum_prev_lenW = proc_dim.x - f_mgrW->frag_left_over;
            decx::conv::CPUK::_conv2_rN_rect_flex_fp32_ST(
                DECX_PTR_SHF_XY_SAME_TYPE(src, i * _BLOCKED_CONV2_FP32_H_, (_sum_prev_lenW << 3), Wsrc << 3),
                kernel,
                DECX_PTR_SHF_XY_SAME_TYPE(dst, i * _BLOCKED_CONV2_FP32_H_, (_sum_prev_lenW << 3), Wdst << 3),
                make_uint2(f_mgrW->frag_left_over, _BLOCKED_CONV2_FP32_H_), ker_dims,
                Wsrc, Wdst, reg_WL, _loop);
        }
    }
    
    if (f_mgrH->is_left)
    {
        const uint _sum_prev_lenH = proc_dim.y - f_mgrH->frag_left_over;
        const uint _loopW = f_mgrW->is_left ? f_mgrW->frag_num - 1 : f_mgrW->frag_num;

        for (int k = 0; k < _loopW; ++k) {
            decx::conv::CPUK::_conv2_rN_rect_flex_fp32_ST(
                DECX_PTR_SHF_XY_SAME_TYPE(src, _sum_prev_lenH, (k << 3) * _BLOCKED_CONV2_FP32_W_, Wsrc << 3),
                kernel,
                DECX_PTR_SHF_XY_SAME_TYPE(dst, _sum_prev_lenH, (k << 3) * _BLOCKED_CONV2_FP32_W_, Wdst << 3),
                make_uint2(_BLOCKED_CONV2_FP32_W_, f_mgrH->frag_left_over),
                ker_dims, Wsrc, Wdst, reg_WL, _loop);
        }
        if (f_mgrW->is_left) {
            const uint _sum_prev_lenW = proc_dim.x - f_mgrW->frag_left_over;
            decx::conv::CPUK::_conv2_rN_rect_flex_fp32_ST(
                DECX_PTR_SHF_XY_SAME_TYPE(src, _sum_prev_lenH, (_sum_prev_lenW << 3), Wsrc << 3),
                kernel,
                DECX_PTR_SHF_XY_SAME_TYPE(dst, _sum_prev_lenH, (_sum_prev_lenW << 3), Wdst << 3),
                make_uint2(f_mgrW->frag_left_over, f_mgrH->frag_left_over), ker_dims,
                Wsrc, Wdst, reg_WL, _loop);
        }
    }
}



_THREAD_FUNCTION_
void decx::conv::CPUK::_conv2_r1_r4_fp32_ST_unconfigured(float* __restrict   src,
                                             float* __restrict   kernel,
                                             float* __restrict   dst, 
                                             const uint2         proc_dim, 
                                             const uint2         ker_dims,
                                             const uint          Wsrc,
                                             const uint          Wdst,
                                             const _fmgr*        f_mgrH,    // fragment info on height
                                             const _fmgr*        f_mgrW)    // fragment info on width
{
    __m256 res_vec8;
    size_t dex_src = 0, dex_dst = 0;

    const uint _loopH = f_mgrH->is_left ? f_mgrH->frag_num - 1 : f_mgrH->frag_num;
    
    for (int i = 0; i < _loopH; ++i) 
    {
        const uint _loopW = f_mgrW->is_left ? f_mgrW->frag_num - 1 : f_mgrW->frag_num;

        for (int k = 0; k < _loopW; ++k) {
            decx::conv::CPUK::_conv2_r1_r4_rect_fixed_fp32_ST(
                DECX_PTR_SHF_XY_SAME_TYPE(src, i * _BLOCKED_CONV2_FP32_H_, (k << 3) * _BLOCKED_CONV2_FP32_W_, Wsrc << 3),
                kernel,
                DECX_PTR_SHF_XY_SAME_TYPE(dst, i * _BLOCKED_CONV2_FP32_H_, (k << 3) * _BLOCKED_CONV2_FP32_W_, Wdst << 3),
                ker_dims, Wsrc, Wdst);
        }
        if (f_mgrW->is_left) {
            const uint _sum_prev_lenW = proc_dim.x - f_mgrW->frag_left_over;
            decx::conv::CPUK::_conv2_r1_r4_rect_flex_fp32_ST(
                DECX_PTR_SHF_XY_SAME_TYPE(src, i * _BLOCKED_CONV2_FP32_H_, (_sum_prev_lenW << 3), Wsrc << 3),
                kernel,
                DECX_PTR_SHF_XY_SAME_TYPE(dst, i * _BLOCKED_CONV2_FP32_H_, (_sum_prev_lenW << 3), Wdst << 3),
                make_uint2(f_mgrW->frag_left_over, _BLOCKED_CONV2_FP32_H_), ker_dims,
                Wsrc, Wdst);
        }
    }
    
    if (f_mgrH->is_left)
    {
        const uint _sum_prev_lenH = proc_dim.y - f_mgrH->frag_left_over;
        const uint _loopW = f_mgrW->is_left ? f_mgrW->frag_num - 1 : f_mgrW->frag_num;

        for (int k = 0; k < _loopW; ++k) {
            decx::conv::CPUK::_conv2_r1_r4_rect_flex_fp32_ST(
                DECX_PTR_SHF_XY_SAME_TYPE(src, _sum_prev_lenH, (k << 3) * _BLOCKED_CONV2_FP32_W_, Wsrc << 3),
                kernel,
                DECX_PTR_SHF_XY_SAME_TYPE(dst, _sum_prev_lenH, (k << 3) * _BLOCKED_CONV2_FP32_W_, Wdst << 3),
                make_uint2(_BLOCKED_CONV2_FP32_W_, f_mgrH->frag_left_over),
                ker_dims, Wsrc, Wdst);
        }
        if (f_mgrW->is_left) {
            const uint _sum_prev_lenW = proc_dim.x - f_mgrW->frag_left_over;
            decx::conv::CPUK::_conv2_r1_r4_rect_flex_fp32_ST(
                DECX_PTR_SHF_XY_SAME_TYPE(src, _sum_prev_lenH, (_sum_prev_lenW << 3), Wsrc << 3),
                kernel,
                DECX_PTR_SHF_XY_SAME_TYPE(dst, _sum_prev_lenH, (_sum_prev_lenW << 3), Wdst << 3),
                make_uint2(f_mgrW->frag_left_over, f_mgrH->frag_left_over), ker_dims,
                Wsrc, Wdst);
        }
    }
}




// ------------------------------------------------------------------------------ level 2 ------------------------------------------------------------------------

_THREAD_CALL_
void decx::conv::CPUK::_conv2_rN_BC_SK_fp32_ST_top(float* __restrict src, float* __restrict tmp_src, float* __restrict kernel, float* __restrict dst, const uint2 proc_dim,
    const decx::_C2_MK32* conv2_mk_props)
{
    const uint2 start = make_uint2(conv2_mk_props->ker_dims.y / 2, conv2_mk_props->ker_dims.x / 2);
    const uint2 end = make_uint2(proc_dim.y + conv2_mk_props->ker_dims.y - 1, (proc_dim.x << 3) + start.y);

    decx::utils::frag_manager f_mgrH, f_mgrW;
    decx::utils::frag_manager_gen_from_fragLen(&f_mgrH, proc_dim.y, _BLOCKED_CONV2_FP32_H_);
    decx::utils::frag_manager_gen_from_fragLen(&f_mgrW, proc_dim.x, _BLOCKED_CONV2_FP32_W_);

    for (int i = 0; i < conv2_mk_props->channel_size; ++i) {
        decx::_general_copy2D_BC<float>(src + i * conv2_mk_props->page_size_dst, tmp_src, start, end, conv2_mk_props->Wdst << 3, conv2_mk_props->Wsrc << 3);

        decx::conv::CPUK::_conv2_rN_fp32_ST_unconfigured(tmp_src, kernel, dst + i * conv2_mk_props->page_size_dst,
            proc_dim, conv2_mk_props->ker_dims, conv2_mk_props->Wsrc, conv2_mk_props->Wdst, conv2_mk_props->reg_WL, 
            &f_mgrH, &f_mgrW, conv2_mk_props->_loop);
    }
}



_THREAD_CALL_
void decx::conv::CPUK::_conv2_rN_BC_SK_fp32_ST_mid(float* __restrict src, float* __restrict tmp_src, float* __restrict kernel, float* __restrict dst,
    const uint2 proc_dim, const decx::_C2_MK32* conv2_mk_props)
{
    const uint2 start = make_uint2(0, conv2_mk_props->ker_dims.x / 2);
    const uint2 end = make_uint2(proc_dim.y + conv2_mk_props->ker_dims.y - 1, (proc_dim.x << 3) + start.y);

    decx::utils::frag_manager f_mgrH, f_mgrW;
    decx::utils::frag_manager_gen_from_fragLen(&f_mgrH, proc_dim.y, _BLOCKED_CONV2_FP32_H_);
    decx::utils::frag_manager_gen_from_fragLen(&f_mgrW, proc_dim.x, _BLOCKED_CONV2_FP32_W_);

    for (int i = 0; i < conv2_mk_props->channel_size; ++i) {
        decx::_general_copy2D_BC<float>(src + i * conv2_mk_props->page_size_dst, tmp_src, start, end, conv2_mk_props->Wdst << 3, conv2_mk_props->Wsrc << 3);

        decx::conv::CPUK::_conv2_rN_fp32_ST_unconfigured(tmp_src, kernel, dst + i * conv2_mk_props->page_size_dst,
            proc_dim, conv2_mk_props->ker_dims, conv2_mk_props->Wsrc, conv2_mk_props->Wdst, conv2_mk_props->reg_WL, 
            &f_mgrH, &f_mgrW, conv2_mk_props->_loop);
    }
}



_THREAD_CALL_
void decx::conv::CPUK::_conv2_rN_BC_SK_fp32_ST_bottom(float* __restrict src, float* __restrict tmp_src, float* __restrict kernel, float* __restrict dst,
    const uint2 proc_dim, const decx::_C2_MK32* conv2_mk_props)
{
    const uint2 start = make_uint2(0, conv2_mk_props->ker_dims.x / 2);
    const uint2 end = make_uint2(proc_dim.y + conv2_mk_props->ker_dims.y / 2, (proc_dim.x << 3) + start.y);

    decx::utils::frag_manager f_mgrH, f_mgrW;
    decx::utils::frag_manager_gen_from_fragLen(&f_mgrH, proc_dim.y, _BLOCKED_CONV2_FP32_H_);
    decx::utils::frag_manager_gen_from_fragLen(&f_mgrW, proc_dim.x, _BLOCKED_CONV2_FP32_W_);

    for (int i = 0; i < conv2_mk_props->channel_size; ++i) {
        decx::_general_copy2D_BC<float>(src + i * conv2_mk_props->page_size_dst, tmp_src, start, end, conv2_mk_props->Wdst << 3, conv2_mk_props->Wsrc << 3);

        decx::conv::CPUK::_conv2_rN_fp32_ST_unconfigured(tmp_src, kernel, dst + i * conv2_mk_props->page_size_dst,
            proc_dim, conv2_mk_props->ker_dims, conv2_mk_props->Wsrc, conv2_mk_props->Wdst, conv2_mk_props->reg_WL, 
            &f_mgrH, &f_mgrW, conv2_mk_props->_loop);
    }
}



// r1, r4
_THREAD_CALL_ 
void decx::conv::CPUK::_conv2_r1_r4_BC_SK_fp32_ST_top(float* __restrict src, float* __restrict tmp_src, float* __restrict kernel, float* __restrict dst, 
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

        decx::conv::CPUK::_conv2_r1_r4_fp32_ST_unconfigured(tmp_src, kernel, dst + i * conv2_mk_props->page_size_dst,
            proc_dim, conv2_mk_props->ker_dims, conv2_mk_props->Wsrc, conv2_mk_props->Wdst, &f_mgrH, &f_mgrW);
    }
}



_THREAD_CALL_ 
void decx::conv::CPUK::_conv2_r1_r4_BC_SK_fp32_ST_mid(float* __restrict src, float* __restrict tmp_src, float* __restrict kernel, float* __restrict dst, 
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

        decx::conv::CPUK::_conv2_r1_r4_fp32_ST_unconfigured(tmp_src, kernel, dst + i * conv2_mk_props->page_size_dst,
            proc_dim, conv2_mk_props->ker_dims, conv2_mk_props->Wsrc, conv2_mk_props->Wdst, &f_mgrH, &f_mgrW);
    }
}



_THREAD_CALL_ 
void decx::conv::CPUK::_conv2_r1_r4_BC_SK_fp32_ST_bottom(float* __restrict src, float* __restrict tmp_src, float* __restrict kernel, float* __restrict dst, 
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

        decx::conv::CPUK::_conv2_r1_r4_fp32_ST_unconfigured(tmp_src, kernel, dst + i * conv2_mk_props->page_size_dst,
            proc_dim, conv2_mk_props->ker_dims, conv2_mk_props->Wsrc, conv2_mk_props->Wdst, &f_mgrH, &f_mgrW);
    }
}





void decx::conv::_conv2_rN_BC_SK_fp32_caller(float* src,                          float* tmp_src, 
                                       float* kernel,                       float* dst, 
                                       const uint2 proc_dim,                decx::utils::_thr_1D* t1D,
                                       decx::_C2_MK32* conv2_props)
{
    float* tmp_src_ptr = src, * tmp_dst_ptr = dst, *tmp_cache_src = tmp_src;
    const size_t frag_cache_src = (conv2_props->f_mgr->frag_len + conv2_props->ker_dims.y - 1) * (conv2_props->Wsrc << 3);
    const size_t frag_dst_size = conv2_props->f_mgr->frag_len * (conv2_props->Wdst << 3);
    
    t1D->_async_thread[0] = decx::cpu::register_task_default(decx::conv::CPUK::_conv2_rN_BC_SK_fp32_ST_top,
        tmp_src_ptr, tmp_cache_src, kernel, tmp_dst_ptr,
        make_uint2(proc_dim.x, conv2_props->f_mgr->frag_len), conv2_props);

    tmp_src_ptr += (conv2_props->f_mgr->frag_len - conv2_props->ker_dims.y / 2) * (conv2_props->Wdst << 3);
    tmp_cache_src += frag_cache_src;
    tmp_dst_ptr += frag_dst_size;

    for (int i = 1; i < t1D->total_thread - 1; ++i)
    {
        t1D->_async_thread[i] = decx::cpu::register_task_default(decx::conv::CPUK::_conv2_rN_BC_SK_fp32_ST_mid,
            tmp_src_ptr, tmp_cache_src, kernel, tmp_dst_ptr,
            make_uint2(proc_dim.x, conv2_props->f_mgr->frag_len), conv2_props);

        tmp_src_ptr += conv2_props->f_mgr->frag_len * (conv2_props->Wdst << 3);
        tmp_cache_src += frag_cache_src;
        tmp_dst_ptr += frag_dst_size;
    }

    if (conv2_props->f_mgr->frag_left_over != 0) {
        t1D->_async_thread[t1D->total_thread - 1] = decx::cpu::register_task_default(decx::conv::CPUK::_conv2_rN_BC_SK_fp32_ST_bottom,
            tmp_src_ptr, tmp_cache_src, kernel, tmp_dst_ptr, make_uint2(proc_dim.x, conv2_props->f_mgr->frag_left_over),
            conv2_props);
    }
    else {
        t1D->_async_thread[t1D->total_thread - 1] = decx::cpu::register_task_default(decx::conv::CPUK::_conv2_rN_BC_SK_fp32_ST_bottom,
            tmp_src_ptr, tmp_cache_src, kernel, tmp_dst_ptr, make_uint2(proc_dim.x, conv2_props->f_mgr->frag_len),
            conv2_props);
    }
    
    t1D->__sync_all_threads();
}





void decx::conv::_conv2_r1_r4_BC_SK_fp32_caller(float* src,                          float* tmp_src, 
                                          float* kernel,                       float* dst, 
                                          const uint2 proc_dim,                decx::utils::_thr_1D* t1D,
                                          decx::_C2_MK32* conv2_props)
{
    float* tmp_src_ptr = src, * tmp_dst_ptr = dst, *tmp_cache_src = tmp_src;
    const size_t frag_cache_src = (conv2_props->f_mgr->frag_len + conv2_props->ker_dims.y - 1) * (conv2_props->Wsrc << 3);
    const size_t frag_dst_size = conv2_props->f_mgr->frag_len * (conv2_props->Wdst << 3);
    
    t1D->_async_thread[0] = decx::cpu::register_task_default(decx::conv::CPUK::_conv2_r1_r4_BC_SK_fp32_ST_top,
        tmp_src_ptr, tmp_cache_src, kernel, tmp_dst_ptr,
        make_uint2(proc_dim.x, conv2_props->f_mgr->frag_len), conv2_props);

    tmp_src_ptr += (conv2_props->f_mgr->frag_len - conv2_props->ker_dims.y / 2) * (conv2_props->Wdst << 3);
    tmp_cache_src += frag_cache_src;
    tmp_dst_ptr += frag_dst_size;

    for (int i = 1; i < t1D->total_thread - 1; ++i)
    {
        t1D->_async_thread[i] = decx::cpu::register_task_default(decx::conv::CPUK::_conv2_r1_r4_BC_SK_fp32_ST_mid,
            tmp_src_ptr, tmp_cache_src, kernel, tmp_dst_ptr,
            make_uint2(proc_dim.x, conv2_props->f_mgr->frag_len), conv2_props);

        tmp_src_ptr += conv2_props->f_mgr->frag_len * (conv2_props->Wdst << 3);
        tmp_cache_src += frag_cache_src;
        tmp_dst_ptr += frag_dst_size;
    }

    if (conv2_props->f_mgr->frag_left_over != 0) {
        t1D->_async_thread[t1D->total_thread - 1] = decx::cpu::register_task_default(decx::conv::CPUK::_conv2_r1_r4_BC_SK_fp32_ST_bottom,
            tmp_src_ptr, tmp_cache_src, kernel, tmp_dst_ptr, make_uint2(proc_dim.x, conv2_props->f_mgr->frag_left_over),
            conv2_props);
    }
    else {
        t1D->_async_thread[t1D->total_thread - 1] = decx::cpu::register_task_default(decx::conv::CPUK::_conv2_r1_r4_BC_SK_fp32_ST_bottom,
            tmp_src_ptr, tmp_cache_src, kernel, tmp_dst_ptr, make_uint2(proc_dim.x, conv2_props->f_mgr->frag_len),
            conv2_props);
    }
    
    t1D->__sync_all_threads();
}
