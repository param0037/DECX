/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/

#include "conv2_fp64_NB_SK_exec.h"



_THREAD_FUNCTION_
void decx::conv::CPUK::_conv2_r1_fp64_ST_unconfigured(double* __restrict     src,
                                          double* __restrict     kernel, 
                                          double* __restrict     dst, 
                                          const uint2            proc_dim, 
                                          const uint2            ker_dims,
                                          const uint             Wsrc,
                                          const uint             Wdst,
                                          const _fmgr*           f_mgrH, 
                                          const _fmgr*           f_mgrW)
{
    __m256 res_vec8;
    size_t dex_src = 0, dex_dst = 0;

    const uint _loopH = f_mgrH->is_left ? f_mgrH->frag_num - 1 : f_mgrH->frag_num;
    
    for (int i = 0; i < _loopH; ++i) 
    {
        const uint _loopW = f_mgrW->is_left ? f_mgrW->frag_num - 1 : f_mgrW->frag_num;

        for (int k = 0; k < _loopW; ++k) {
            decx::conv::CPUK::_conv2_r1_rect_fixed_fp64_ST(
                DECX_PTR_SHF_XY_SAME_TYPE(src, i * _BLOCKED_CONV2_FP64_H_, (k << 2) * _BLOCKED_CONV2_FP64_W_, Wsrc << 2),
                kernel,
                DECX_PTR_SHF_XY_SAME_TYPE(dst, i * _BLOCKED_CONV2_FP64_H_, (k << 2) * _BLOCKED_CONV2_FP64_W_, Wdst << 2),
                ker_dims, Wsrc, Wdst);
        }
        if (f_mgrW->is_left) {
            const uint _sum_prev_lenW = proc_dim.x - f_mgrW->frag_left_over;
            decx::conv::CPUK::_conv2_r1_rect_flex_fp64_ST(
                DECX_PTR_SHF_XY_SAME_TYPE(src, i * _BLOCKED_CONV2_FP64_H_, (_sum_prev_lenW << 2), Wsrc << 2),
                kernel,
                DECX_PTR_SHF_XY_SAME_TYPE(dst, i * _BLOCKED_CONV2_FP64_H_, (_sum_prev_lenW << 2), Wdst << 2),
                make_uint2(f_mgrW->frag_left_over, _BLOCKED_CONV2_FP64_H_), ker_dims,
                Wsrc, Wdst);
        }
    }
    
    if (f_mgrH->is_left)
    {
        const uint _sum_prev_lenH = proc_dim.y - f_mgrH->frag_left_over;
        const uint _loopW = f_mgrW->is_left ? f_mgrW->frag_num - 1 : f_mgrW->frag_num;

        for (int k = 0; k < _loopW; ++k) {
            decx::conv::CPUK::_conv2_r1_rect_flex_fp64_ST(
                DECX_PTR_SHF_XY_SAME_TYPE(src, _sum_prev_lenH, (k << 2) * _BLOCKED_CONV2_FP64_W_, Wsrc << 2),
                kernel,
                DECX_PTR_SHF_XY_SAME_TYPE(dst, _sum_prev_lenH, (k << 2) * _BLOCKED_CONV2_FP64_W_, Wdst << 2),
                make_uint2(_BLOCKED_CONV2_FP64_W_, f_mgrH->frag_left_over),
                ker_dims, Wsrc, Wdst);
        }
        if (f_mgrW->is_left) {
            const uint _sum_prev_lenW = proc_dim.x - f_mgrW->frag_left_over;
            decx::conv::CPUK::_conv2_r1_rect_flex_fp64_ST(
                DECX_PTR_SHF_XY_SAME_TYPE(src, _sum_prev_lenH, (_sum_prev_lenW << 2), Wsrc << 2),
                kernel,
                DECX_PTR_SHF_XY_SAME_TYPE(dst, _sum_prev_lenH, (_sum_prev_lenW << 2), Wdst << 2),
                make_uint2(f_mgrW->frag_left_over, f_mgrH->frag_left_over), ker_dims,
                Wsrc, Wdst);
        }
    }
}



_THREAD_FUNCTION_
void decx::conv::CPUK::_conv2_r2_fp64_ST_unconfigured(double* __restrict     src,
                                          double* __restrict     kernel, 
                                          double* __restrict     dst, 
                                          const uint2            proc_dim, 
                                          const uint2            ker_dims,
                                          const uint             Wsrc,
                                          const uint             Wdst,
                                          const _fmgr*           f_mgrH, 
                                          const _fmgr*           f_mgrW)
{
    __m256 res_vec8;
    size_t dex_src = 0, dex_dst = 0;

    const uint _loopH = f_mgrH->is_left ? f_mgrH->frag_num - 1 : f_mgrH->frag_num;
    
    for (int i = 0; i < _loopH; ++i) 
    {
        const uint _loopW = f_mgrW->is_left ? f_mgrW->frag_num - 1 : f_mgrW->frag_num;

        for (int k = 0; k < _loopW; ++k) {
            decx::conv::CPUK::_conv2_r2_rect_fixed_fp64_ST(
                DECX_PTR_SHF_XY_SAME_TYPE(src, i * _BLOCKED_CONV2_FP64_H_, (k << 2) * _BLOCKED_CONV2_FP64_W_, Wsrc << 2),
                kernel,
                DECX_PTR_SHF_XY_SAME_TYPE(dst, i * _BLOCKED_CONV2_FP64_H_, (k << 2) * _BLOCKED_CONV2_FP64_W_, Wdst << 2),
                ker_dims, Wsrc, Wdst);
        }
        if (f_mgrW->is_left) {
            const uint _sum_prev_lenW = proc_dim.x - f_mgrW->frag_left_over;
            decx::conv::CPUK::_conv2_r2_rect_flex_fp64_ST(
                DECX_PTR_SHF_XY_SAME_TYPE(src, i * _BLOCKED_CONV2_FP64_H_, (_sum_prev_lenW << 2), Wsrc << 2),
                kernel,
                DECX_PTR_SHF_XY_SAME_TYPE(dst, i * _BLOCKED_CONV2_FP64_H_, (_sum_prev_lenW << 2), Wdst << 2),
                make_uint2(f_mgrW->frag_left_over, _BLOCKED_CONV2_FP64_H_), ker_dims,
                Wsrc, Wdst);
        }
    }
    
    if (f_mgrH->is_left)
    {
        const uint _sum_prev_lenH = proc_dim.y - f_mgrH->frag_left_over;
        const uint _loopW = f_mgrW->is_left ? f_mgrW->frag_num - 1 : f_mgrW->frag_num;

        for (int k = 0; k < _loopW; ++k) {
            decx::conv::CPUK::_conv2_r2_rect_flex_fp64_ST(
                DECX_PTR_SHF_XY_SAME_TYPE(src, _sum_prev_lenH, (k << 2) * _BLOCKED_CONV2_FP64_W_, Wsrc << 2),
                kernel,
                DECX_PTR_SHF_XY_SAME_TYPE(dst, _sum_prev_lenH, (k << 2) * _BLOCKED_CONV2_FP64_W_, Wdst << 2),
                make_uint2(_BLOCKED_CONV2_FP64_W_, f_mgrH->frag_left_over),
                ker_dims, Wsrc, Wdst);
        }
        if (f_mgrW->is_left) {
            const uint _sum_prev_lenW = proc_dim.x - f_mgrW->frag_left_over;
            decx::conv::CPUK::_conv2_r2_rect_flex_fp64_ST(
                DECX_PTR_SHF_XY_SAME_TYPE(src, _sum_prev_lenH, (_sum_prev_lenW << 2), Wsrc << 2),
                kernel,
                DECX_PTR_SHF_XY_SAME_TYPE(dst, _sum_prev_lenH, (_sum_prev_lenW << 2), Wdst << 2),
                make_uint2(f_mgrW->frag_left_over, f_mgrH->frag_left_over), ker_dims,
                Wsrc, Wdst);
        }
    }
}




_THREAD_FUNCTION_
void decx::conv::CPUK::_conv2_rN_fp64_ST_rw2_unconfigured(double* __restrict     src,
                                              double* __restrict     kernel, 
                                              double* __restrict     dst, 
                                              const uint2            proc_dim, 
                                              const uint2            ker_dims,
                                              const uint             Wsrc,
                                              const uint             Wdst,
                                              const _fmgr*           f_mgrH, 
                                              const _fmgr*           f_mgrW, 
                                              const uint             _loop)
{
    __m256 res_vec8;
    size_t dex_src = 0, dex_dst = 0;

    const uint _loopH = f_mgrH->is_left ? f_mgrH->frag_num - 1 : f_mgrH->frag_num;
    
    for (int i = 0; i < _loopH; ++i) 
    {
        const uint _loopW = f_mgrW->is_left ? f_mgrW->frag_num - 1 : f_mgrW->frag_num;

        for (int k = 0; k < _loopW; ++k) {
            decx::conv::CPUK::_conv2_rN_rect_fixed_fp64_ST_rw2(
                DECX_PTR_SHF_XY_SAME_TYPE(src, i * _BLOCKED_CONV2_FP64_H_, (k << 2) * _BLOCKED_CONV2_FP64_W_, Wsrc << 2),
                kernel,
                DECX_PTR_SHF_XY_SAME_TYPE(dst, i * _BLOCKED_CONV2_FP64_H_, (k << 2) * _BLOCKED_CONV2_FP64_W_, Wdst << 2),
                ker_dims, Wsrc, Wdst, _loop);
        }
        if (f_mgrW->is_left) {
            const uint _sum_prev_lenW = proc_dim.x - f_mgrW->frag_left_over;
            decx::conv::CPUK::_conv2_rN_rect_flex_fp64_ST_rw2(
                DECX_PTR_SHF_XY_SAME_TYPE(src, i * _BLOCKED_CONV2_FP64_H_, (_sum_prev_lenW << 2), Wsrc << 2),
                kernel,
                DECX_PTR_SHF_XY_SAME_TYPE(dst, i * _BLOCKED_CONV2_FP64_H_, (_sum_prev_lenW << 2), Wdst << 2),
                make_uint2(f_mgrW->frag_left_over, _BLOCKED_CONV2_FP64_H_), ker_dims,
                Wsrc, Wdst, _loop);
        }
    }
    
    if (f_mgrH->is_left)
    {
        const uint _sum_prev_lenH = proc_dim.y - f_mgrH->frag_left_over;
        const uint _loopW = f_mgrW->is_left ? f_mgrW->frag_num - 1 : f_mgrW->frag_num;

        for (int k = 0; k < _loopW; ++k) {
            decx::conv::CPUK::_conv2_rN_rect_flex_fp64_ST_rw2(
                DECX_PTR_SHF_XY_SAME_TYPE(src, _sum_prev_lenH, (k << 2) * _BLOCKED_CONV2_FP64_W_, Wsrc << 2),
                kernel,
                DECX_PTR_SHF_XY_SAME_TYPE(dst, _sum_prev_lenH, (k << 2) * _BLOCKED_CONV2_FP64_W_, Wdst << 2),
                make_uint2(_BLOCKED_CONV2_FP64_W_, f_mgrH->frag_left_over),
                ker_dims, Wsrc, Wdst, _loop);
        }
        if (f_mgrW->is_left) {
            const uint _sum_prev_lenW = proc_dim.x - f_mgrW->frag_left_over;
            decx::conv::CPUK::_conv2_rN_rect_flex_fp64_ST_rw2(
                DECX_PTR_SHF_XY_SAME_TYPE(src, _sum_prev_lenH, (_sum_prev_lenW << 2), Wsrc << 2),
                kernel,
                DECX_PTR_SHF_XY_SAME_TYPE(dst, _sum_prev_lenH, (_sum_prev_lenW << 2), Wdst << 2),
                make_uint2(f_mgrW->frag_left_over, f_mgrH->frag_left_over), ker_dims,
                Wsrc, Wdst, _loop);
        }
    }
}



_THREAD_FUNCTION_
void decx::conv::CPUK::_conv2_rN_fp64_ST_rw4_unconfigured(double* __restrict     src,
                                              double* __restrict     kernel, 
                                              double* __restrict     dst, 
                                              const uint2            proc_dim, 
                                              const uint2            ker_dims,
                                              const uint             Wsrc,
                                              const uint             Wdst,
                                              const _fmgr*           f_mgrH, 
                                              const _fmgr*           f_mgrW, 
                                              const uint             _loop)
{
    __m256 res_vec8;
    size_t dex_src = 0, dex_dst = 0;

    const uint _loopH = f_mgrH->is_left ? f_mgrH->frag_num - 1 : f_mgrH->frag_num;
    
    for (int i = 0; i < _loopH; ++i) 
    {
        const uint _loopW = f_mgrW->is_left ? f_mgrW->frag_num - 1 : f_mgrW->frag_num;

        for (int k = 0; k < _loopW; ++k) {
            decx::conv::CPUK::_conv2_rN_rect_fixed_fp64_ST_rw4(
                DECX_PTR_SHF_XY_SAME_TYPE(src, i * _BLOCKED_CONV2_FP64_H_, (k << 2) * _BLOCKED_CONV2_FP64_W_, Wsrc << 2),
                kernel,
                DECX_PTR_SHF_XY_SAME_TYPE(dst, i * _BLOCKED_CONV2_FP64_H_, (k << 2) * _BLOCKED_CONV2_FP64_W_, Wdst << 2),
                ker_dims, Wsrc, Wdst, _loop);
        }
        if (f_mgrW->is_left) {
            const uint _sum_prev_lenW = proc_dim.x - f_mgrW->frag_left_over;
            decx::conv::CPUK::_conv2_rN_rect_flex_fp64_ST_rw4(
                DECX_PTR_SHF_XY_SAME_TYPE(src, i * _BLOCKED_CONV2_FP64_H_, (_sum_prev_lenW << 2), Wsrc << 2),
                kernel,
                DECX_PTR_SHF_XY_SAME_TYPE(dst, i * _BLOCKED_CONV2_FP64_H_, (_sum_prev_lenW << 2), Wdst << 2),
                make_uint2(f_mgrW->frag_left_over, _BLOCKED_CONV2_FP64_H_), ker_dims,
                Wsrc, Wdst, _loop);
        }
    }
    
    if (f_mgrH->is_left)
    {
        const uint _sum_prev_lenH = proc_dim.y - f_mgrH->frag_left_over;
        const uint _loopW = f_mgrW->is_left ? f_mgrW->frag_num - 1 : f_mgrW->frag_num;

        for (int k = 0; k < _loopW; ++k) {
            decx::conv::CPUK::_conv2_rN_rect_flex_fp64_ST_rw4(
                DECX_PTR_SHF_XY_SAME_TYPE(src, _sum_prev_lenH, (k << 2) * _BLOCKED_CONV2_FP64_W_, Wsrc << 2),
                kernel,
                DECX_PTR_SHF_XY_SAME_TYPE(dst, _sum_prev_lenH, (k << 2) * _BLOCKED_CONV2_FP64_W_, Wdst << 2),
                make_uint2(_BLOCKED_CONV2_FP64_W_, f_mgrH->frag_left_over),
                ker_dims, Wsrc, Wdst, _loop);
        }
        if (f_mgrW->is_left) {
            const uint _sum_prev_lenW = proc_dim.x - f_mgrW->frag_left_over;
            decx::conv::CPUK::_conv2_rN_rect_flex_fp64_ST_rw4(
                DECX_PTR_SHF_XY_SAME_TYPE(src, _sum_prev_lenH, (_sum_prev_lenW << 2), Wsrc << 2),
                kernel,
                DECX_PTR_SHF_XY_SAME_TYPE(dst, _sum_prev_lenH, (_sum_prev_lenW << 2), Wdst << 2),
                make_uint2(f_mgrW->frag_left_over, f_mgrH->frag_left_over), ker_dims,
                Wsrc, Wdst, _loop);
        }
    }
}




_THREAD_CALL_ 
void decx::conv::CPUK::_conv2_rN_fp64_NB_SK_rw2(double* __restrict src,             double* __restrict kernel,
                                    double* __restrict dst,             const uint2 proc_dim, 
                                    const size_t page_size_src,         const size_t page_size_dst,
                                    const uint2 ker_dims,               const uint channel_size, 
                                    const uint Wsrc,                    const uint Wdst,
                                    const uint _loop)
{
    decx::utils::frag_manager f_mgrH, f_mgrW;
    decx::utils::frag_manager_gen_from_fragLen(&f_mgrH, proc_dim.y, _BLOCKED_CONV2_FP64_H_);
    decx::utils::frag_manager_gen_from_fragLen(&f_mgrW, proc_dim.x, _BLOCKED_CONV2_FP64_W_);

    for (int i = 0; i < channel_size; ++i) {
        decx::conv::CPUK::_conv2_rN_fp64_ST_rw2_unconfigured(src + i * page_size_src,
            kernel, dst + i * page_size_dst,
            proc_dim, ker_dims, Wsrc, Wdst, &f_mgrH, &f_mgrW, _loop);
    }
}



_THREAD_CALL_ 
void decx::conv::CPUK::_conv2_rN_fp64_NB_SK_rw4(double* __restrict src,             double* __restrict kernel,
                                    double* __restrict dst,             const uint2 proc_dim, 
                                    const size_t page_size_src,         const size_t page_size_dst,
                                    const uint2 ker_dims,               const uint channel_size, 
                                    const uint Wsrc,                    const uint Wdst,
                                    const uint _loop)
{
    decx::utils::frag_manager f_mgrH, f_mgrW;
    decx::utils::frag_manager_gen_from_fragLen(&f_mgrH, proc_dim.y, _BLOCKED_CONV2_FP64_H_);
    decx::utils::frag_manager_gen_from_fragLen(&f_mgrW, proc_dim.x, _BLOCKED_CONV2_FP64_W_);

    for (int i = 0; i < channel_size; ++i) {
        decx::conv::CPUK::_conv2_rN_fp64_ST_rw4_unconfigured(src + i * page_size_src,
            kernel, dst + i * page_size_dst,
            proc_dim, ker_dims, Wsrc, Wdst, &f_mgrH, &f_mgrW, _loop);
    }
}



_THREAD_CALL_ 
void decx::conv::CPUK::_conv2_r1_fp64_NB_SK(double* __restrict src,             double* __restrict kernel,
                                double* __restrict dst,             const uint2 proc_dim, 
                                const size_t page_size_src,         const size_t page_size_dst,
                                const uint2 ker_dims,               const uint channel_size, 
                                const uint Wsrc,                    const uint Wdst)
{
    decx::utils::frag_manager f_mgrH, f_mgrW;
    decx::utils::frag_manager_gen_from_fragLen(&f_mgrH, proc_dim.y, _BLOCKED_CONV2_FP64_H_);
    decx::utils::frag_manager_gen_from_fragLen(&f_mgrW, proc_dim.x, _BLOCKED_CONV2_FP64_W_);

    for (int i = 0; i < channel_size; ++i) {
        decx::conv::CPUK::_conv2_r1_fp64_ST_unconfigured(src + i * page_size_src,
            kernel, dst + i * page_size_dst,
            proc_dim, ker_dims, Wsrc, Wdst, &f_mgrH, &f_mgrW);
    }
}



_THREAD_CALL_ 
void decx::conv::CPUK::_conv2_r2_fp64_NB_SK(double* __restrict src,             double* __restrict kernel,
                                double* __restrict dst,             const uint2 proc_dim, 
                                const size_t page_size_src,         const size_t page_size_dst,
                                const uint2 ker_dims,               const uint channel_size, 
                                const uint Wsrc,                    const uint Wdst)
{
    decx::utils::frag_manager f_mgrH, f_mgrW;
    decx::utils::frag_manager_gen_from_fragLen(&f_mgrH, proc_dim.y, _BLOCKED_CONV2_FP64_H_);
    decx::utils::frag_manager_gen_from_fragLen(&f_mgrW, proc_dim.x, _BLOCKED_CONV2_FP64_W_);

    for (int i = 0; i < channel_size; ++i) {
        decx::conv::CPUK::_conv2_r2_fp64_ST_unconfigured(src + i * page_size_src,
            kernel, dst + i * page_size_dst,
            proc_dim, ker_dims, Wsrc, Wdst, &f_mgrH, &f_mgrW);
    }
}





void decx::conv::_conv2_rN_NB_SK_fp64_caller(double* src,
                                       double* kernel,
                                       double* dst,
                                       const uint2                  proc_dim,
                                       decx::utils::_thr_1D* t1D,
                                       decx::_C2_MK64* conv2_mk_props)
{
    double* tmp_src_ptr = src, * tmp_dst_ptr = dst;
    const size_t frag_size_src = conv2_mk_props->f_mgr->frag_len * (conv2_mk_props->Wsrc << 2);
    const size_t frag_size_dst = conv2_mk_props->f_mgr->frag_len * (proc_dim.x << 2);

    typedef void (*__called_ST_func)(double*, double*, double*, const uint2, const size_t, const size_t, const uint2,
        const uint, const uint, const uint, const uint);

    __called_ST_func __func = NULL;
    if (conv2_mk_props->reg_WL == 2) __func = decx::conv::CPUK::_conv2_rN_fp64_NB_SK_rw2;
    else __func = decx::conv::CPUK::_conv2_rN_fp64_NB_SK_rw4;

    if (conv2_mk_props->f_mgr->frag_left_over != 0) {
        for (int i = 0; i < t1D->total_thread - 1; ++i)
        {
            t1D->_async_thread[i] = decx::cpu::register_task(&decx::thread_pool, __func,
                tmp_src_ptr, kernel, tmp_dst_ptr,
                make_uint2(proc_dim.x, conv2_mk_props->f_mgr->frag_len),
                conv2_mk_props->page_size_src, conv2_mk_props->page_size_dst, conv2_mk_props->ker_dims,
                conv2_mk_props->channel_size, conv2_mk_props->Wsrc, conv2_mk_props->Wdst, conv2_mk_props->_loop);

            tmp_src_ptr += frag_size_src;
            tmp_dst_ptr += frag_size_dst;
        }
        t1D->_async_thread[t1D->total_thread - 1] = decx::cpu::register_task(&decx::thread_pool, __func,
            tmp_src_ptr, kernel, tmp_dst_ptr,
            make_uint2(proc_dim.x, conv2_mk_props->f_mgr->frag_left_over),
            conv2_mk_props->page_size_src, conv2_mk_props->page_size_dst, conv2_mk_props->ker_dims,
            conv2_mk_props->channel_size, conv2_mk_props->Wsrc, conv2_mk_props->Wdst, conv2_mk_props->_loop);
    }
    else {
        for (int i = 0; i < t1D->total_thread; ++i)
        {
            t1D->_async_thread[i] = decx::cpu::register_task(&decx::thread_pool, __func,
                tmp_src_ptr, kernel, tmp_dst_ptr,
                make_uint2(proc_dim.x, conv2_mk_props->f_mgr->frag_len),
                conv2_mk_props->page_size_src, conv2_mk_props->page_size_dst, conv2_mk_props->ker_dims,
                conv2_mk_props->channel_size, conv2_mk_props->Wsrc, conv2_mk_props->Wdst, conv2_mk_props->_loop);

            tmp_src_ptr += frag_size_src;
            tmp_dst_ptr += frag_size_dst;
        }
    }
    t1D->__sync_all_threads();
}



void decx::conv::_conv2_r1_r2_NB_SK_fp64_caller(double* src,
                                          double* kernel,
                                          double* dst,
                                          const uint2                  proc_dim,
                                          decx::utils::_thr_1D* t1D,
                                          decx::_C2_MK64* conv2_mk_props)
{
    double* tmp_src_ptr = src, * tmp_dst_ptr = dst;
    const size_t frag_size_src = conv2_mk_props->f_mgr->frag_len * (conv2_mk_props->Wsrc << 2);
    const size_t frag_size_dst = conv2_mk_props->f_mgr->frag_len * (proc_dim.x << 2);

    const uint _HKx = conv2_mk_props->ker_dims.x / 2;

    typedef void (*__called_ST_func)(double*, double*, double*, const uint2, const size_t, const size_t, const uint2,
        const uint, const uint, const uint);

    __called_ST_func __func = NULL;
    if (_HKx == 1) __func = decx::conv::CPUK::_conv2_r1_fp64_NB_SK;
    else __func = decx::conv::CPUK::_conv2_r2_fp64_NB_SK;

    if (conv2_mk_props->f_mgr->frag_left_over != 0) {
        for (int i = 0; i < t1D->total_thread - 1; ++i)
        {
            t1D->_async_thread[i] = decx::cpu::register_task(&decx::thread_pool, __func,
                tmp_src_ptr, kernel, tmp_dst_ptr,
                make_uint2(proc_dim.x, conv2_mk_props->f_mgr->frag_len),
                conv2_mk_props->page_size_src, conv2_mk_props->page_size_dst, conv2_mk_props->ker_dims,
                conv2_mk_props->channel_size, conv2_mk_props->Wsrc, conv2_mk_props->Wdst);

            tmp_src_ptr += frag_size_src;
            tmp_dst_ptr += frag_size_dst;
        }
        t1D->_async_thread[t1D->total_thread - 1] = decx::cpu::register_task(&decx::thread_pool, __func,
            tmp_src_ptr, kernel, tmp_dst_ptr,
            make_uint2(proc_dim.x, conv2_mk_props->f_mgr->frag_left_over),
            conv2_mk_props->page_size_src, conv2_mk_props->page_size_dst, conv2_mk_props->ker_dims,
            conv2_mk_props->channel_size, conv2_mk_props->Wsrc, conv2_mk_props->Wdst);
    }
    else {
        for (int i = 0; i < t1D->total_thread; ++i)
        {
            t1D->_async_thread[i] = decx::cpu::register_task(&decx::thread_pool, __func,
                tmp_src_ptr, kernel, tmp_dst_ptr,
                make_uint2(proc_dim.x, conv2_mk_props->f_mgr->frag_len),
                conv2_mk_props->page_size_src, conv2_mk_props->page_size_dst, conv2_mk_props->ker_dims,
                conv2_mk_props->channel_size, conv2_mk_props->Wsrc, conv2_mk_props->Wdst);

            tmp_src_ptr += frag_size_src;
            tmp_dst_ptr += frag_size_dst;
        }
    }
    t1D->__sync_all_threads();
}
