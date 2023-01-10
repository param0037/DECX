/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/


#include "rcp2_SQDIFF_fp32_exec.h"



_THREAD_FUNCTION_ void 
decx::rcp::CPUK::_rcp2_SQDIFF_rN_fp32_ST(float* __restrict     src,
                             float* __restrict     kernel, 
                             float* __restrict     dst, 
                             const uint2           proc_dim, 
                             const uint2           ker_dims,
                             const uint            Wsrc,
                             const uint            Wdst,
                             const ushort          reg_WL,
                             const uint            _loop)
{
    __m256 res_vec8;
    size_t dex_src = 0, dex_dst = 0;

    decx::utils::frag_manager f_mgrH, f_mgrW;
    decx::utils::frag_manager_gen_from_fragLen(&f_mgrH, proc_dim.y, _BLOCKED_RCP2_FP32_H_);
    decx::utils::frag_manager_gen_from_fragLen(&f_mgrW, proc_dim.x, _BLOCKED_RCP2_FP32_W_);

    const uint _loopH = f_mgrH.is_left ? f_mgrH.frag_num - 1 : f_mgrH.frag_num;
    
    for (int i = 0; i < _loopH; ++i) 
    {
        const uint _loopW = f_mgrW.is_left ? f_mgrW.frag_num - 1 : f_mgrW.frag_num;

        for (int k = 0; k < _loopW; ++k) {
            decx::rcp::CPUK::_rcp2_SQDIFF_rN_rect_fixed_fp32_ST(
                DECX_PTR_SHF_XY_SAME_TYPE(src, i * _BLOCKED_RCP2_FP32_H_, (k << 3) * _BLOCKED_RCP2_FP32_W_, Wsrc << 3),
                kernel,
                DECX_PTR_SHF_XY_SAME_TYPE(dst, i * _BLOCKED_RCP2_FP32_H_, (k << 3) * _BLOCKED_RCP2_FP32_W_, Wdst << 3),
                ker_dims, Wsrc, Wdst, reg_WL, _loop);
        }
        if (f_mgrW.is_left) {
            const uint _sum_prev_lenW = proc_dim.x - f_mgrW.frag_left_over;
            decx::rcp::CPUK::_rcp2_SQDIFF_rN_rect_flex_fp32_ST(
                DECX_PTR_SHF_XY_SAME_TYPE(src, i * _BLOCKED_RCP2_FP32_H_, (_sum_prev_lenW << 3), Wsrc << 3),
                kernel,
                DECX_PTR_SHF_XY_SAME_TYPE(dst, i * _BLOCKED_RCP2_FP32_H_, (_sum_prev_lenW << 3), Wdst << 3),
                make_uint2(f_mgrW.frag_left_over, _BLOCKED_RCP2_FP32_H_), ker_dims,
                Wsrc, Wdst, reg_WL, _loop);
        }
    }
    
    if (f_mgrH.is_left)
    {
        const uint _sum_prev_lenH = proc_dim.y - f_mgrH.frag_left_over;
        const uint _loopW = f_mgrW.is_left ? f_mgrW.frag_num - 1 : f_mgrW.frag_num;

        for (int k = 0; k < _loopW; ++k) {
            decx::rcp::CPUK::_rcp2_SQDIFF_rN_rect_flex_fp32_ST(
                DECX_PTR_SHF_XY_SAME_TYPE(src, _sum_prev_lenH, (k << 3) * _BLOCKED_RCP2_FP32_W_, Wsrc << 3),
                kernel,
                DECX_PTR_SHF_XY_SAME_TYPE(dst, _sum_prev_lenH, (k << 3) * _BLOCKED_RCP2_FP32_W_, Wdst << 3),
                make_uint2(_BLOCKED_RCP2_FP32_W_, f_mgrH.frag_left_over),
                ker_dims, Wsrc, Wdst, reg_WL, _loop);
        }
        if (f_mgrW.is_left) {
            const uint _sum_prev_lenW = proc_dim.x - f_mgrW.frag_left_over;
            decx::rcp::CPUK::_rcp2_SQDIFF_rN_rect_flex_fp32_ST(
                DECX_PTR_SHF_XY_SAME_TYPE(src, _sum_prev_lenH, (_sum_prev_lenW << 3), Wsrc << 3),
                kernel,
                DECX_PTR_SHF_XY_SAME_TYPE(dst, _sum_prev_lenH, (_sum_prev_lenW << 3), Wdst << 3),
                make_uint2(f_mgrW.frag_left_over, f_mgrH.frag_left_over), ker_dims,
                Wsrc, Wdst, reg_WL, _loop);
        }
    }
}



_THREAD_FUNCTION_ void 
decx::rcp::CPUK::_rcp2_SQDIFF_r1_r4_fp32_ST(float* __restrict   src,
                                float* __restrict   kernel, 
                                float* __restrict   dst, 
                                const uint2         proc_dim, 
                                const uint2         ker_dims,
                                const uint          Wsrc,
                                const uint          Wdst)
{
    __m256 res_vec8;
    size_t dex_src = 0, dex_dst = 0;

    decx::utils::frag_manager f_mgrH, f_mgrW;
    decx::utils::frag_manager_gen_from_fragLen(&f_mgrH, proc_dim.y, _BLOCKED_RCP2_FP32_H_);
    decx::utils::frag_manager_gen_from_fragLen(&f_mgrW, proc_dim.x, _BLOCKED_RCP2_FP32_W_);

    const uint _loopH = f_mgrH.is_left ? f_mgrH.frag_num - 1 : f_mgrH.frag_num;
    
    for (int i = 0; i < _loopH; ++i) 
    {
        const uint _loopW = f_mgrW.is_left ? f_mgrW.frag_num - 1 : f_mgrW.frag_num;

        for (int k = 0; k < _loopW; ++k) {
            decx::rcp::CPUK::_rcp2_SQDIFF_r1_r4_rect_fixed_fp32_ST(
                DECX_PTR_SHF_XY_SAME_TYPE(src, i * _BLOCKED_RCP2_FP32_H_, (k << 3) * _BLOCKED_RCP2_FP32_W_, Wsrc << 3),
                kernel,
                DECX_PTR_SHF_XY_SAME_TYPE(dst, i * _BLOCKED_RCP2_FP32_H_, (k << 3) * _BLOCKED_RCP2_FP32_W_, Wdst << 3),
                ker_dims, Wsrc, Wdst);
        }
        if (f_mgrW.is_left) {
            const uint _sum_prev_lenW = proc_dim.x - f_mgrW.frag_left_over;
            decx::rcp::CPUK::_rcp2_SQDIFF_r1_r4_rect_flex_fp32_ST(
                DECX_PTR_SHF_XY_SAME_TYPE(src, i * _BLOCKED_RCP2_FP32_H_, (_sum_prev_lenW << 3), Wsrc << 3),
                kernel,
                DECX_PTR_SHF_XY_SAME_TYPE(dst, i * _BLOCKED_RCP2_FP32_H_, (_sum_prev_lenW << 3), Wdst << 3),
                make_uint2(f_mgrW.frag_left_over, _BLOCKED_RCP2_FP32_H_), ker_dims,
                Wsrc, Wdst);
        }
    }
    
    if (f_mgrH.is_left)
    {
        const uint _sum_prev_lenH = proc_dim.y - f_mgrH.frag_left_over;
        const uint _loopW = f_mgrW.is_left ? f_mgrW.frag_num - 1 : f_mgrW.frag_num;

        for (int k = 0; k < _loopW; ++k) {
            decx::rcp::CPUK::_rcp2_SQDIFF_r1_r4_rect_flex_fp32_ST(
                DECX_PTR_SHF_XY_SAME_TYPE(src, _sum_prev_lenH, (k << 3) * _BLOCKED_RCP2_FP32_W_, Wsrc << 3),
                kernel,
                DECX_PTR_SHF_XY_SAME_TYPE(dst, _sum_prev_lenH, (k << 3) * _BLOCKED_RCP2_FP32_W_, Wdst << 3),
                make_uint2(_BLOCKED_RCP2_FP32_W_, f_mgrH.frag_left_over),
                ker_dims, Wsrc, Wdst);
        }
        if (f_mgrW.is_left) {
            const uint _sum_prev_lenW = proc_dim.x - f_mgrW.frag_left_over;
            decx::rcp::CPUK::_rcp2_SQDIFF_r1_r4_rect_flex_fp32_ST(
                DECX_PTR_SHF_XY_SAME_TYPE(src, _sum_prev_lenH, (_sum_prev_lenW << 3), Wsrc << 3),
                kernel,
                DECX_PTR_SHF_XY_SAME_TYPE(dst, _sum_prev_lenH, (_sum_prev_lenW << 3), Wdst << 3),
                make_uint2(f_mgrW.frag_left_over, f_mgrH.frag_left_over), ker_dims,
                Wsrc, Wdst);
        }
    }
}



_THREAD_FUNCTION_ void 
decx::rcp::CPUK::_rcp2_SQDIFF_norm_rN_fp32_ST(float* __restrict     src,
                             float* __restrict     kernel, 
                             float* __restrict     dst, 
                             const float           _sqrt_k_sum,
                             const uint2           proc_dim, 
                             const uint2           ker_dims,
                             const uint            Wsrc,
                             const uint            Wdst,
                             const ushort          reg_WL,
                             const uint            _loop)
{
    __m256 res_vec8;
    size_t dex_src = 0, dex_dst = 0;

    decx::utils::frag_manager f_mgrH, f_mgrW;
    decx::utils::frag_manager_gen_from_fragLen(&f_mgrH, proc_dim.y, _BLOCKED_RCP2_FP32_H_);
    decx::utils::frag_manager_gen_from_fragLen(&f_mgrW, proc_dim.x, _BLOCKED_RCP2_FP32_W_);

    const uint _loopH = f_mgrH.is_left ? f_mgrH.frag_num - 1 : f_mgrH.frag_num;
    
    for (int i = 0; i < _loopH; ++i) 
    {
        const uint _loopW = f_mgrW.is_left ? f_mgrW.frag_num - 1 : f_mgrW.frag_num;

        for (int k = 0; k < _loopW; ++k) {
            decx::rcp::CPUK::_rcp2_SQDIFF_norm_rN_rect_fixed_fp32_ST(
                DECX_PTR_SHF_XY_SAME_TYPE(src, i * _BLOCKED_RCP2_FP32_H_, (k << 3) * _BLOCKED_RCP2_FP32_W_, Wsrc << 3),
                kernel,
                DECX_PTR_SHF_XY_SAME_TYPE(dst, i * _BLOCKED_RCP2_FP32_H_, (k << 3) * _BLOCKED_RCP2_FP32_W_, Wdst << 3),
                _sqrt_k_sum,
                ker_dims, Wsrc, Wdst, reg_WL, _loop);
        }
        if (f_mgrW.is_left) {
            const uint _sum_prev_lenW = proc_dim.x - f_mgrW.frag_left_over;
            decx::rcp::CPUK::_rcp2_SQDIFF_norm_rN_rect_flex_fp32_ST(
                DECX_PTR_SHF_XY_SAME_TYPE(src, i * _BLOCKED_RCP2_FP32_H_, (_sum_prev_lenW << 3), Wsrc << 3),
                kernel,
                DECX_PTR_SHF_XY_SAME_TYPE(dst, i * _BLOCKED_RCP2_FP32_H_, (_sum_prev_lenW << 3), Wdst << 3),
                _sqrt_k_sum,
                make_uint2(f_mgrW.frag_left_over, _BLOCKED_RCP2_FP32_H_), ker_dims,
                Wsrc, Wdst, reg_WL, _loop);
        }
    }
    
    if (f_mgrH.is_left)
    {
        const uint _sum_prev_lenH = proc_dim.y - f_mgrH.frag_left_over;
        const uint _loopW = f_mgrW.is_left ? f_mgrW.frag_num - 1 : f_mgrW.frag_num;

        for (int k = 0; k < _loopW; ++k) {
            decx::rcp::CPUK::_rcp2_SQDIFF_norm_rN_rect_flex_fp32_ST(
                DECX_PTR_SHF_XY_SAME_TYPE(src, _sum_prev_lenH, (k << 3) * _BLOCKED_RCP2_FP32_W_, Wsrc << 3),
                kernel,
                DECX_PTR_SHF_XY_SAME_TYPE(dst, _sum_prev_lenH, (k << 3) * _BLOCKED_RCP2_FP32_W_, Wdst << 3),
                _sqrt_k_sum,
                make_uint2(_BLOCKED_RCP2_FP32_W_, f_mgrH.frag_left_over),
                ker_dims, Wsrc, Wdst, reg_WL, _loop);
        }
        if (f_mgrW.is_left) {
            const uint _sum_prev_lenW = proc_dim.x - f_mgrW.frag_left_over;
            decx::rcp::CPUK::_rcp2_SQDIFF_norm_rN_rect_flex_fp32_ST(
                DECX_PTR_SHF_XY_SAME_TYPE(src, _sum_prev_lenH, (_sum_prev_lenW << 3), Wsrc << 3),
                kernel,
                DECX_PTR_SHF_XY_SAME_TYPE(dst, _sum_prev_lenH, (_sum_prev_lenW << 3), Wdst << 3),
                _sqrt_k_sum,
                make_uint2(f_mgrW.frag_left_over, f_mgrH.frag_left_over), ker_dims,
                Wsrc, Wdst, reg_WL, _loop);
        }
    }
}



_THREAD_FUNCTION_ void 
decx::rcp::CPUK::_rcp2_SQDIFF_norm_r1_r4_fp32_ST(float* __restrict   src,
                                float* __restrict   kernel, 
                                float* __restrict   dst, 
                                const float         _sqrt_k_sum,
                                const uint2         proc_dim, 
                                const uint2         ker_dims,
                                const uint          Wsrc,
                                const uint          Wdst)
{
    __m256 res_vec8;
    size_t dex_src = 0, dex_dst = 0;

    decx::utils::frag_manager f_mgrH, f_mgrW;
    decx::utils::frag_manager_gen_from_fragLen(&f_mgrH, proc_dim.y, _BLOCKED_RCP2_FP32_H_);
    decx::utils::frag_manager_gen_from_fragLen(&f_mgrW, proc_dim.x, _BLOCKED_RCP2_FP32_W_);

    const uint _loopH = f_mgrH.is_left ? f_mgrH.frag_num - 1 : f_mgrH.frag_num;
    
    for (int i = 0; i < _loopH; ++i) 
    {
        const uint _loopW = f_mgrW.is_left ? f_mgrW.frag_num - 1 : f_mgrW.frag_num;

        for (int k = 0; k < _loopW; ++k) {
            decx::rcp::CPUK::_rcp2_SQDIFF_norm_r1_r4_rect_fixed_fp32_ST(
                DECX_PTR_SHF_XY_SAME_TYPE(src, i * _BLOCKED_RCP2_FP32_H_, (k << 3) * _BLOCKED_RCP2_FP32_W_, Wsrc << 3),
                kernel,
                DECX_PTR_SHF_XY_SAME_TYPE(dst, i * _BLOCKED_RCP2_FP32_H_, (k << 3) * _BLOCKED_RCP2_FP32_W_, Wdst << 3),
                _sqrt_k_sum,
                ker_dims, Wsrc, Wdst);
        }
        if (f_mgrW.is_left) {
            const uint _sum_prev_lenW = proc_dim.x - f_mgrW.frag_left_over;
            decx::rcp::CPUK::_rcp2_SQDIFF_norm_r1_r4_rect_flex_fp32_ST(
                DECX_PTR_SHF_XY_SAME_TYPE(src, i * _BLOCKED_RCP2_FP32_H_, (_sum_prev_lenW << 3), Wsrc << 3),
                kernel,
                DECX_PTR_SHF_XY_SAME_TYPE(dst, i * _BLOCKED_RCP2_FP32_H_, (_sum_prev_lenW << 3), Wdst << 3),
                _sqrt_k_sum,
                make_uint2(f_mgrW.frag_left_over, _BLOCKED_RCP2_FP32_H_), ker_dims,
                Wsrc, Wdst);
        }
    }
    
    if (f_mgrH.is_left)
    {
        const uint _sum_prev_lenH = proc_dim.y - f_mgrH.frag_left_over;
        const uint _loopW = f_mgrW.is_left ? f_mgrW.frag_num - 1 : f_mgrW.frag_num;

        for (int k = 0; k < _loopW; ++k) {
            decx::rcp::CPUK::_rcp2_SQDIFF_norm_r1_r4_rect_flex_fp32_ST(
                DECX_PTR_SHF_XY_SAME_TYPE(src, _sum_prev_lenH, (k << 3) * _BLOCKED_RCP2_FP32_W_, Wsrc << 3),
                kernel,
                DECX_PTR_SHF_XY_SAME_TYPE(dst, _sum_prev_lenH, (k << 3) * _BLOCKED_RCP2_FP32_W_, Wdst << 3),
                _sqrt_k_sum,
                make_uint2(_BLOCKED_RCP2_FP32_W_, f_mgrH.frag_left_over),
                ker_dims, Wsrc, Wdst);
        }
        if (f_mgrW.is_left) {
            const uint _sum_prev_lenW = proc_dim.x - f_mgrW.frag_left_over;
            decx::rcp::CPUK::_rcp2_SQDIFF_norm_r1_r4_rect_flex_fp32_ST(
                DECX_PTR_SHF_XY_SAME_TYPE(src, _sum_prev_lenH, (_sum_prev_lenW << 3), Wsrc << 3),
                kernel,
                DECX_PTR_SHF_XY_SAME_TYPE(dst, _sum_prev_lenH, (_sum_prev_lenW << 3), Wdst << 3),
                _sqrt_k_sum,
                make_uint2(f_mgrW.frag_left_over, f_mgrH.frag_left_over), ker_dims,
                Wsrc, Wdst);
        }
    }
}



void decx::rcp::_rcp2_SQDIFF_rN_fp32_caller(float*                src,
                                 float*                      kernel, 
                                 float*                      dst, 
                                 const uint2                 proc_dim, 
                                 const uint2                 ker_dims,
                                 const uint                  Wsrc,
                                 const uint                  Wdst,
                                 const ushort                reg_WL,
                                 decx::utils::_thr_1D*       t1D,
                                 decx::utils::frag_manager*  f_mgr,
                                 const uint                  _loop)
{
    float* tmp_src_ptr = src, * tmp_dst_ptr = dst;
    if (f_mgr->frag_left_over != 0) {
        size_t frag_src = (size_t)f_mgr->frag_len * (size_t)(Wsrc << 3);
        size_t frag_dst = (size_t)f_mgr->frag_len * (size_t)(proc_dim.x << 3);

        for (int i = 0; i < t1D->total_thread - 1; ++i) {
            t1D->_async_thread[i] = decx::cpu::register_task(&decx::thread_pool, decx::rcp::CPUK::_rcp2_SQDIFF_rN_fp32_ST,
                tmp_src_ptr, kernel, tmp_dst_ptr,
                make_uint2(proc_dim.x, f_mgr->frag_len), ker_dims, Wsrc, Wdst, reg_WL, _loop);

            tmp_src_ptr += frag_src;
            tmp_dst_ptr += frag_dst;
        }
        t1D->_async_thread[t1D->total_thread - 1] = decx::cpu::register_task(&decx::thread_pool,decx::rcp::CPUK::_rcp2_SQDIFF_rN_fp32_ST,
            tmp_src_ptr, kernel, tmp_dst_ptr,
            make_uint2(proc_dim.x, f_mgr->frag_left_over), ker_dims, Wsrc, Wdst, reg_WL, _loop);
    }
    else {
        size_t frag_src = (size_t)f_mgr->frag_len * (size_t)(Wsrc << 3);
        size_t frag_dst = (size_t)f_mgr->frag_len * (size_t)(proc_dim.x << 3);

        for (int i = 0; i < t1D->total_thread; ++i) {
            t1D->_async_thread[i] = decx::cpu::register_task(&decx::thread_pool, decx::rcp::CPUK::_rcp2_SQDIFF_rN_fp32_ST,
                tmp_src_ptr, kernel, tmp_dst_ptr,
                make_uint2(proc_dim.x, f_mgr->frag_len), ker_dims, Wsrc, Wdst, reg_WL, _loop);

            tmp_src_ptr += frag_src;
            tmp_dst_ptr += frag_dst;
        }
    }

    t1D->__sync_all_threads();
}



void decx::rcp::_rcp2_SQDIFF_r1_r4_fp32_caller(float*                      src,
                                               float*                      kernel, 
                                               float*                      dst, 
                                               const uint2                 proc_dim, 
                                               const uint2                 ker_dims,
                                               const uint                  Wsrc,
                                               const uint                  Wdst,
                                               decx::utils::_thr_1D*       t1D,
                                               decx::utils::frag_manager*  f_mgr)
{
    float* tmp_src_ptr = src, * tmp_dst_ptr = dst;
    if (f_mgr->frag_left_over != 0) {
        size_t frag_src = (size_t)f_mgr->frag_len * (size_t)(Wsrc << 3);
        size_t frag_dst = (size_t)f_mgr->frag_len * (size_t)(proc_dim.x << 3);

        for (int i = 0; i < t1D->total_thread - 1; ++i) {
            t1D->_async_thread[i] = decx::cpu::register_task(&decx::thread_pool,decx::rcp::CPUK::_rcp2_SQDIFF_r1_r4_fp32_ST,
                tmp_src_ptr, kernel, tmp_dst_ptr,
                make_uint2(proc_dim.x, f_mgr->frag_len), ker_dims, Wsrc, Wdst);

            tmp_src_ptr += frag_src;
            tmp_dst_ptr += frag_dst;
        }
        t1D->_async_thread[t1D->total_thread - 1] = decx::cpu::register_task(&decx::thread_pool,decx::rcp::CPUK::_rcp2_SQDIFF_r1_r4_fp32_ST,
            tmp_src_ptr, kernel, tmp_dst_ptr,
            make_uint2(proc_dim.x, f_mgr->frag_left_over), ker_dims, Wsrc, Wdst);
    }
    else {
        size_t frag_src = (size_t)f_mgr->frag_len * (size_t)(Wsrc << 3);
        size_t frag_dst = (size_t)f_mgr->frag_len * (size_t)(proc_dim.x << 3);

        for (int i = 0; i < t1D->total_thread; ++i) {
            t1D->_async_thread[i] = decx::cpu::register_task(&decx::thread_pool,decx::rcp::CPUK::_rcp2_SQDIFF_r1_r4_fp32_ST,
                tmp_src_ptr, kernel, tmp_dst_ptr,
                make_uint2(proc_dim.x, f_mgr->frag_len), ker_dims, Wsrc, Wdst);

            tmp_src_ptr += frag_src;
            tmp_dst_ptr += frag_dst;
        }
    }

    t1D->__sync_all_threads();
}




void decx::rcp::_rcp2_SQDIFF_NORM_rN_fp32_caller(float*                src,
                                 float*                      kernel, 
                                 float*                      dst, 
                                 const float                 _sqrt_k_sum,
                                 const uint2                 proc_dim, 
                                 const uint2                 ker_dims,
                                 const uint                  Wsrc,
                                 const uint                  Wdst,
                                 const ushort                reg_WL,
                                 decx::utils::_thr_1D*       t1D,
                                 decx::utils::frag_manager*  f_mgr,
                                 const uint                  _loop)
{
    float* tmp_src_ptr = src, * tmp_dst_ptr = dst;
    if (f_mgr->frag_left_over != 0) {
        size_t frag_src = (size_t)f_mgr->frag_len * (size_t)(Wsrc << 3);
        size_t frag_dst = (size_t)f_mgr->frag_len * (size_t)(proc_dim.x << 3);

        for (int i = 0; i < t1D->total_thread - 1; ++i) {
            t1D->_async_thread[i] = decx::cpu::register_task(&decx::thread_pool, decx::rcp::CPUK::_rcp2_SQDIFF_norm_rN_fp32_ST,
                tmp_src_ptr, kernel, tmp_dst_ptr, _sqrt_k_sum,
                make_uint2(proc_dim.x, f_mgr->frag_len), ker_dims, Wsrc, Wdst, reg_WL, _loop);

            tmp_src_ptr += frag_src;
            tmp_dst_ptr += frag_dst;
        }
        t1D->_async_thread[t1D->total_thread - 1] = decx::cpu::register_task(&decx::thread_pool,decx::rcp::CPUK::_rcp2_SQDIFF_norm_rN_fp32_ST,
            tmp_src_ptr, kernel, tmp_dst_ptr, _sqrt_k_sum,
            make_uint2(proc_dim.x, f_mgr->frag_left_over), ker_dims, Wsrc, Wdst, reg_WL, _loop);
    }
    else {
        size_t frag_src = (size_t)f_mgr->frag_len * (size_t)(Wsrc << 3);
        size_t frag_dst = (size_t)f_mgr->frag_len * (size_t)(proc_dim.x << 3);

        for (int i = 0; i < t1D->total_thread; ++i) {
            t1D->_async_thread[i] = decx::cpu::register_task(&decx::thread_pool, decx::rcp::CPUK::_rcp2_SQDIFF_norm_rN_fp32_ST,
                tmp_src_ptr, kernel, tmp_dst_ptr, _sqrt_k_sum,
                make_uint2(proc_dim.x, f_mgr->frag_len), ker_dims, Wsrc, Wdst, reg_WL, _loop);

            tmp_src_ptr += frag_src;
            tmp_dst_ptr += frag_dst;
        }
    }

    t1D->__sync_all_threads();
}



void decx::rcp::_rcp2_SQDIFF_NORM_r1_r4_fp32_caller(float*                      src,
                                               float*                      kernel, 
                                               float*                      dst,
                                               const float                 _sqrt_k_sum,
                                               const uint2                 proc_dim, 
                                               const uint2                 ker_dims,
                                               const uint                  Wsrc,
                                               const uint                  Wdst,
                                               decx::utils::_thr_1D*       t1D,
                                               decx::utils::frag_manager*  f_mgr)
{
    float* tmp_src_ptr = src, * tmp_dst_ptr = dst;
    if (f_mgr->frag_left_over != 0) {
        size_t frag_src = (size_t)f_mgr->frag_len * (size_t)(Wsrc << 3);
        size_t frag_dst = (size_t)f_mgr->frag_len * (size_t)(proc_dim.x << 3);

        for (int i = 0; i < t1D->total_thread - 1; ++i) {
            t1D->_async_thread[i] = decx::cpu::register_task(&decx::thread_pool,decx::rcp::CPUK::_rcp2_SQDIFF_norm_r1_r4_fp32_ST,
                tmp_src_ptr, kernel, tmp_dst_ptr, _sqrt_k_sum,
                make_uint2(proc_dim.x, f_mgr->frag_len), ker_dims, Wsrc, Wdst);

            tmp_src_ptr += frag_src;
            tmp_dst_ptr += frag_dst;
        }
        t1D->_async_thread[t1D->total_thread - 1] = decx::cpu::register_task(&decx::thread_pool,decx::rcp::CPUK::_rcp2_SQDIFF_norm_r1_r4_fp32_ST,
            tmp_src_ptr, kernel, tmp_dst_ptr, _sqrt_k_sum,
            make_uint2(proc_dim.x, f_mgr->frag_left_over), ker_dims, Wsrc, Wdst);
    }
    else {
        size_t frag_src = (size_t)f_mgr->frag_len * (size_t)(Wsrc << 3);
        size_t frag_dst = (size_t)f_mgr->frag_len * (size_t)(proc_dim.x << 3);

        for (int i = 0; i < t1D->total_thread; ++i) {
            t1D->_async_thread[i] = decx::cpu::register_task(&decx::thread_pool,decx::rcp::CPUK::_rcp2_SQDIFF_norm_r1_r4_fp32_ST,
                tmp_src_ptr, kernel, tmp_dst_ptr, _sqrt_k_sum,
                make_uint2(proc_dim.x, f_mgr->frag_len), ker_dims, Wsrc, Wdst);

            tmp_src_ptr += frag_src;
            tmp_dst_ptr += frag_dst;
        }
    }

    t1D->__sync_all_threads();
}