/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#include "conv2_uint8_exec.h"


_THREAD_FUNCTION_
void decx::_conv2_rN_uint8_I32_ST(double* __restrict     src,
                             uint8_t* __restrict     kernel, 
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
    decx::utils::frag_manager_gen_from_fragLen(&f_mgrH, proc_dim.y, _BLOCKED_CONV2_UINT8_H_);
    decx::utils::frag_manager_gen_from_fragLen(&f_mgrW, proc_dim.x, _BLOCKED_CONV2_UINT8_W_);

    const uint _loopH = f_mgrH.is_left ? f_mgrH.frag_num - 1 : f_mgrH.frag_num;
    
    for (int i = 0; i < _loopH; ++i) 
    {
        const uint _loopW = f_mgrW.is_left ? f_mgrW.frag_num - 1 : f_mgrW.frag_num;

        for (int k = 0; k < _loopW; ++k) {
            decx::conv::CPUK::_conv2_rN_rect_fixed_uint8_I32_ST(
                DECX_PTR_SHF_XY_SAME_TYPE(src, i * _BLOCKED_CONV2_UINT8_H_, k * _BLOCKED_CONV2_UINT8_W_, Wsrc),
                kernel,
                DECX_PTR_SHF_XY_SAME_TYPE(dst, i * _BLOCKED_CONV2_UINT8_H_, (k << 3) * _BLOCKED_CONV2_UINT8_W_, Wdst << 3),
                ker_dims, Wsrc, Wdst, reg_WL, _loop);
        }
        if (f_mgrW.is_left) {
            const uint _sum_prev_lenW = proc_dim.x - f_mgrW.frag_left_over;
            decx::conv::CPUK::_conv2_rN_rect_flex_uint8_I32_ST(
                DECX_PTR_SHF_XY_SAME_TYPE(src, i * _BLOCKED_CONV2_UINT8_H_, _sum_prev_lenW, Wsrc),
                kernel,
                DECX_PTR_SHF_XY_SAME_TYPE(dst, i * _BLOCKED_CONV2_UINT8_H_, (_sum_prev_lenW << 3), Wdst << 3),
                make_uint2(f_mgrW.frag_left_over, _BLOCKED_CONV2_UINT8_H_), ker_dims,
                Wsrc, Wdst, reg_WL, _loop);
        }
    }
    
    if (f_mgrH.is_left)
    {
        const uint _sum_prev_lenH = proc_dim.y - f_mgrH.frag_left_over;
        const uint _loopW = f_mgrW.is_left ? f_mgrW.frag_num - 1 : f_mgrW.frag_num;

        for (int k = 0; k < _loopW; ++k) {
            decx::conv::CPUK::_conv2_rN_rect_flex_uint8_I32_ST(
                DECX_PTR_SHF_XY_SAME_TYPE(src, _sum_prev_lenH, k * _BLOCKED_CONV2_UINT8_W_, Wsrc),
                kernel,
                DECX_PTR_SHF_XY_SAME_TYPE(dst, _sum_prev_lenH, (k << 3) * _BLOCKED_CONV2_UINT8_W_, Wdst << 3),
                make_uint2(_BLOCKED_CONV2_UINT8_W_, f_mgrH.frag_left_over),
                ker_dims, Wsrc, Wdst, reg_WL, _loop);
        }
        if (f_mgrW.is_left) {
            const uint _sum_prev_lenW = proc_dim.x - f_mgrW.frag_left_over;
            decx::conv::CPUK::_conv2_rN_rect_flex_uint8_I32_ST(
                DECX_PTR_SHF_XY_SAME_TYPE(src, _sum_prev_lenH, _sum_prev_lenW, Wsrc),
                kernel,
                DECX_PTR_SHF_XY_SAME_TYPE(dst, _sum_prev_lenH, (_sum_prev_lenW << 3), Wdst << 3),
                make_uint2(f_mgrW.frag_left_over, f_mgrH.frag_left_over), ker_dims,
                Wsrc, Wdst, reg_WL, _loop);
        }
    }
}




_THREAD_FUNCTION_
void decx::_conv2_rN_uint8_F32_ST(double* __restrict     src,
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
    decx::utils::frag_manager_gen_from_fragLen(&f_mgrH, proc_dim.y, _BLOCKED_CONV2_UINT8_H_);
    decx::utils::frag_manager_gen_from_fragLen(&f_mgrW, proc_dim.x, _BLOCKED_CONV2_UINT8_W_);

    const uint _loopH = f_mgrH.is_left ? f_mgrH.frag_num - 1 : f_mgrH.frag_num;
    
    for (int i = 0; i < _loopH; ++i) 
    {
        const uint _loopW = f_mgrW.is_left ? f_mgrW.frag_num - 1 : f_mgrW.frag_num;

        for (int k = 0; k < _loopW; ++k) {
            decx::conv::CPUK::_conv2_rN_rect_fixed_uint8_F32_ST(
                DECX_PTR_SHF_XY_SAME_TYPE(src, i * _BLOCKED_CONV2_UINT8_H_, k * _BLOCKED_CONV2_UINT8_W_, Wsrc),
                kernel,
                DECX_PTR_SHF_XY_SAME_TYPE(dst, i * _BLOCKED_CONV2_UINT8_H_, (k << 3) * _BLOCKED_CONV2_UINT8_W_, Wdst << 3),
                ker_dims, Wsrc, Wdst, reg_WL, _loop);
        }
        if (f_mgrW.is_left) {
            const uint _sum_prev_lenW = proc_dim.x - f_mgrW.frag_left_over;
            decx::conv::CPUK::_conv2_rN_rect_flex_uint8_F32_ST(
                DECX_PTR_SHF_XY_SAME_TYPE(src, i * _BLOCKED_CONV2_UINT8_H_, _sum_prev_lenW, Wsrc),
                kernel,
                DECX_PTR_SHF_XY_SAME_TYPE(dst, i * _BLOCKED_CONV2_UINT8_H_, (_sum_prev_lenW << 3), Wdst << 3),
                make_uint2(f_mgrW.frag_left_over, _BLOCKED_CONV2_UINT8_H_), ker_dims,
                Wsrc, Wdst, reg_WL, _loop);
        }
    }
    
    if (f_mgrH.is_left)
    {
        const uint _sum_prev_lenH = proc_dim.y - f_mgrH.frag_left_over;
        const uint _loopW = f_mgrW.is_left ? f_mgrW.frag_num - 1 : f_mgrW.frag_num;

        for (int k = 0; k < _loopW; ++k) {
            decx::conv::CPUK::_conv2_rN_rect_flex_uint8_F32_ST(
                DECX_PTR_SHF_XY_SAME_TYPE(src, _sum_prev_lenH, k * _BLOCKED_CONV2_UINT8_W_, Wsrc),
                kernel,
                DECX_PTR_SHF_XY_SAME_TYPE(dst, _sum_prev_lenH, (k << 3) * _BLOCKED_CONV2_UINT8_W_, Wdst << 3),
                make_uint2(_BLOCKED_CONV2_UINT8_W_, f_mgrH.frag_left_over),
                ker_dims, Wsrc, Wdst, reg_WL, _loop);
        }
        if (f_mgrW.is_left) {
            const uint _sum_prev_lenW = proc_dim.x - f_mgrW.frag_left_over;
            decx::conv::CPUK::_conv2_rN_rect_flex_uint8_F32_ST(
                DECX_PTR_SHF_XY_SAME_TYPE(src, _sum_prev_lenH, _sum_prev_lenW, Wsrc),
                kernel,
                DECX_PTR_SHF_XY_SAME_TYPE(dst, _sum_prev_lenH, (_sum_prev_lenW << 3), Wdst << 3),
                make_uint2(f_mgrW.frag_left_over, f_mgrH.frag_left_over), ker_dims,
                Wsrc, Wdst, reg_WL, _loop);
        }
    }
}



_THREAD_FUNCTION_
void decx::_conv2_rN_uint8_uint8_ST(double* __restrict     src,
                             float* __restrict     kernel, 
                             double* __restrict     dst, 
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
    decx::utils::frag_manager_gen_from_fragLen(&f_mgrH, proc_dim.y, _BLOCKED_CONV2_UINT8_H_);
    decx::utils::frag_manager_gen_from_fragLen(&f_mgrW, proc_dim.x, _BLOCKED_CONV2_UINT8_W_);

    const uint _loopH = f_mgrH.is_left ? f_mgrH.frag_num - 1 : f_mgrH.frag_num;
    
    for (int i = 0; i < _loopH; ++i) 
    {
        const uint _loopW = f_mgrW.is_left ? f_mgrW.frag_num - 1 : f_mgrW.frag_num;

        for (int k = 0; k < _loopW; ++k) {
            decx::conv::CPUK::_conv2_rN_rect_fixed_uint8_uint8_ST(
                DECX_PTR_SHF_XY_SAME_TYPE(src, i * _BLOCKED_CONV2_UINT8_H_, k * 2 * _BLOCKED_CONV2_UINT8_W_, Wsrc * 2),
                kernel,
                DECX_PTR_SHF_XY_SAME_TYPE(dst, i * _BLOCKED_CONV2_UINT8_H_, k * 2 * _BLOCKED_CONV2_UINT8_W_, Wdst * 2),
                ker_dims, Wsrc, Wdst, reg_WL, _loop);
        }
        if (f_mgrW.is_left) {
            const uint _sum_prev_lenW = proc_dim.x - f_mgrW.frag_left_over;
            decx::conv::CPUK::_conv2_rN_rect_flex_uint8_uint8_ST(
                DECX_PTR_SHF_XY_SAME_TYPE(src, i * _BLOCKED_CONV2_UINT8_H_, _sum_prev_lenW * 2, Wsrc * 2),
                kernel,
                DECX_PTR_SHF_XY_SAME_TYPE(dst, i * _BLOCKED_CONV2_UINT8_H_, _sum_prev_lenW * 2, Wdst * 2),
                make_uint2(f_mgrW.frag_left_over, _BLOCKED_CONV2_UINT8_H_), ker_dims,
                Wsrc, Wdst, reg_WL, _loop);
        }
    }
    
    if (f_mgrH.is_left)
    {
        const uint _sum_prev_lenH = proc_dim.y - f_mgrH.frag_left_over;
        const uint _loopW = f_mgrW.is_left ? f_mgrW.frag_num - 1 : f_mgrW.frag_num;

        for (int k = 0; k < _loopW; ++k) {
            decx::conv::CPUK::_conv2_rN_rect_flex_uint8_uint8_ST(
                DECX_PTR_SHF_XY_SAME_TYPE(src, _sum_prev_lenH, k * _BLOCKED_CONV2_UINT8_W_ * 2, Wsrc * 2),
                kernel,
                DECX_PTR_SHF_XY_SAME_TYPE(dst, _sum_prev_lenH, k * _BLOCKED_CONV2_UINT8_W_ * 2, Wdst * 2),
                make_uint2(_BLOCKED_CONV2_UINT8_W_, f_mgrH.frag_left_over),
                ker_dims, Wsrc, Wdst, reg_WL, _loop);
        }
        if (f_mgrW.is_left) {
            const uint _sum_prev_lenW = proc_dim.x - f_mgrW.frag_left_over;
            decx::conv::CPUK::_conv2_rN_rect_flex_uint8_uint8_ST(
                DECX_PTR_SHF_XY_SAME_TYPE(src, _sum_prev_lenH, _sum_prev_lenW * 2, Wsrc * 2),
                kernel,
                DECX_PTR_SHF_XY_SAME_TYPE(dst, _sum_prev_lenH, _sum_prev_lenW * 2, Wdst * 2),
                make_uint2(f_mgrW.frag_left_over, f_mgrH.frag_left_over), ker_dims,
                Wsrc, Wdst, reg_WL, _loop);
        }
    }
}




void decx::conv::_conv2_rN_uint8_I32_caller(double*                src,
                                 uint8_t*                     kernel, 
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
    double* tmp_src_ptr = src;
    float *tmp_dst_ptr = dst;
    if (f_mgr->frag_left_over != 0) {
        size_t frag_src = (size_t)f_mgr->frag_len * (size_t)Wsrc;
        size_t frag_dst = (size_t)f_mgr->frag_len * (size_t)(Wdst << 3);

        for (int i = 0; i < t1D->total_thread - 1; ++i) {
            t1D->_async_thread[i] = decx::cpu::register_task_default(decx::_conv2_rN_uint8_I32_ST,
                tmp_src_ptr, kernel, tmp_dst_ptr,
                make_uint2(proc_dim.x, f_mgr->frag_len), ker_dims, Wsrc, Wdst, reg_WL, _loop);

            tmp_src_ptr += frag_src;
            tmp_dst_ptr += frag_dst;
        }
        t1D->_async_thread[t1D->total_thread - 1] = decx::cpu::register_task_default(decx::_conv2_rN_uint8_I32_ST,
            tmp_src_ptr, kernel, tmp_dst_ptr,
            make_uint2(proc_dim.x, f_mgr->frag_left_over), ker_dims, Wsrc, Wdst, reg_WL, _loop);
    }
    else {
        size_t frag_src = (size_t)f_mgr->frag_len * (size_t)Wsrc;
        size_t frag_dst = (size_t)f_mgr->frag_len * (size_t)(Wdst << 3);

        for (int i = 0; i < t1D->total_thread; ++i) {
            t1D->_async_thread[i] = decx::cpu::register_task_default(decx::_conv2_rN_uint8_I32_ST,
                tmp_src_ptr, kernel, tmp_dst_ptr,
                make_uint2(proc_dim.x, f_mgr->frag_len), ker_dims, Wsrc, Wdst, reg_WL, _loop);

            tmp_src_ptr += frag_src;
            tmp_dst_ptr += frag_dst;
        }
    }

    t1D->__sync_all_threads();
}




void decx::conv::_conv2_rN_uint8_F32_caller(double*                src,
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
    double* tmp_src_ptr = src;
    float *tmp_dst_ptr = dst;
    if (f_mgr->frag_left_over != 0) {
        size_t frag_src = (size_t)f_mgr->frag_len * (size_t)Wsrc;
        size_t frag_dst = (size_t)f_mgr->frag_len * (size_t)(Wdst << 3);

        for (int i = 0; i < t1D->total_thread - 1; ++i) {
            t1D->_async_thread[i] = decx::cpu::register_task_default(decx::_conv2_rN_uint8_F32_ST,
                tmp_src_ptr, kernel, tmp_dst_ptr,
                make_uint2(proc_dim.x, f_mgr->frag_len), ker_dims, Wsrc, Wdst, reg_WL, _loop);

            tmp_src_ptr += frag_src;
            tmp_dst_ptr += frag_dst;
        }
        t1D->_async_thread[t1D->total_thread - 1] = decx::cpu::register_task_default(decx::_conv2_rN_uint8_F32_ST,
            tmp_src_ptr, kernel, tmp_dst_ptr,
            make_uint2(proc_dim.x, f_mgr->frag_left_over), ker_dims, Wsrc, Wdst, reg_WL, _loop);
    }
    else {
        size_t frag_src = (size_t)f_mgr->frag_len * (size_t)Wsrc;
        size_t frag_dst = (size_t)f_mgr->frag_len * (size_t)(Wdst << 3);

        for (int i = 0; i < t1D->total_thread; ++i) {
            t1D->_async_thread[i] = decx::cpu::register_task_default(decx::_conv2_rN_uint8_F32_ST,
                tmp_src_ptr, kernel, tmp_dst_ptr,
                make_uint2(proc_dim.x, f_mgr->frag_len), ker_dims, Wsrc, Wdst, reg_WL, _loop);

            tmp_src_ptr += frag_src;
            tmp_dst_ptr += frag_dst;
        }
    }

    t1D->__sync_all_threads();
}




void decx::conv::_conv2_rN_uint8_uint8_caller(double*                src,
                                 float*                      kernel, 
                                 double*                      dst, 
                                 const uint2                 proc_dim, 
                                 const uint2                 ker_dims,
                                 const uint                  Wsrc,
                                 const uint                  Wdst,
                                 const ushort                reg_WL,
                                 decx::utils::_thr_1D*       t1D,
                                 decx::utils::frag_manager*  f_mgr,
                                 const uint                  _loop)
{
    double* tmp_src_ptr = src;
    double *tmp_dst_ptr = dst;
    if (f_mgr->frag_left_over != 0) {
        size_t frag_src = (size_t)f_mgr->frag_len * (size_t)Wsrc * 2;
        size_t frag_dst = (size_t)f_mgr->frag_len * (size_t)Wdst * 2;

        for (int i = 0; i < t1D->total_thread - 1; ++i) {
            t1D->_async_thread[i] = decx::cpu::register_task_default(decx::_conv2_rN_uint8_uint8_ST,
                tmp_src_ptr, kernel, tmp_dst_ptr,
                make_uint2(proc_dim.x, f_mgr->frag_len), ker_dims, Wsrc, Wdst, reg_WL, _loop);

            tmp_src_ptr += frag_src;
            tmp_dst_ptr += frag_dst;
        }
        t1D->_async_thread[t1D->total_thread - 1] = decx::cpu::register_task_default(decx::_conv2_rN_uint8_uint8_ST,
            tmp_src_ptr, kernel, tmp_dst_ptr,
            make_uint2(proc_dim.x, f_mgr->frag_left_over), ker_dims, Wsrc, Wdst, reg_WL, _loop);
    }
    else {
        size_t frag_src = (size_t)f_mgr->frag_len * (size_t)Wsrc * 2;
        size_t frag_dst = (size_t)f_mgr->frag_len * (size_t)Wdst * 2;

        for (int i = 0; i < t1D->total_thread; ++i) {
            t1D->_async_thread[i] = decx::cpu::register_task_default(decx::_conv2_rN_uint8_uint8_ST,
                tmp_src_ptr, kernel, tmp_dst_ptr,
                make_uint2(proc_dim.x, f_mgr->frag_len), ker_dims, Wsrc, Wdst, reg_WL, _loop);

            tmp_src_ptr += frag_src;
            tmp_dst_ptr += frag_dst;
        }
    }

    t1D->__sync_all_threads();
}