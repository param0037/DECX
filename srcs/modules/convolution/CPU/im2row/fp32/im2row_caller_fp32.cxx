/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/


#include "conv2_im2row_fp32.h"



void decx::conv_I2R::_im2row_caller_fp32(const float*                       src,
                               float*                             row_buf, 
                               float*                             I2C_buf, 
                               const uint                         _depth_v128,
                               const uint2                        ker_dims, 
                               const uint                         ker_wp, 
                               const uint                         src_dp_x_wp, 
                               const size_t                       WI2C, 
                               const uint2                        proc_dims_dst,
                               decx::utils::_thread_arrange_1D*   t1D)
{
    const float* loc_src = src;
    float* loc_I2C_buf = I2C_buf;

    decx::utils::frag_manager f_mgr;
    decx::utils::frag_manager_gen(&f_mgr, proc_dims_dst.y, t1D->total_thread);

    size_t frag_src = (size_t)f_mgr.frag_len * (size_t)src_dp_x_wp,
           frag_I2C = (size_t)f_mgr.frag_len * (size_t)proc_dims_dst.x * (size_t)WI2C,
           frag_row_buf = (size_t)ker_dims.x * (size_t)_depth_v128 * 4;

    for (int i = 0; i < t1D->total_thread - 1; ++i) {
        t1D->_async_thread[i] = decx::cpu::register_task(&decx::thread_pool,
            decx::conv::CPUK::_im2col_v128, loc_src, row_buf + i * frag_row_buf, loc_I2C_buf, _depth_v128,
            ker_dims, ker_wp, src_dp_x_wp, WI2C, make_uint2(proc_dims_dst.x, f_mgr.frag_len));

        loc_src += frag_src;
        loc_I2C_buf += frag_I2C;
    }
    uint _L = f_mgr.is_left ? f_mgr.frag_left_over : f_mgr.frag_len;
    t1D->_async_thread[t1D->total_thread - 1] = decx::cpu::register_task(&decx::thread_pool,
        decx::conv::CPUK::_im2col_v128, loc_src, row_buf + frag_row_buf * (t1D->total_thread - 1), loc_I2C_buf, _depth_v128,
        ker_dims, ker_wp, src_dp_x_wp, WI2C, make_uint2(proc_dims_dst.x, _L));

    t1D->__sync_all_threads();
}



void decx::conv_I2R::_im2row_caller_fp32_stride(const float*                       src, 
                                                float*                             row_buf, 
                                                float*                             I2C_buf, 
                                                const uint2                        strideXY,
                                                const uint                         _depth_v128,
                                                const uint2                        ker_dims, 
                                                const uint                         ker_wp, 
                                                const uint                         src_dp_x_wp, 
                                                const size_t                       WI2C, 
                                                const uint2                        proc_dims_dst,
                                                decx::utils::_thread_arrange_1D*   t1D)
{
    const float* loc_src = src;
    float* loc_I2C_buf = I2C_buf;

    decx::utils::frag_manager f_mgr;
    decx::utils::frag_manager_gen(&f_mgr, proc_dims_dst.y, t1D->total_thread);

    size_t frag_src = (size_t)f_mgr.frag_len * (size_t)src_dp_x_wp * (size_t)strideXY.y,
           frag_I2C = (size_t)f_mgr.frag_len * (size_t)proc_dims_dst.x * (size_t)WI2C,
           frag_row_buf = _depth_v128 * 4;

    for (int i = 0; i < t1D->total_thread - 1; ++i) {
        t1D->_async_thread[i] = decx::cpu::register_task(&decx::thread_pool,
            decx::conv::CPUK::_im2col_v128_stride, loc_src, row_buf + i * frag_row_buf, loc_I2C_buf, strideXY, _depth_v128,
            ker_dims, ker_wp, src_dp_x_wp, WI2C, make_uint2(proc_dims_dst.x, f_mgr.frag_len));

        loc_src += frag_src;
        loc_I2C_buf += frag_I2C;
    }
    uint _L = f_mgr.is_left ? f_mgr.frag_left_over : f_mgr.frag_len;
    t1D->_async_thread[t1D->total_thread - 1] = decx::cpu::register_task(&decx::thread_pool,
        decx::conv::CPUK::_im2col_v128_stride, loc_src, row_buf + frag_row_buf * (t1D->total_thread - 1), loc_I2C_buf, strideXY, _depth_v128,
        ker_dims, ker_wp, src_dp_x_wp, WI2C, make_uint2(proc_dims_dst.x, _L));

    t1D->__sync_all_threads();
}