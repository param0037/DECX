/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/

#ifndef _CONV2_FP32_EXEC_H_
#define _CONV2_FP32_EXEC_H_


#include "../../../../core/thread_management/thread_pool.h"
#include "../../../../core/thread_management/thread_arrange.h"
#include "conv2_fp32_K_loop_core.h"
#include "../../../conv_utils.h"


#define _BLOCKED_CONV2_FP32_H_ 8
#define _BLOCKED_CONV2_FP32_W_ 8


namespace decx
{
    namespace conv {
        namespace CPUK {
            static _THREAD_CALL_ void _conv2_rN_rect_fixed_fp32_ST(float* __restrict src, float* kernel, float* __restrict dst,
                const uint2 ker_dims, const uint Wsrc, const uint Wdst, const ushort reg_WL, const uint _loop);



            static _THREAD_CALL_ void _conv2_rN_rect_flex_fp32_ST(float* __restrict src, float* kernel, float* __restrict dst,
                const uint2 proc_dim, const uint2 ker_dims, const uint Wsrc, const uint Wdst, const ushort reg_WL, const uint _loop);



            static _THREAD_CALL_ void _conv2_r1_r4_rect_fixed_fp32_ST(float* __restrict src, float* kernel, float* __restrict dst,
                const uint2 ker_dims, const uint Wsrc, const uint Wdst);



            static _THREAD_CALL_ void _conv2_r1_r4_rect_flex_fp32_ST(float* __restrict src, float* kernel, float* __restrict dst,
                const uint2 proc_dim, const uint2 ker_dims, const uint Wsrc, const uint Wdst);
        }
    }
}



_THREAD_CALL_
void decx::conv::CPUK::_conv2_rN_rect_fixed_fp32_ST(float* __restrict   src, 
                                        float* __restrict   kernel, 
                                        float* __restrict   dst,
                                        const uint2         ker_dims,
                                        const uint          Wsrc,
                                        const uint          Wdst,
                                        const ushort        reg_WL,
                                        const uint          _loop)
{
    __m256 res_vec8;
    size_t dex_src = 0, dex_dst = 0;

    for (int i = 0; i < _BLOCKED_CONV2_FP32_H_; ++i) {
#pragma unroll _BLOCKED_CONV2_FP32_W_
        for (int j = 0; j < _BLOCKED_CONV2_FP32_W_; ++j) {
            res_vec8 = decx::conv::CPUK::_conv2_fp32_loop_in_kernel_Nregs(src + dex_src, kernel, ker_dims, reg_WL, Wsrc, _loop);
            _mm256_store_ps(dst + dex_dst, res_vec8);
            dex_src += 8;
            dex_dst += 8;
        }
        dex_dst += (Wdst << 3) - (_BLOCKED_CONV2_FP32_W_ << 3);
        dex_src += (Wsrc << 3) - (_BLOCKED_CONV2_FP32_W_ << 3);
    }
}



_THREAD_CALL_
void decx::conv::CPUK::_conv2_rN_rect_flex_fp32_ST(float* __restrict   src, 
                                       float* __restrict   kernel, 
                                       float* __restrict   dst,
                                       const uint2         proc_dim,
                                       const uint2         ker_dims,
                                       const uint          Wsrc,
                                       const uint          Wdst,
                                       const ushort        reg_WL,
                                       const uint          _loop)
{
    __m256 res_vec8;
    size_t dex_src = 0, dex_dst = 0;

    for (int i = 0; i < proc_dim.y; ++i) {
        for (int j = 0; j < proc_dim.x; ++j) {
            res_vec8 = decx::conv::CPUK::_conv2_fp32_loop_in_kernel_Nregs(src + dex_src, kernel, ker_dims, reg_WL, Wsrc, _loop);
            _mm256_store_ps(dst + dex_dst, res_vec8);
            dex_src += 8;
            dex_dst += 8;
        }
        dex_dst += (Wdst << 3) - (proc_dim.x << 3);
        dex_src += (Wsrc << 3) - (proc_dim.x << 3);
    }
}



_THREAD_CALL_
void decx::conv::CPUK::_conv2_r1_r4_rect_fixed_fp32_ST(float* __restrict   src, 
                                           float* __restrict   kernel, 
                                           float* __restrict   dst,
                                           const uint2         ker_dims,
                                           const uint          Wsrc,
                                           const uint          Wdst)
{
    __m256 res_vec8;
    size_t dex_src = 0, dex_dst = 0;

    for (int i = 0; i < _BLOCKED_CONV2_FP32_H_; ++i) {
#pragma unroll _BLOCKED_CONV2_FP32_W_
        for (int j = 0; j < _BLOCKED_CONV2_FP32_W_; ++j) {
            res_vec8 = decx::conv::CPUK::_conv2_fp32_loop_in_kernel_2regs(src + dex_src, kernel, ker_dims, Wsrc);
            _mm256_store_ps(dst + dex_dst, res_vec8);
            dex_src += 8;
            dex_dst += 8;
        }
        dex_dst += (Wdst << 3) - (_BLOCKED_CONV2_FP32_W_ << 3);
        dex_src += (Wsrc << 3) - (_BLOCKED_CONV2_FP32_W_ << 3);
    }
}



_THREAD_CALL_
void decx::conv::CPUK::_conv2_r1_r4_rect_flex_fp32_ST(float* __restrict   src, 
                                          float* __restrict   kernel, 
                                          float* __restrict   dst,
                                          const uint2         proc_dim,
                                          const uint2         ker_dims,
                                          const uint          Wsrc,
                                          const uint          Wdst)
{
    __m256 res_vec8;
    size_t dex_src = 0, dex_dst = 0;

    for (int i = 0; i < proc_dim.y; ++i) {
        for (int j = 0; j < proc_dim.x; ++j) {
            res_vec8 = decx::conv::CPUK::_conv2_fp32_loop_in_kernel_2regs(src + dex_src, kernel, ker_dims, Wsrc);
            _mm256_store_ps(dst + dex_dst, res_vec8);
            dex_src += 8;
            dex_dst += 8;
        }
        dex_dst += (Wdst << 3) - (proc_dim.x << 3);
        dex_src += (Wsrc << 3) - (proc_dim.x << 3);
    }
}



namespace decx
{
    /*
    * This function is suitable only when 3 __m256 are needed to cover all the data on one row
    * which indicates that half_kerdim.x -> [5, 8]
    * 
    * @param proc_dim : .x -> in _m256, the width of proccess area of signle thread (on dst matrix)
    *                   .y -> the height of proccess area of signle thread (on dst matrix)
    * @param ker_dim : .x -> the width of kernel (in float); .y -> the height of kernel
    * @param reg_WL : the leftover on width. ( = (half_ker_dims.x * 2 + 8) - 2 * 8)
    * @param Wsrc : the pitch of src matrix, in __m256
    */
    _THREAD_FUNCTION_
    void _conv2_r1_r4_fp32_ST(float* src, float* kernel, float* dst, 
                              const uint2 proc_dim, const uint2 ker_dims, const uint Wsrc, const uint Wdst);


    /*
    * This function is suitable only when 3 __m256 are needed to cover all the data on one row
    * which indicates that half_kerdim.x -> [5, 8]
    * 
    * @param proc_dim : .x -> in _m256, the width of proccess area of signle thread (on dst matrix)
    *                   .y -> the height of proccess area of signle thread (on dst matrix)
    * @param ker_dim : .x -> the width of kernel (in float); .y -> the height of kernel
    * @param reg_WL : the leftover on width. ( = (half_ker_dims.x * 2 + 8) - 2 * 8)
    * @param Wsrc : the pitch of src matrix, in __m256
    */
    _THREAD_FUNCTION_
    void _conv2_rN_fp32_ST(float* src, float* kernel, float* dst, const uint2 proc_dim, const uint2 ker_dims, 
        const uint Wsrc, const uint Wdst, const ushort reg_WL, const uint _loop);
}



namespace decx
{
    namespace conv{
        void _conv2_rN_fp32_caller(float* src, float* kernel, float* dst, const uint2 proc_dim, const uint2 ker_dims,
            const uint Wsrc, const uint Wdst, const ushort reg_WL,
            decx::utils::_thr_1D* t1D, decx::utils::frag_manager* f_mgr,
            const uint _loop);




        void _conv2_r1_r4_fp32_caller(float* src, float* kernel, float* dst, const uint2 proc_dim, const uint2 ker_dims,
                                             const uint Wsrc, const uint Wdst,
                                             decx::utils::_thr_1D* t1D, decx::utils::frag_manager* f_mgr);
    }
}



namespace decx
{
    namespace conv{
        static void _conv2_r8_fp32_organiser(float* src, float* kernel, float* dst, 
                                  const uint2 proc_dim, const uint2 ker_dims, const uint Wsrc,
                                  const uint Wdst, decx::utils::_thr_1D* t1D, decx::utils::frag_manager* f_mgr);
    }
}



void decx::conv::_conv2_r8_fp32_organiser(float*                       src,
                                    float*                       kernel, 
                                    float*                             dst, 
                                    const uint2                        proc_dim, 
                                    const uint2                        ker_dims,
                                    const uint                         Wsrc,
                                    const uint                         Wdst,
                                    decx::utils::_thr_1D*              t1D,
                                    decx::utils::frag_manager*         f_mgr)
{
    const uint half_kernel_w = ker_dims.x / 2;
    if (half_kernel_w < 5) {
        decx::conv::_conv2_r1_r4_fp32_caller(src, kernel, dst, proc_dim, ker_dims, Wsrc, Wdst, t1D, f_mgr);
    }
    else {
        const uint _loop = ((ker_dims.x / 2) - 5) / 4;
        ushort reg_WL = (ushort)(ker_dims.x - 1 - 8 - _loop * 8);
        decx::conv::_conv2_rN_fp32_caller(src, kernel, dst, proc_dim, ker_dims, Wsrc, Wdst, reg_WL, t1D, f_mgr, _loop);
    }
}


#endif