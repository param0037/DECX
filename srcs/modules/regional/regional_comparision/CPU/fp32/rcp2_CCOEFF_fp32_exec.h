/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/


#ifndef _RCP2_CCOEFF_FP32_EXEC_H_
#define _RCP2_CCOEFF_FP32_EXEC_H_


#include "../rcp_CCOEFF_T_loop_core.h"
#include "../rcp2_sliding_window_macros.h"
#include "../../../../core/utils/fragment_arrangment.h"



namespace decx
{
    namespace rcp {
        namespace CPUK {
            static _THREAD_CALL_ void _rcp2_CCOEFF_rN_rect_fixed_fp32_ST(const float* src, float* kernel, const float* I_src, float* dst, 
                const uint2 ker_dims, const uint Wsrc, const uint W_Isrc, const uint Wdst, const ushort reg_WL, const uint _loop);



            static _THREAD_CALL_ void _rcp2_CCOEFF_rN_rect_flex_fp32_ST(const float* src, float* kernel, const float* I_src, float* dst, 
                const uint2 proc_dim, const uint2 ker_dims, const uint Wsrc, const uint W_Isrc, const uint Wdst, const ushort reg_WL, const uint _loop);



            static _THREAD_CALL_ void _rcp2_CCOEFF_r1_r4_rect_fixed_fp32_ST(const float* src, float* kernel, const float* I_src, float* dst, 
                const uint2 ker_dims, const uint Wsrc, const uint W_Isrc, const uint Wdst);



            static _THREAD_CALL_ void _rcp2_CCOEFF_r1_r4_rect_flex_fp32_ST(const float* src, float* kernel, const float* I_src, float* dst, 
                const uint2 proc_dim, const uint2 ker_dims, const uint Wsrc, const uint W_Isrc, const uint Wdst);

            // ------------------------------------------------------- norm -------------------------------------------------------

            /*
            * @param _sqrt_k_sum : SUM<i, j>(kernel(i, j) ^ 2)
            */
            static _THREAD_CALL_ void _rcp2_CCOEFF_norm_rN_rect_fixed_fp32_ST(const float* src, float* kernel, const float* I_src, float* dst, 
                const float _sqrt_k_sum, const uint2 ker_dims, const uint Wsrc, const uint W_Isrc, const uint Wdst, const ushort reg_WL, const uint _loop);



            static _THREAD_CALL_ void _rcp2_CCOEFF_norm_rN_rect_flex_fp32_ST(const float* src, float* kernel, const float* I_src, float* dst, 
                const float _sqrt_k_sum, const uint2 proc_dim, const uint2 ker_dims, const uint Wsrc, const uint W_Isrc, const uint Wdst, const ushort reg_WL, const uint _loop);



            static _THREAD_CALL_ void _rcp2_CCOEFF_norm_r1_r4_rect_fixed_fp32_ST(const float* src, float* kernel, const float* I_src, float* dst, 
                const float _sqrt_k_sum, const uint2 ker_dims, const uint Wsrc, const uint W_Isrc, const uint Wdst);



            static _THREAD_CALL_ void _rcp2_CCOEFF_norm_r1_r4_rect_flex_fp32_ST(const float* src, float* kernel, const float* I_src, float* dst, 
                const float _sqrt_k_sum, const uint2 proc_dim, const uint2 ker_dims, const uint Wsrc, const uint W_Isrc, const uint Wdst);
        }
    }
}



_THREAD_CALL_
void decx::rcp::CPUK::_rcp2_CCOEFF_rN_rect_fixed_fp32_ST(const float* __restrict   src, 
                                        float* __restrict   kernel, 
                                        const float* __restrict   I_src,
                                        float* __restrict   dst,
                                        const uint2         ker_dims,
                                        const uint          Wsrc,
                                        const uint          W_Isrc,
                                        const uint          Wdst,
                                        const ushort        reg_WL,
                                        const uint          _loop)
{
    _rcp2_fixed_fp32_CCOEFF(decx::rcp::CPUK::_rcp_CCOEFF_fp32_loop_in_kernel_Nregs(
        src + dex_src, kernel, I_src + dex_Isrc, ker_dims, reg_WL, Wsrc, W_Isrc, _loop));
}



_THREAD_CALL_
void decx::rcp::CPUK::_rcp2_CCOEFF_rN_rect_flex_fp32_ST(const float* __restrict   src, 
                                        float* __restrict   kernel, 
                                        const float* __restrict   I_src,
                                        float* __restrict   dst,
                                       const uint2         proc_dim,
                                       const uint2         ker_dims,
                                       const uint          Wsrc,
                                       const uint          W_Isrc,
                                       const uint          Wdst,
                                       const ushort        reg_WL,
                                       const uint          _loop)
{
    _rcp2_flex_fp32_CCOEFF(decx::rcp::CPUK::_rcp_CCOEFF_fp32_loop_in_kernel_Nregs(
        src + dex_src, kernel, I_src + dex_Isrc, ker_dims, reg_WL, Wsrc, W_Isrc, _loop));
}



_THREAD_CALL_
void decx::rcp::CPUK::_rcp2_CCOEFF_r1_r4_rect_fixed_fp32_ST(const float* __restrict   src, 
                                                            float* __restrict   kernel, 
                                                            const float* __restrict   I_src,
                                                            float* __restrict   dst,
                                                            const uint2         ker_dims,
                                                            const uint          Wsrc,
                                                            const uint          W_Isrc,
                                                            const uint          Wdst)
{
    _rcp2_fixed_fp32_CCOEFF(decx::rcp::CPUK::_rcp_CCOEFF_fp32_loop_in_kernel_2regs(
        src + dex_src, kernel, I_src + dex_Isrc, ker_dims, Wsrc, W_Isrc));
}



_THREAD_CALL_
void decx::rcp::CPUK::_rcp2_CCOEFF_r1_r4_rect_flex_fp32_ST(const float* __restrict   src, 
                                                           float* __restrict   kernel, 
                                                           const float* __restrict   I_src,
                                                           float* __restrict   dst,
                                                           const uint2         proc_dim,
                                                           const uint2         ker_dims,
                                                           const uint          Wsrc,
                                                           const uint          W_Isrc,
                                                           const uint          Wdst)
{
    _rcp2_flex_fp32_CCOEFF(decx::rcp::CPUK::_rcp_CCOEFF_fp32_loop_in_kernel_2regs(
        src + dex_src, kernel, I_src + dex_Isrc, ker_dims, Wsrc, W_Isrc));
}




// -------------------------------------------- norm --------------------------------------------------



_THREAD_CALL_
void decx::rcp::CPUK::_rcp2_CCOEFF_norm_rN_rect_fixed_fp32_ST(const float* __restrict   src, 
                                        float* __restrict   kernel, 
                                        const float* __restrict   I_src,
                                        float* __restrict   dst,
                                        const float         _sqrt_k_sum,
                                        const uint2         ker_dims,
                                        const uint          Wsrc,
                                        const uint          W_Isrc,
                                        const uint          Wdst,
                                        const ushort        reg_WL,
                                        const uint          _loop)
{
    _rcp2_fixed_fp32_CCOEFF(decx::rcp::CPUK::_rcp_CCOEFF_NORM_fp32_loop_in_kernel_Nregs(
        src + dex_src, kernel, I_src + dex_Isrc, _sqrt_k_sum, ker_dims, reg_WL, Wsrc, W_Isrc, _loop));
}



_THREAD_CALL_
void decx::rcp::CPUK::_rcp2_CCOEFF_norm_rN_rect_flex_fp32_ST(const float* __restrict   src, 
                                        float* __restrict   kernel, 
                                        const float* __restrict   I_src,
                                        float* __restrict   dst,
                                       const float        _sqrt_k_sum,
                                       const uint2         proc_dim,
                                       const uint2         ker_dims,
                                       const uint          Wsrc,
                                       const uint          W_Isrc,
                                       const uint          Wdst,
                                       const ushort        reg_WL,
                                       const uint          _loop)
{
    _rcp2_flex_fp32_CCOEFF(decx::rcp::CPUK::_rcp_CCOEFF_NORM_fp32_loop_in_kernel_Nregs(
        src + dex_src, kernel, I_src + dex_Isrc, _sqrt_k_sum, ker_dims, reg_WL, Wsrc, W_Isrc, _loop));
}



_THREAD_CALL_
void decx::rcp::CPUK::_rcp2_CCOEFF_norm_r1_r4_rect_fixed_fp32_ST(const float* __restrict   src, 
                                        float* __restrict   kernel, 
                                        const float* __restrict   I_src,
                                        float* __restrict   dst,
                                           const float         _sqrt_k_sum,
                                           const uint2         ker_dims,
                                           const uint          Wsrc,
                                           const uint          W_Isrc,
                                           const uint          Wdst)
{
    _rcp2_fixed_fp32_CCOEFF(decx::rcp::CPUK::_rcp_CCOEFF_NORM_fp32_loop_in_kernel_2regs(
        src + dex_src, kernel, I_src + dex_Isrc, _sqrt_k_sum, ker_dims, Wsrc, W_Isrc));
}



_THREAD_CALL_
void decx::rcp::CPUK::_rcp2_CCOEFF_norm_r1_r4_rect_flex_fp32_ST(const float* __restrict   src, 
                                        float* __restrict   kernel, 
                                        const float* __restrict   I_src,
                                        float* __restrict   dst,
                                          const float         _sqrt_k_sum,
                                          const uint2         proc_dim,
                                          const uint2         ker_dims,
                                          const uint          Wsrc,
                                          const uint          W_Isrc,
                                          const uint          Wdst)
{
    _rcp2_flex_fp32_CCOEFF(decx::rcp::CPUK::_rcp_CCOEFF_NORM_fp32_loop_in_kernel_2regs(
        src + dex_src, kernel, I_src + dex_Isrc, _sqrt_k_sum, ker_dims, Wsrc, W_Isrc));
}



namespace decx
{
    namespace rcp {
        namespace CPUK {
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
            void _rcp2_CCOEFF_r1_r4_fp32_ST(float* src, float* kernel, float* I_src, float* dst,
                const uint2 proc_dim, const uint2 ker_dims, const uint Wsrc, const uint W_Isrc, const uint Wdst);


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
            void _rcp2_CCOEFF_rN_fp32_ST(float* src, float* kernel, float* I_src, float* dst, const uint2 proc_dim, const uint2 ker_dims,
                const uint Wsrc, const uint W_Isrc, const uint Wdst, const ushort reg_WL, const uint _loop);


            _THREAD_FUNCTION_
            void _rcp2_CCOEFF_norm_r1_r4_fp32_ST(float* src, float* kernel, float* I_src, float* dst,const float _sqrt_k_sum, 
                const uint2 proc_dim, const uint2 ker_dims, const uint Wsrc, const uint W_Isrc, const uint Wdst);


            _THREAD_FUNCTION_
            void _rcp2_CCOEFF_norm_rN_fp32_ST(float* src, float* kernel, float* I_src, float* dst, const float _sqrt_k_sum, const uint2 proc_dim, const uint2 ker_dims,
                const uint Wsrc, const uint W_Isrc, const uint Wdst, const ushort reg_WL, const uint _loop);
        }
    }
}



namespace decx
{
    namespace rcp{
        void _rcp2_CCOEFF_rN_fp32_caller(float* src, float* kernel, float* I_src, float* dst, const uint2 proc_dim, const uint2 ker_dims,
            const uint Wsrc, const uint W_Isrc, const uint Wdst, const ushort reg_WL,
            decx::utils::_thr_1D* t1D, decx::utils::frag_manager* f_mgr,
            const uint _loop);




        void _rcp2_CCOEFF_r1_r4_fp32_caller(float* src, float* kernel, float* I_src, float* dst, const uint2 proc_dim, const uint2 ker_dims,
                                             const uint Wsrc, const uint W_Isrc, const uint Wdst,
                                             decx::utils::_thr_1D* t1D, decx::utils::frag_manager* f_mgr);


        void _rcp2_CCOEFF_NORM_rN_fp32_caller(float* src, float* kernel, float* I_src, float* dst, const float _sqrt_k_sum, const uint2 proc_dim, const uint2 ker_dims,
            const uint Wsrc, const uint W_Isrc, const uint Wdst, const ushort reg_WL,
            decx::utils::_thr_1D* t1D, decx::utils::frag_manager* f_mgr,
            const uint _loop);




        void _rcp2_CCOEFF_NORM_r1_r4_fp32_caller(float* src, float* kernel, float* I_src, float* dst, const float _sqrt_k_sum, const uint2 proc_dim, const uint2 ker_dims,
                                             const uint Wsrc, const uint W_Isrc, const uint Wdst,
                                             decx::utils::_thr_1D* t1D, decx::utils::frag_manager* f_mgr);
    }
}



namespace decx
{
    namespace rcp{
        template <bool _norm>
        static void _rcp2_CCOEFF_fp32_organiser(float* src, float* kernel, float* I_src, float* dst, const float _sqrt_k_sum,
                                  const uint2 proc_dim, const uint2 ker_dims, const uint Wsrc, const uint W_I_src,
                                  const uint Wdst, decx::utils::_thr_1D* t1D, decx::utils::frag_manager* f_mgr);
    }
}


template <bool _norm>
void decx::rcp::_rcp2_CCOEFF_fp32_organiser(float*                       src,
                                    float*                             kernel, 
                                    float*                             I_src, 
                                    float*                             dst, 
                                    const float                        _sqrt_k_sum,
                                    const uint2                        proc_dim, 
                                    const uint2                        ker_dims,
                                    const uint                         Wsrc,
                                    const uint                         W_Isrc,
                                    const uint                         Wdst,
                                    decx::utils::_thr_1D*              t1D,
                                    decx::utils::frag_manager*         f_mgr)
{
    const uint half_kernel_w = ker_dims.x / 2;

    if (_norm) {
        if (half_kernel_w < 5) {
            decx::rcp::_rcp2_CCOEFF_NORM_r1_r4_fp32_caller(src, kernel, I_src, dst, 
                _sqrt_k_sum, proc_dim, ker_dims, Wsrc, W_Isrc, Wdst, t1D, f_mgr);
        }
        else {
            const uint _loop = ((ker_dims.x / 2) - 5) / 4;
            ushort reg_WL = (ushort)(ker_dims.x - 1 - 8 - _loop * 8);
            decx::rcp::_rcp2_CCOEFF_NORM_rN_fp32_caller(src, kernel, I_src, dst, 
                _sqrt_k_sum, proc_dim, ker_dims, Wsrc, W_Isrc, Wdst, reg_WL, t1D, f_mgr, _loop);
        }
    }
    else {
        if (half_kernel_w < 5) {
            decx::rcp::_rcp2_CCOEFF_r1_r4_fp32_caller(src, kernel, I_src, dst, 
                proc_dim, ker_dims, Wsrc, W_Isrc, Wdst, t1D, f_mgr);
        }
        else {
            const uint _loop = ((ker_dims.x / 2) - 5) / 4;
            ushort reg_WL = (ushort)(ker_dims.x - 1 - 8 - _loop * 8);
            decx::rcp::_rcp2_CCOEFF_rN_fp32_caller(src, kernel, I_src, dst, 
                proc_dim, ker_dims, Wsrc, W_Isrc, Wdst, reg_WL, t1D, f_mgr, _loop);
        }
    }
}


#endif