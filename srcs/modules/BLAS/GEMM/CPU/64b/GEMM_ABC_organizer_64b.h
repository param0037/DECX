/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.8
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.8, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _GEMM_ABC_ORGANIZER_FP64_H_
#define _GEMM_ABC_ORGANIZER_FP64_H_


#include "GEMM_ABC_kernel_64b.h"


namespace decx
{
    template <bool _is_cpl>
    /**
    * @param pitch_A : The pitch of matrix A (in __m256)
    * @param pitch_B : The pitch of cache B, the same as that of matrix dst (in double)
    * @param proc_hA : The height of matrix A
    * @param proc_wB : The width of matrix B, in __m256 * 2
    * @param __linear : _A->width == _B->height, in __m256
    */
    static void GEMM_ABC_fp64_caller_16x(double* A, double* B, double* C, double* dst, const uint pitchA, const uint pitchB,
        const uint pitchC, const uint pitchdst, const uint2 proc_dims, const uint linear_len, decx::utils::_thr_2D *t2D);


    template <bool _is_cpl>
    /**
    * @param pitch_A : The pitch of matrix A (in __m256)
    * @param pitch_B : The pitch of cache B, the same as that of matrix dst (in double)
    * @param proc_hA : The height of matrix A
    * @param proc_wB : The width of matrix B, in __m256 * 2
    * @param __linear : _A->width == _B->height, in __m256
    */
    static void GEMM_ABC_fp64_caller_8x(double* A, double* B, double* C, double* dst, double* extra_dst, const uint pitchA, const uint pitchB,
        const uint pitchC, const uint pitchdst, const uint pitch_ext_dst, const uint2 proc_dims, const uint linear_len, decx::utils::_thr_2D *t2D);
}


template <bool _is_cpl>
static void decx::GEMM_ABC_fp64_caller_16x(double* A,                         double* B, 
                                           double* C,                         double* dst,
                                           const uint pitchA,                const uint pitchB,
                                           const uint pitchdst,              const uint pitchC,
                                           const uint2 proc_dims,            
                                           const uint linear_len,            decx::utils::_thr_2D* t2D)
{
    decx::utils::frag_manager f_mgr_H, f_mgr_W;
    uint _L_thr_dex = 0;

    decx::utils::frag_manager_gen(&f_mgr_H, proc_dims.y, t2D->thread_h);
    decx::utils::frag_manager_gen(&f_mgr_W, proc_dims.x, t2D->thread_w);
    uint proc_w = f_mgr_W.is_left ? f_mgr_W.frag_left_over : f_mgr_W.frag_len;
    double* A_local_ptr = NULL, * B_local_ptr = NULL, *C_local_ptr = NULL, * dst_local_ptr = NULL;

    if (f_mgr_H.is_left || f_mgr_W.is_left) {
        for (int i = 0; i < t2D->thread_h - 1; ++i) {
            for (int j = 0; j < t2D->thread_w - 1; ++j) 
            {
                A_local_ptr     = DECX_PTR_SHF_XY<double, double>(A, i * f_mgr_H.frag_len, 0, pitchA);
                B_local_ptr     = DECX_PTR_SHF_XY<double, double>(B, j * f_mgr_W.frag_len, 0, pitchB);
                dst_local_ptr   = DECX_PTR_SHF_XY<double, double>(dst, i * f_mgr_H.frag_len, j * f_mgr_W.frag_len * 8, pitchdst);
                C_local_ptr     = DECX_PTR_SHF_XY<double, double>(C, i * f_mgr_H.frag_len, j * f_mgr_W.frag_len * 8, pitchdst);

                t2D->_async_thread[_L_thr_dex] = decx::cpu::register_task_default( decx::gemm::CPUK::GEMM_ABC_fp64_flexWH<_is_cpl>,
                    A_local_ptr, B_local_ptr, C_local_ptr, dst_local_ptr,
                    pitchA, pitchB, pitchC, pitchdst, linear_len, 
                    make_uint2(f_mgr_W.frag_len, f_mgr_H.frag_len));

                ++_L_thr_dex;
            }
            A_local_ptr     = DECX_PTR_SHF_XY<double, double>(A, i * f_mgr_H.frag_len, 0, pitchA);
            B_local_ptr     = DECX_PTR_SHF_XY<double, double>(B, (t2D->thread_w - 1) * f_mgr_W.frag_len, 0, pitchB);
            dst_local_ptr   = DECX_PTR_SHF_XY<double, double>(dst, i * f_mgr_H.frag_len, (t2D->thread_w - 1) * f_mgr_W.frag_len * 8, pitchdst);
            C_local_ptr     = DECX_PTR_SHF_XY<double, double>(C, i * f_mgr_H.frag_len, (t2D->thread_w - 1) * f_mgr_W.frag_len * 8, pitchdst);

            t2D->_async_thread[_L_thr_dex] = decx::cpu::register_task_default( decx::gemm::CPUK::GEMM_ABC_fp64_flexWH<_is_cpl>,
                A_local_ptr, B_local_ptr, C_local_ptr, dst_local_ptr,
                pitchA, pitchB, pitchC, pitchdst, linear_len,
                make_uint2(proc_w, f_mgr_H.frag_len));

            ++_L_thr_dex;
        }
        uint proc_h = f_mgr_H.is_left ? f_mgr_H.frag_left_over : f_mgr_H.frag_len;
        for (int j = 0; j < t2D->thread_w - 1; ++j)
        {
            A_local_ptr = DECX_PTR_SHF_XY<double, double>(A, (t2D->thread_h - 1) * f_mgr_H.frag_len, 0, pitchA);
            B_local_ptr = DECX_PTR_SHF_XY<double, double>(B, j * f_mgr_W.frag_len, 0, pitchB);
            dst_local_ptr = DECX_PTR_SHF_XY<double, double>(dst, (t2D->thread_h - 1) * f_mgr_H.frag_len, j * f_mgr_W.frag_len * 8, pitchdst);
            C_local_ptr = DECX_PTR_SHF_XY<double, double>(C, (t2D->thread_h - 1) * f_mgr_H.frag_len, j * f_mgr_W.frag_len * 8, pitchdst);

            t2D->_async_thread[_L_thr_dex] = decx::cpu::register_task_default( decx::gemm::CPUK::GEMM_ABC_fp64_flexWH<_is_cpl>,
                A_local_ptr, B_local_ptr, C_local_ptr, dst_local_ptr,
                pitchA, pitchB, pitchC, pitchdst, linear_len,
                make_uint2(f_mgr_W.frag_len, proc_h));

            ++_L_thr_dex;
        }
        A_local_ptr = DECX_PTR_SHF_XY<double, double>(A, (t2D->thread_h - 1) * f_mgr_H.frag_len, 0, pitchA);
        B_local_ptr = DECX_PTR_SHF_XY<double, double>(B, (t2D->thread_w - 1) * f_mgr_W.frag_len, 0, pitchB);
        dst_local_ptr = DECX_PTR_SHF_XY<double, double>(dst, (t2D->thread_h - 1) * f_mgr_H.frag_len, (t2D->thread_w - 1) * f_mgr_W.frag_len * 8, pitchdst);
        C_local_ptr = DECX_PTR_SHF_XY<double, double>(C, (t2D->thread_h - 1) * f_mgr_H.frag_len, (t2D->thread_w - 1) * f_mgr_W.frag_len * 8, pitchdst);

        t2D->_async_thread[_L_thr_dex] = decx::cpu::register_task_default( decx::gemm::CPUK::GEMM_ABC_fp64_flexWH<_is_cpl>,
            A_local_ptr, B_local_ptr, C_local_ptr, dst_local_ptr,
            pitchA, pitchB, pitchC, pitchdst, linear_len,
            make_uint2(proc_w, proc_h));
    }
    else {
        for (int i = 0; i < t2D->thread_h; ++i) {
            for (int j = 0; j < t2D->thread_w; ++j) 
            {
                A_local_ptr     = DECX_PTR_SHF_XY<double, double>(A, i * f_mgr_H.frag_len, 0, pitchA);
                B_local_ptr     = DECX_PTR_SHF_XY<double, double>(B, j * f_mgr_W.frag_len, 0, pitchB);
                dst_local_ptr   = DECX_PTR_SHF_XY<double, double>(dst, i * f_mgr_H.frag_len, j * f_mgr_W.frag_len * 8, pitchdst);
                C_local_ptr     = DECX_PTR_SHF_XY<double, double>(C, i * f_mgr_H.frag_len, j * f_mgr_W.frag_len * 8, pitchdst);

                t2D->_async_thread[_L_thr_dex] = decx::cpu::register_task_default( decx::gemm::CPUK::GEMM_ABC_fp64_AllFixed<_is_cpl>,
                    A_local_ptr, B_local_ptr, C_local_ptr, dst_local_ptr,
                    pitchA, pitchB, pitchC, pitchdst, linear_len,
                    make_uint2(f_mgr_W.frag_len, f_mgr_H.frag_len));
                    
                ++_L_thr_dex;
            }
        }
    }

    t2D->__sync_all_threads();
}



template <bool _is_cpl>
static void decx::GEMM_ABC_fp64_caller_8x(double* A,                         double* B, 
                                          double* C,                         double* dst,
                                          double* extra_dst,                 const uint pitchA,                
                                          const uint pitchB,                const uint pitchC,
                                          const uint pitchdst,              const uint pitch_ext_dst,
                                          const uint2 proc_dims,          
                                          const uint linear_len,            decx::utils::_thr_2D* t2D)
{
    decx::utils::frag_manager f_mgr_H, f_mgr_W;
    uint _L_thr_dex = 0;
    decx::utils::frag_manager_gen(&f_mgr_H, proc_dims.y, t2D->thread_h);
    decx::utils::frag_manager_gen(&f_mgr_W, proc_dims.x, t2D->thread_w);
    uint proc_w = f_mgr_W.is_left ? f_mgr_W.frag_left_over : f_mgr_W.frag_len;

    double* A_local_ptr = NULL, * B_local_ptr = NULL, *C_local_ptr = NULL, * dst_local_ptr = NULL;

    for (int i = 0; i < t2D->thread_h - 1; ++i) {
        for (int j = 0; j < t2D->thread_w - 1; ++j)
        {
            A_local_ptr = DECX_PTR_SHF_XY<double, double>(A, i * f_mgr_H.frag_len, 0, pitchA);
            B_local_ptr = DECX_PTR_SHF_XY<double, double>(B, j * f_mgr_W.frag_len, 0, pitchB);
            dst_local_ptr = DECX_PTR_SHF_XY<double, double>(dst, i * f_mgr_H.frag_len, j * f_mgr_W.frag_len * 8, pitchdst);
            C_local_ptr = DECX_PTR_SHF_XY<double, double>(C, i * f_mgr_H.frag_len, j * f_mgr_W.frag_len * 8, pitchdst);

            t2D->_async_thread[_L_thr_dex] = decx::cpu::register_task_default( decx::gemm::CPUK::GEMM_ABC_fp64_flexWH<_is_cpl>,
                A_local_ptr, B_local_ptr, C_local_ptr, dst_local_ptr,
                pitchA, pitchB, pitchC, pitchdst, linear_len,
                make_uint2(f_mgr_W.frag_len, f_mgr_H.frag_len));

            ++_L_thr_dex;
        }

        A_local_ptr = DECX_PTR_SHF_XY<double, double>(A, i * f_mgr_H.frag_len, 0, pitchA);
        B_local_ptr = DECX_PTR_SHF_XY<double, double>(B, (t2D->thread_w - 1) * f_mgr_W.frag_len, 0, pitchB);
        dst_local_ptr = DECX_PTR_SHF_XY<double, double>(extra_dst, i * f_mgr_H.frag_len, 0, pitch_ext_dst);
        C_local_ptr = DECX_PTR_SHF_XY<double, double>(C, i * f_mgr_H.frag_len, (t2D->thread_w - 1) * f_mgr_W.frag_len * 8, pitchdst);

        t2D->_async_thread[_L_thr_dex] = decx::cpu::register_task_default( decx::gemm::CPUK::GEMM_ABC_fp64_flexWH<_is_cpl>,
            A_local_ptr, B_local_ptr, C_local_ptr, dst_local_ptr,
            pitchA, pitchB, pitchC, pitch_ext_dst, linear_len,
            make_uint2(proc_w, f_mgr_H.frag_len));

        ++_L_thr_dex;
    }
    uint proc_h = f_mgr_H.is_left ? f_mgr_H.frag_left_over : f_mgr_H.frag_len;
    for (int j = 0; j < t2D->thread_w - 1; ++j)
    {
        A_local_ptr = DECX_PTR_SHF_XY<double, double>(A, (t2D->thread_h - 1) * f_mgr_H.frag_len, 0, pitchA);
        B_local_ptr = DECX_PTR_SHF_XY<double, double>(B, j * f_mgr_W.frag_len, 0, pitchB);
        dst_local_ptr = DECX_PTR_SHF_XY<double, double>(dst, (t2D->thread_h - 1) * f_mgr_H.frag_len, j * f_mgr_W.frag_len * 8, pitchdst);
        C_local_ptr = DECX_PTR_SHF_XY<double, double>(C, (t2D->thread_h - 1) * f_mgr_H.frag_len, j * f_mgr_W.frag_len * 8, pitchdst);

        t2D->_async_thread[_L_thr_dex] = decx::cpu::register_task_default( decx::gemm::CPUK::GEMM_ABC_fp64_flexWH<_is_cpl>,
            A_local_ptr, B_local_ptr, C_local_ptr, dst_local_ptr,
            pitchA, pitchB, pitchC, pitchdst, linear_len,
            make_uint2(f_mgr_W.frag_len, proc_h));

        ++_L_thr_dex;
    }

    A_local_ptr = DECX_PTR_SHF_XY<double, double>(A, (t2D->thread_h - 1) * f_mgr_H.frag_len, 0, pitchA);
    B_local_ptr = DECX_PTR_SHF_XY<double, double>(B, (t2D->thread_w - 1) * f_mgr_W.frag_len, 0, pitchB);
    dst_local_ptr = DECX_PTR_SHF_XY<double, double>(extra_dst, (t2D->thread_h - 1) * f_mgr_H.frag_len, 0, pitch_ext_dst);
    C_local_ptr = DECX_PTR_SHF_XY<double, double>(C, (t2D->thread_h - 1) * f_mgr_H.frag_len, (t2D->thread_w - 1) * f_mgr_W.frag_len * 8, pitch_ext_dst);

    t2D->_async_thread[_L_thr_dex] = decx::cpu::register_task_default( decx::gemm::CPUK::GEMM_ABC_fp64_flexWH<_is_cpl>,
        A_local_ptr, B_local_ptr, C_local_ptr, dst_local_ptr,
        pitchA, pitchB, pitchC, pitch_ext_dst, linear_len,
        make_uint2(proc_w, proc_h));

    t2D->__sync_all_threads();
}



#endif