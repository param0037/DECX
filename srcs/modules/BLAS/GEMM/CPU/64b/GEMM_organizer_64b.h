/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _GEMM_ORGANIZER_64B_H_
#define _GEMM_ORGANIZER_64B_H_


#include "GEMM_kernel_64b.h"


namespace decx
{
    template<bool _is_cpl>
    /**
    * @param pitch_A : The pitch of matrix A (in __m256)
    * @param pitch_B : The pitch of cache B, the same as that of matrix dst (in float)
    * @param proc_hA : The height of matrix A
    * @param proc_wB : The width of matrix B, in __m256 * 2
    * @param __linear : _A->width == _B->height, in __m256
    */
    static void GEMM_AB_fp64_caller_16x(const double* A, const double* B, double* dst, const uint32_t pitchA, const uint32_t pitchB,
        const uint32_t pitchdst, const uint2 proc_dims, const uint32_t linear_len, decx::utils::_thr_2D* t2D);

    template<bool _is_cpl>
    /**
    * @param pitch_A : The pitch of matrix A (in __m256)
    * @param pitch_B : The pitch of cache B, the same as that of matrix dst (in float)
    * @param proc_hA : The height of matrix A
    * @param proc_wB : The width of matrix B, in __m256 * 2
    * @param __linear : _A->width == _B->height, in __m256
    */
    static void GEMM_AB_fp64_caller_8x(const double* A, const double* B, double* dst, double* extra_dst, const uint32_t pitchA, const uint32_t pitchB,
        const uint32_t pitchdst, const uint32_t pitch_ext_dst, const uint2 proc_dims, const uint32_t linear_len, decx::utils::_thr_2D* t2D);
}


template<bool _is_cpl>
static void decx::GEMM_AB_fp64_caller_16x(const double* A,                  const double* B, 
                                          double* dst,                       
                                          const uint32_t pitchA,                const uint32_t pitchB,
                                          const uint32_t pitchdst,              const uint2 proc_dims,            
                                          const uint32_t linear_len,            decx::utils::_thr_2D* t2D)
{
    decx::utils::frag_manager f_mgr_H, f_mgr_W;
    uint32_t _L_thr_dex = 0;

    decx::utils::frag_manager_gen(&f_mgr_H, proc_dims.y, t2D->thread_h);
    decx::utils::frag_manager_gen(&f_mgr_W, proc_dims.x, t2D->thread_w);
    uint32_t proc_w = f_mgr_W.is_left ? f_mgr_W.frag_left_over : f_mgr_W.frag_len;
    const double* A_local_ptr = NULL, * B_local_ptr = NULL;
    double *dst_local_ptr = NULL;

    if (f_mgr_H.is_left || f_mgr_W.is_left) {
        for (int i = 0; i < t2D->thread_h - 1; ++i) {
            for (int j = 0; j < t2D->thread_w - 1; ++j) 
            {
                A_local_ptr = DECX_PTR_SHF_XY<const double, const double>(A, i * f_mgr_H.frag_len, 0, pitchA);
                B_local_ptr = DECX_PTR_SHF_XY<const double, const double>(B, j * f_mgr_W.frag_len, 0, pitchB);
                dst_local_ptr = DECX_PTR_SHF_XY<double, double>(dst, i * f_mgr_H.frag_len, j * f_mgr_W.frag_len * 8, pitchdst);

                t2D->_async_thread[_L_thr_dex] = decx::cpu::register_task_default( decx::gemm::CPUK::GEMM_AB_fp64_flexWH<_is_cpl>,
                    A_local_ptr, B_local_ptr, dst_local_ptr,
                    pitchA, pitchB, pitchdst, linear_len,
                    make_uint2(f_mgr_W.frag_len, f_mgr_H.frag_len));

                ++_L_thr_dex;
            }
            A_local_ptr = DECX_PTR_SHF_XY<const double, const double>(A, i * f_mgr_H.frag_len, 0, pitchA);
            B_local_ptr = DECX_PTR_SHF_XY<const double, const double>(B, (t2D->thread_w - 1) * f_mgr_W.frag_len, 0, pitchB);
            dst_local_ptr = DECX_PTR_SHF_XY<double, double>(dst, i * f_mgr_H.frag_len, (t2D->thread_w - 1) * f_mgr_W.frag_len * 8, pitchdst);

            t2D->_async_thread[_L_thr_dex] = decx::cpu::register_task_default( decx::gemm::CPUK::GEMM_AB_fp64_flexWH<_is_cpl>,
                A_local_ptr, B_local_ptr, dst_local_ptr,
                pitchA, pitchB, pitchdst, linear_len,
                make_uint2(proc_w, f_mgr_H.frag_len));

            ++_L_thr_dex;
        }
        uint32_t proc_h = f_mgr_H.is_left ? f_mgr_H.frag_left_over : f_mgr_H.frag_len;
        for (int j = 0; j < t2D->thread_w - 1; ++j)
        {
            A_local_ptr = DECX_PTR_SHF_XY<const double, const double>(A, (t2D->thread_h - 1) * f_mgr_H.frag_len, 0, pitchA);
            B_local_ptr = DECX_PTR_SHF_XY<const double, const double>(B, j * f_mgr_W.frag_len, 0, pitchB);
            dst_local_ptr = DECX_PTR_SHF_XY<double, double>(dst, (t2D->thread_h - 1) * f_mgr_H.frag_len, j * f_mgr_W.frag_len * 8, pitchdst);

            t2D->_async_thread[_L_thr_dex] = decx::cpu::register_task_default( decx::gemm::CPUK::GEMM_AB_fp64_flexWH<_is_cpl>,
                A_local_ptr, B_local_ptr, dst_local_ptr,
                pitchA, pitchB, pitchdst, linear_len,
                make_uint2(f_mgr_W.frag_len, proc_h));

            ++_L_thr_dex;
        }
        A_local_ptr = DECX_PTR_SHF_XY<const double, const double>(A, (t2D->thread_h - 1) * f_mgr_H.frag_len, 0, pitchA);
        B_local_ptr = DECX_PTR_SHF_XY<const double, const double>(B, (t2D->thread_w - 1) * f_mgr_W.frag_len, 0, pitchB);
        dst_local_ptr = DECX_PTR_SHF_XY<double, double>(dst, (t2D->thread_h - 1) * f_mgr_H.frag_len, (t2D->thread_w - 1) * f_mgr_W.frag_len * 8, pitchdst);

        t2D->_async_thread[_L_thr_dex] = decx::cpu::register_task_default( decx::gemm::CPUK::GEMM_AB_fp64_flexWH<_is_cpl>,
            A_local_ptr, B_local_ptr, dst_local_ptr,
            pitchA, pitchB, pitchdst, linear_len,
            make_uint2(proc_w, proc_h));
    }
    else {
        for (int i = 0; i < t2D->thread_h; ++i) {
            for (int j = 0; j < t2D->thread_w; ++j)
            {
                A_local_ptr = DECX_PTR_SHF_XY<const double, const double>(A, i * f_mgr_H.frag_len, 0, pitchA);
                B_local_ptr = DECX_PTR_SHF_XY<const double, const double>(B, j * f_mgr_W.frag_len, 0, pitchB);
                dst_local_ptr = DECX_PTR_SHF_XY<double, double>(dst, i * f_mgr_H.frag_len, j * f_mgr_W.frag_len * 8, pitchdst);

                t2D->_async_thread[_L_thr_dex] = decx::cpu::register_task_default( decx::gemm::CPUK::GEMM_AB_fp64_AllFixed<_is_cpl>,
                    A_local_ptr, B_local_ptr, dst_local_ptr, 
                    pitchA, pitchB, pitchdst, linear_len,
                    make_uint2(f_mgr_W.frag_len, f_mgr_H.frag_len));
                    
                ++_L_thr_dex;
            }
        }
    }

    t2D->__sync_all_threads();
}


template<bool _is_cpl>
static void decx::GEMM_AB_fp64_caller_8x(const double* A,                  const double* B, 
                                         double* dst,                      double* extra_dst,
                                         const uint32_t pitchA,                const uint32_t pitchB,
                                         const uint32_t pitchdst,              const uint32_t pitch_ext_dst,
                                         const uint2 proc_dims,          
                                         const uint32_t linear_len,            decx::utils::_thr_2D* t2D)
{
    decx::utils::frag_manager f_mgr_H, f_mgr_W;
    uint32_t _L_thr_dex = 0;
    decx::utils::frag_manager_gen(&f_mgr_H, proc_dims.y, t2D->thread_h);
    decx::utils::frag_manager_gen(&f_mgr_W, proc_dims.x, t2D->thread_w);
    uint32_t proc_w = f_mgr_W.is_left ? f_mgr_W.frag_left_over : f_mgr_W.frag_len;

    const double* A_local_ptr = NULL, * B_local_ptr = NULL;
    double *dst_local_ptr = NULL;

    for (int i = 0; i < t2D->thread_h - 1; ++i) {
        for (int j = 0; j < t2D->thread_w - 1; ++j)
        {
            A_local_ptr = DECX_PTR_SHF_XY<const double, const double>(A, i * f_mgr_H.frag_len, 0, pitchA);
            B_local_ptr = DECX_PTR_SHF_XY<const double, const double>(B, j * f_mgr_W.frag_len, 0, pitchB);
            dst_local_ptr = DECX_PTR_SHF_XY<double, double>(dst, i * f_mgr_H.frag_len, j * f_mgr_W.frag_len * 8, pitchdst);

            t2D->_async_thread[_L_thr_dex] = decx::cpu::register_task_default( decx::gemm::CPUK::GEMM_AB_fp64_flexWH<_is_cpl>,
                A_local_ptr, B_local_ptr, dst_local_ptr,
                pitchA, pitchB, pitchdst, linear_len,
                make_uint2(f_mgr_W.frag_len, f_mgr_H.frag_len));

            ++_L_thr_dex;
        }
        
        A_local_ptr = DECX_PTR_SHF_XY<const double, const double>(A, i * f_mgr_H.frag_len, 0, pitchA);
        B_local_ptr = DECX_PTR_SHF_XY<const double, const double>(B, (t2D->thread_w - 1) * f_mgr_W.frag_len, 0, pitchB);
        dst_local_ptr = DECX_PTR_SHF_XY<double, double>(extra_dst, i * f_mgr_H.frag_len, 0, pitch_ext_dst);

        t2D->_async_thread[_L_thr_dex] = decx::cpu::register_task_default( decx::gemm::CPUK::GEMM_AB_fp64_flexWH<_is_cpl>,
            A_local_ptr, B_local_ptr, dst_local_ptr,
            pitchA, pitchB, pitch_ext_dst, linear_len,
            make_uint2(proc_w, f_mgr_H.frag_len));

        ++_L_thr_dex;
    }
    uint32_t proc_h = f_mgr_H.is_left ? f_mgr_H.frag_left_over : f_mgr_H.frag_len;
    for (int j = 0; j < t2D->thread_w - 1; ++j)
    {
        A_local_ptr = DECX_PTR_SHF_XY<const double, const double>(A, (t2D->thread_h - 1) * f_mgr_H.frag_len, 0, pitchA);
        B_local_ptr = DECX_PTR_SHF_XY<const double, const double>(B, j * f_mgr_W.frag_len, 0, pitchB);
        dst_local_ptr = DECX_PTR_SHF_XY<double, double>(dst, (t2D->thread_h - 1) * f_mgr_H.frag_len, j * f_mgr_W.frag_len * 8, pitchdst);

        t2D->_async_thread[_L_thr_dex] = decx::cpu::register_task_default( decx::gemm::CPUK::GEMM_AB_fp64_flexWH<_is_cpl>,
            A_local_ptr, B_local_ptr, dst_local_ptr,
            pitchA, pitchB, pitchdst, linear_len,
            make_uint2(f_mgr_W.frag_len, proc_h));

        ++_L_thr_dex;
    }

    A_local_ptr = DECX_PTR_SHF_XY<const double, const double>(A, (t2D->thread_h - 1) * f_mgr_H.frag_len, 0, pitchA);
    B_local_ptr = DECX_PTR_SHF_XY<const double, const double>(B, (t2D->thread_w - 1) * f_mgr_W.frag_len, 0, pitchB);
    dst_local_ptr = DECX_PTR_SHF_XY<double, double>(extra_dst, (t2D->thread_h - 1) * f_mgr_H.frag_len, 0, pitch_ext_dst);

    t2D->_async_thread[_L_thr_dex] = decx::cpu::register_task_default( decx::gemm::CPUK::GEMM_AB_fp64_flexWH<_is_cpl>,
        A_local_ptr, B_local_ptr, dst_local_ptr,
        pitchA, pitchB, pitch_ext_dst, linear_len,
        make_uint2(proc_w, proc_h));

    t2D->__sync_all_threads();
}



#endif