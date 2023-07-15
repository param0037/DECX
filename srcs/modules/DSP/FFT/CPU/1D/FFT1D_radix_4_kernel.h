/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _RADIX_4_KERNEL_H_
#define _RADIX_4_KERNEL_H_

#include "../../../../core/basic.h"
#include "../../../../core/thread_management/thread_pool.h"
#include "../../../../core/utils/fragment_arrangment.h"
#include "../../../../classes/classes_util.h"
#include "../../fft_utils.h"
#include "../../../CPU_cpf32_avx.h"
#include "../../../../core/thread_management/thread_arrange.h"


namespace decx
{
    namespace signal
    {
        namespace CPUK
        {
        _THREAD_FUNCTION_
        void _FFT1D_R4_fp32_R2C_first_ST(const float* __restrict src, double* __restrict dst, const size_t signal_length,
            const uint2 b_op_dex_range);


        _THREAD_FUNCTION_
        void _IFFT1D_R4_fp32_C2C_first_ST(const double* __restrict src, double* __restrict dst, const size_t signal_length,
            const uint2 b_op_dex_range);


        _THREAD_FUNCTION_
        void _FFT1D_R4_fp32_C2C_first_ST(const double* __restrict src, double* __restrict dst, const size_t signal_length,
            const uint2 b_op_dex_range);


        _THREAD_FUNCTION_
        void _FFT1D_R4_fp32_R2C_first_ST_vec4(const float* __restrict src, double* __restrict dst, const size_t signal_length,
            const uint2 b_op_dex_range);


        _THREAD_FUNCTION_
        void _IFFT1D_R4_fp32_C2C_first_ST_vec4(const double* __restrict src, double* __restrict dst, const size_t signal_length,
            const uint2 b_op_dex_range);


        _THREAD_FUNCTION_
        void _FFT1D_R4_fp32_C2C_first_ST_vec4(const double* __restrict src, double* __restrict dst, const size_t signal_length,
            const uint2 b_op_dex_range);


        _THREAD_FUNCTION_
        void _FFT1D_R4_fp32_C2C_ST(const double* __restrict src, double* __restrict dst, const size_t signal_length, 
            const size_t warp_proc_len, const uint2 b_op_dex_range);


        _THREAD_FUNCTION_
        void _FFT1D_R4_fp32_C2C_ST_vec4(const double* __restrict src, double* __restrict dst, const size_t signal_length, 
            const size_t warp_proc_len, const uint2 b_op_dex_range);
        }
    }
}




namespace decx
{
    namespace signal
    {
        namespace cpu {
            void FFT_R4_R2C_assign_task_1D_first(decx::utils::_thr_1D* t1D, decx::utils::frag_manager* f_mgr,
                const float* __restrict src, double* __restrict dst, const size_t signal_length);


            void IFFT_R4_C2C_assign_task_1D_first(decx::utils::_thr_1D* t1D, decx::utils::frag_manager* f_mgr,
                const double* __restrict src, double* __restrict dst, const size_t signal_length);


            void FFT_R4_C2C_assign_task_1D_first(decx::utils::_thr_1D* t1D, decx::utils::frag_manager* f_mgr,
                const double* __restrict src, double* __restrict dst, const size_t signal_length);


            void FFT_R4_C2C_assign_task_1D(decx::utils::_thr_1D* t1D, decx::utils::frag_manager* f_mgr,
                decx::alloc::MIF<double>* MIF_0, decx::alloc::MIF<double>* MIF_1, const size_t signal_length,
                const size_t warp_proc_len);
        }
    }
}




#endif