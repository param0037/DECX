/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _GEMM_CALLERS_H_
#define _GEMM_CALLERS_H_

#include "../GEMM_utils.h"
#include "cpu_GEMM_config.h"

namespace decx
{
namespace blas {
    template <bool _ABC>
    void GEMM_fp32_caller(const float* A, const float* B, float* dst, const decx::_matrix_layout* layout_A,
        const decx::_matrix_layout* layout_dst, const uint32_t Llen, const decx::utils::frag_manager *f_mgrH,
        const decx::blas::GEMM_blocking_config* _thread_configs, decx::utils::_thr_2D* t1D, const float* C = NULL);


    template <bool _ABC, bool _cplxf>
    void GEMM_64b_caller(const double* A, const double* B, double* dst, const decx::_matrix_layout* layout_A,
        const decx::_matrix_layout* layout_dst, const uint32_t Llen, const decx::utils::frag_manager* f_mgrH,
        const decx::blas::GEMM_blocking_config* _thread_configs, decx::utils::_thr_2D* t1D, const double* C = NULL);


    template <bool _ABC>
    void GEMM_cplxd_caller(const de::CPd* A, const de::CPd* B, de::CPd* dst, const decx::_matrix_layout* layout_A,
        const decx::_matrix_layout* layout_dst, const uint32_t Llen, const decx::utils::frag_manager* f_mgrH,
        const decx::blas::GEMM_blocking_config* _thread_configs, decx::utils::_thr_2D* t1D, const de::CPd* C = NULL);



    template <bool _ABC>
    void GEMM_fp32(decx::_Matrix* A, decx::_Matrix* B, decx::_Matrix* dst, de::DH* handle, decx::_Matrix* C = NULL);


    template <bool _ABC, bool _cplxf>
    void GEMM_64b(decx::_Matrix* A, decx::_Matrix* B, decx::_Matrix* dst, de::DH* handle, decx::_Matrix* C = NULL);


    template <bool _ABC>
    void GEMM_cplxd(decx::_Matrix* A, decx::_Matrix* B, decx::_Matrix* dst, de::DH* handle, decx::_Matrix* C = NULL);


    namespace CPUK {
        typedef void 
        (*GEMM_64b_kernel)(const double* __restrict, const double* __restrict, double* __restrict, 
        const decx::blas::GEMM_blocking_config*, const uint32_t, const uint32_t, const uint32_t, const double* __restrict);
    }
}
}




#endif