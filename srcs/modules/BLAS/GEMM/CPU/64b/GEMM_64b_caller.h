/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _GEMM_FP64_CALLER_H_
#define _GEMM_FP64_CALLER_H_


#include "../../../../classes/Matrix.h"
#include "GEMM_Matrix_B_arrange_64b.h"
#include "../../../../classes/classes_util.h"
#include "GEMM_organizer_64b.h"
#include "GEMM_ABC_organizer_64b.h"
#include "../../../basic_process/rect_and_cube/CPU/rect_copy2D_exec.h"
#include "../../GEMM_utils.h"



namespace decx
{
    namespace cpu
    {
        template <bool _is_cpl, bool _sync>
        static void GEMM_64b(decx::_Matrix* A, decx::_Matrix* B, decx::_Matrix* dst, de::DH *handle);

        template <bool _is_cpl, bool _sync>
        static void GEMM_64b_ABC(decx::_Matrix* A, decx::_Matrix* B, decx::_Matrix* C, decx::_Matrix* dst, de::DH *handle);
    }
}


template <bool _is_cpl, bool _sync>
void decx::cpu::GEMM_64b(decx::_Matrix* _A, decx::_Matrix* _B, decx::_Matrix* _dst, de::DH* handle)
{
    decx::PtrInfo<double> arranged_B, extra_dst;
    uint2 arranged_B_dims = make_uint2(_A->Pitch() * 8, decx::utils::ceil<uint>(_B->Pitch(), 8));     // assume that it is 8x

    const bool is_L8 = _B->Pitch() % 8;

    if (decx::alloc::_host_virtual_page_malloc(
        &arranged_B, arranged_B_dims.x * arranged_B_dims.y * sizeof(double), true)) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_ALLOCATION,
            ALLOC_FAIL);
    }

    const uint thr_num = (uint)decx::cpu::_get_permitted_concurrency();

    decx::utils::_thr_2D t2D(3, 4);
    decx::utils::frag_manager f_mgr_sort_B, pre_f_mgrW;
    
    uint pitch_extdst = 0;      // the pitch of extra_cache_dst
    if (is_L8) {
        /* The model of _B->Pitch() is 16N + 8. 
         * In the program below, fullfilling the pitch is used, which is 16N + 8 + 8 = 16(N+1) */
        decx::utils::frag_manager_gen(&pre_f_mgrW, (_B->Pitch() + 4) / 8, t2D.thread_w);

        // assign value for the pitch of extra_cache_dst, which is the proc_w of the thread that process the end of a row
        pitch_extdst = pre_f_mgrW.is_left ? pre_f_mgrW.frag_left_over : pre_f_mgrW.frag_len;

        // allocate memory space for extra_dst
        if (decx::alloc::_host_virtual_page_malloc(&extra_dst, pitch_extdst * _dst->Height() * 8 * sizeof(double), true)) {
            decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_ALLOCATION,
                ALLOC_FAIL);
        }
        // generate the configuration for sorting matrix B
        decx::utils::frag_manager_gen(&f_mgr_sort_B, (_B->Pitch() + 4) / 8, t2D.total_thread);

        // rearrange matrix B
        decx::arrange_MatB_fp64_caller((double*)_B->Mat.ptr, arranged_B.ptr, _B->Pitch(), arranged_B_dims.x, _B->Height(), true, &t2D, &f_mgr_sort_B);

        // execute the GEMM_fp32 kernels
        decx::GEMM_AB_fp64_caller_8x<_is_cpl>((double*)_A->Mat.ptr, arranged_B.ptr, (double*)_dst->Mat.ptr, extra_dst.ptr, _A->Pitch(), arranged_B_dims.x, _dst->Pitch(),
            pitch_extdst * 8, make_uint2((_B->Pitch() + 4) / 8, _dst->Height()), _A->Pitch(), &t2D);

        // copy the data from extra_cache_dst to _dst->Mat.ptr
        double* start_dst = DECX_PTR_SHF_XY<double, double>((double*)_dst->Mat.ptr, make_uint2(0, pre_f_mgrW.frag_len * (t2D.thread_w - 1) * 8), _dst->Pitch());
        decx::gemm::CPUK::GEMM_fp64_cpy_L8(extra_dst.ptr, start_dst, pitch_extdst * 8, _dst->Pitch(), make_uint2(pitch_extdst * 2 - 1, _dst->Height()));
        // deallocate temporary space of extra_cache_dst
        decx::alloc::_host_virtual_page_dealloc(&extra_dst);
    }
    else {
        // generate the configuration for sorting matrix B
        decx::utils::frag_manager_gen(&f_mgr_sort_B, arranged_B_dims.y, t2D.total_thread);

        // rearrange matrix B
        decx::arrange_MatB_fp64_caller((double*)_B->Mat.ptr, arranged_B.ptr, _B->Pitch(), arranged_B_dims.x, _B->Height(), false, &t2D, &f_mgr_sort_B);

        // execute the GEMM_fp32 kernels
        decx::GEMM_AB_fp64_caller_16x<_is_cpl>((double*)_A->Mat.ptr, arranged_B.ptr, (double*)_dst->Mat.ptr, _A->Pitch(), arranged_B_dims.x, _dst->Pitch(),
            make_uint2(_B->Pitch() / 8, _dst->Height()), _A->Pitch(), &t2D);
    }
    // deallocate temporary space of arranged matrix B
    decx::alloc::_host_virtual_page_dealloc(&arranged_B);
}


template <bool _is_cpl, bool _sync>
void decx::cpu::GEMM_64b_ABC(decx::_Matrix* _A, decx::_Matrix* _B, decx::_Matrix* _C, decx::_Matrix* _dst, de::DH* handle)
{
    decx::PtrInfo<double> arranged_B, extra_dst;
    uint2 arranged_B_dims = make_uint2(_A->Pitch() * 8, decx::utils::ceil<uint>(_B->Pitch(), 8));     // assume that it is 16x

    const bool is_L8 = _B->Pitch() % 8;

    if (decx::alloc::_host_virtual_page_malloc(
        &arranged_B, arranged_B_dims.x * arranged_B_dims.y * sizeof(double), true)) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_ALLOCATION,
            ALLOC_FAIL);
    }

    const uint thr_num = (uint)decx::cpu::_get_permitted_concurrency();

    decx::utils::_thr_2D t2D(3, 4);
    decx::utils::frag_manager f_mgr_sort_B, pre_f_mgrW;
    
    uint pitch_extdst = 0;      // the pitch of extra_cache_dst
    if (is_L8) {
        /* The model of _B->Pitch() is 16N + 8. 
         * In the program below, fullfilling the pitch is used, which is 16N + 8 + 8 = 16(N+1) */
        decx::utils::frag_manager_gen(&pre_f_mgrW, (_B->Pitch() + 4) / 8, t2D.thread_w);

        // assign value for the pitch of extra_cache_dst, which is the proc_w of the thread that process the end of a row
        pitch_extdst = pre_f_mgrW.is_left ? pre_f_mgrW.frag_left_over : pre_f_mgrW.frag_len;

        // allocate memory space for extra_dst
        if (decx::alloc::_host_virtual_page_malloc(&extra_dst, pitch_extdst * _dst->Height() * 8 * sizeof(double), true)) {
            decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_ALLOCATION,
                ALLOC_FAIL);
        }
        // generate the configuration for sorting matrix B
        decx::utils::frag_manager_gen(&f_mgr_sort_B, (_B->Pitch() + 4) / 8, t2D.total_thread);

        // rearrange matrix B
        decx::arrange_MatB_fp64_caller((double*)_B->Mat.ptr, arranged_B.ptr, _B->Pitch(), arranged_B_dims.x, _B->Height(), true, &t2D, &f_mgr_sort_B);

        // execute the GEMM_fp64 kernels
        decx::GEMM_ABC_fp64_caller_8x<_is_cpl>((double*)_A->Mat.ptr, arranged_B.ptr, (double*)_C->Mat.ptr, (double*)_dst->Mat.ptr, extra_dst.ptr, 
            _A->Pitch(), arranged_B_dims.x, _C->Pitch(), _dst->Pitch(),
            pitch_extdst * 8, make_uint2((_B->Pitch() + 4) / 8, _dst->Height()), _A->Pitch(), &t2D);

        // copy the data from extra_cache_dst to _dst->Mat.ptr
        double* start_dst = DECX_PTR_SHF_XY<double, double>((double*)_dst->Mat.ptr, make_uint2(0, pre_f_mgrW.frag_len * (t2D.thread_w - 1) * 8), _dst->Pitch());
        decx::gemm::CPUK::GEMM_fp64_cpy_L8(extra_dst.ptr, start_dst, pitch_extdst * 8, _dst->Pitch(), make_uint2(pitch_extdst * 2 - 1, _dst->Height()));
        // deallocate temporary space of extra_cache_dst
        decx::alloc::_host_virtual_page_dealloc(&extra_dst);
    }
    else {
        // generate the configuration for sorting matrix B
        decx::utils::frag_manager_gen(&f_mgr_sort_B, arranged_B_dims.y, t2D.total_thread);

        // rearrange matrix B
        decx::arrange_MatB_fp64_caller((double*)_B->Mat.ptr, arranged_B.ptr, _B->Pitch(), arranged_B_dims.x, _B->Height(), false, &t2D, &f_mgr_sort_B);

        // execute the GEMM_fp64 kernels
        decx::GEMM_ABC_fp64_caller_16x<_is_cpl>((double*)_A->Mat.ptr, arranged_B.ptr, (double*)_C->Mat.ptr, (double*)_dst->Mat.ptr,
            _A->Pitch(), arranged_B_dims.x, _C->Pitch(), _dst->Pitch(),
            make_uint2(_B->Pitch() / 8, _dst->Height()), _A->Pitch(), &t2D);
    }
    // deallocate temporary space of arranged matrix B
    decx::alloc::_host_virtual_page_dealloc(&arranged_B);
}



#endif