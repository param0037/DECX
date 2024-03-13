/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _CONV2_FP32_SK_H_
#define _CONV2_FP32_SK_H_


#include "conv2_fp32_exec.h"
#include "../../../../../classes/Matrix.h"
#include "../../../../../classes/MatrixArray.h"
#include "../../../../../core/thread_management/thread_arrange.h"
#include "../../../../../core/utils/fragment_arrangment.h"
#include "conv2_fp32_NB_SK_exec.h"


namespace decx
{
    namespace conv 
    {
        template <bool _print>
        static void _conv2_fp32_SK_NB(decx::_MatrixArray* src, decx::_Matrix* kernel, decx::_MatrixArray* dst, de::DH* handle);

        template <bool _print>
        static void _conv2_fp32_SK_BC(decx::_MatrixArray* src, decx::_Matrix* kernel, decx::_MatrixArray* dst, de::DH* handle);
    }
}


template <bool _print>
static void decx::conv::_conv2_fp32_SK_NB(decx::_MatrixArray* src, decx::_Matrix* kernel, decx::_MatrixArray* dst, de::DH* handle)
{
    uint2 tmp_src_dims = make_uint2(decx::utils::ceil<uint>(src->Width() + kernel->Width() - 1, 8) * 8,
        src->Height() + kernel->Height() - 1);

    decx::PtrInfo<float> tmp_ker;
    if (decx::alloc::_host_virtual_page_malloc(&tmp_ker, kernel->get_total_bytes())) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_ALLOCATION,
            ALLOC_FAIL);
        return;
    }
    // copy kernel data to a linear memory to eliminate margins in the end of every row
    decx::_cpy2D_plane((float*)kernel->Mat.ptr, tmp_ker.ptr, kernel->Pitch(), kernel->Width(),
        make_uint2(kernel->Width(), kernel->Height()));

    const uint conc_thr = (uint)decx::cpu::_get_permitted_concurrency();
    decx::utils::frag_manager* f_mgr = new decx::utils::frag_manager;
    decx::_thread_dispatch_for_conv2(&f_mgr, dst->Height(), conc_thr, _BLOCKED_CONV2_FP32_H_, dst->Pitch() / 8);
    if (f_mgr == NULL) {
        return;
    }
    decx::utils::_thr_1D t1D(conc_thr);
    decx::_C2_MK32 conv2_mk_props(make_uint2(kernel->Width(), kernel->Height()), dst->Pitch() / 8, src->Pitch() / 8,
        dst->_plane, dst->ArrayNumber, f_mgr, src->_plane);

    decx::conv::_conv2_r8_NB_SK_fp32_organiser((float*)src->MatArr.ptr, tmp_ker.ptr, (float*)dst->MatArr.ptr,
        make_uint2(dst->Pitch() / 8, dst->Height()), &t1D, &conv2_mk_props);

    decx::alloc::_host_virtual_page_dealloc(&tmp_ker);
}


template <bool _print>
static void decx::conv::_conv2_fp32_SK_BC(decx::_MatrixArray* src, decx::_Matrix* kernel, decx::_MatrixArray* dst, de::DH* handle)
{
    decx::PtrInfo<float> tmp_ker;
    if (decx::alloc::_host_virtual_page_malloc(&tmp_ker, kernel->get_total_bytes())) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_ALLOCATION,
            ALLOC_FAIL);
        return;
    }
    // copy kernel data to a linear memory to eliminate margins in the end of every row
    decx::_cpy2D_plane((float*)kernel->Mat.ptr, tmp_ker.ptr, kernel->Pitch(), kernel->Width(),
        make_uint2(kernel->Width(), kernel->Height()));

    decx::PtrInfo<float> tmp_src;

    const uint Wsrc = decx::utils::ceil<uint>(src->Width() + kernel->Width() - 1, 8) * 8;

    const uint conc_thr = (uint)decx::cpu::_get_permitted_concurrency();
    decx::utils::frag_manager* f_mgr = NULL;
    _thread_dispatch_for_conv2(&f_mgr, dst->Height(), conc_thr, _BLOCKED_CONV2_FP32_H_, dst->Pitch() / 8);
    if (f_mgr == NULL) {
        return;
    }
    decx::utils::_thr_1D t1D(conc_thr);
    decx::_C2_MK32 conv2_mk_props(
        make_uint2(kernel->Width(), kernel->Height()), src->Pitch() / 8, Wsrc / 8, dst->_plane, dst->ArrayNumber, f_mgr);

    uint H_tmp_src;
    if (f_mgr->is_left) {
        H_tmp_src = (f_mgr->frag_num - 1) * (f_mgr->frag_len + kernel->Height() - 1) +
            f_mgr->frag_left_over + kernel->Height() - 1;
    }
    else {
        H_tmp_src = f_mgr->frag_num * (f_mgr->frag_len + kernel->Height() - 1);
    }

    if (decx::alloc::_host_virtual_page_malloc(&tmp_src, Wsrc * H_tmp_src * sizeof(float))) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_ALLOCATION,
            ALLOC_FAIL);
        return;
    }

    decx::conv::_conv2_SK_BC_fp32_organiser((float*)src->MatArr.ptr, tmp_src.ptr, tmp_ker.ptr, (float*)dst->MatArr.ptr,
        make_uint2(dst->Pitch() / 8, dst->Height()), &conv2_mk_props, &t1D);

    decx::alloc::_host_virtual_page_dealloc(&tmp_ker);
}


#endif