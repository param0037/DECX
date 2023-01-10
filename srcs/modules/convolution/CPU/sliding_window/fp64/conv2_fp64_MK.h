/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/


#ifndef _CONV2_FP64_MK_H_
#define _CONV2_FP64_MK_H_


#include "conv2_fp64_exec.h"
#include "../../../../classes/Matrix.h"
#include "../../../../classes/MatrixArray.h"
#include "../../../../core/thread_management/thread_arrange.h"
#include "../../../../core/utils/fragment_arrangment.h"
#include "conv2_fp64_NB_MK_exec.h"
#include "conv2_fp64_BC_MK_exec.h"


namespace decx
{
    namespace conv {
        static void _conv2_fp64_MK_NB(decx::_MatrixArray* src, decx::_MatrixArray* kernel, decx::_MatrixArray* dst, de::DH* handle);


        static void _conv2_fp64_MK_BC(decx::_MatrixArray* src, decx::_MatrixArray* kernel, decx::_MatrixArray* dst, de::DH* handle);
    }
}


static void decx::conv::_conv2_fp64_MK_NB(decx::_MatrixArray* src, decx::_MatrixArray* kernel, decx::_MatrixArray* dst, de::DH* handle)
{
    uint2 tmp_src_dims = make_uint2(decx::utils::ceil<uint>(src->width + kernel->width - 1, _MATRIX_ALIGN_8B_) * _MATRIX_ALIGN_8B_,
        src->height + kernel->height - 1);

    decx::PtrInfo<double> tmp_ker;
    if (decx::alloc::_host_virtual_page_malloc(&tmp_ker, kernel->_element_num * sizeof(double))) {
        decx::err::AllocateFailure(handle);
        Print_Error_Message(4, ALLOC_FAIL);
        return;
    }

    decx::_cpy2D_plane((double*)kernel->MatArr.ptr, tmp_ker.ptr, kernel->pitch, kernel->width,
        make_uint2(kernel->width, kernel->height * kernel->ArrayNumber));

    const uint conc_thr = (uint)decx::cpI.cpu_concurrency;
    decx::utils::frag_manager* f_mgr = NULL;
    _thread_dispatch_for_conv2_fp64(&f_mgr, dst->height, conc_thr, _BLOCKED_CONV2_FP64_H_, dst->pitch / _MATRIX_ALIGN_8B_);
    if (f_mgr == NULL) {
        return;
    }
    decx::utils::_thr_1D t1D(conc_thr);

    decx::_C2_MK64 conv2_mk_props(make_uint2(kernel->width, kernel->height),
        dst->pitch / _MATRIX_ALIGN_8B_, src->pitch / _MATRIX_ALIGN_8B_,
        dst->_plane, dst->ArrayNumber, f_mgr,
        src->_plane, kernel->plane);

    decx::conv::_conv2_NB_MK_fp64_organiser((double*)src->MatArr.ptr, tmp_ker.ptr, (double*)dst->MatArr.ptr,
        make_uint2(dst->pitch / _MATRIX_ALIGN_8B_, dst->height), &t1D, &conv2_mk_props);

    decx::alloc::_host_virtual_page_dealloc(&tmp_ker);
}



static void decx::conv::_conv2_fp64_MK_BC(decx::_MatrixArray* src, decx::_MatrixArray* kernel, decx::_MatrixArray* dst, de::DH* handle)
{
    decx::PtrInfo<double> tmp_ker;
    if (decx::alloc::_host_virtual_page_malloc(&tmp_ker, kernel->total_bytes)) {
        decx::err::AllocateFailure(handle);
        Print_Error_Message(4, ALLOC_FAIL);
        return;
    }
    // copy kernel data to a linear memory to eliminate margins in the end of every row
    decx::_cpy2D_plane((double*)kernel->MatArr.ptr, tmp_ker.ptr, kernel->pitch, kernel->width,
        make_uint2(kernel->width, kernel->height * kernel->ArrayNumber));

    decx::PtrInfo<double> tmp_src;

    const uint Wsrc = decx::utils::ceil<uint>(src->width + kernel->width - 1, _MATRIX_ALIGN_8B_) * _MATRIX_ALIGN_8B_;

    const uint conc_thr = (uint)decx::cpI.cpu_concurrency;
    decx::utils::frag_manager *f_mgr = NULL;
    _thread_dispatch_for_conv2_fp64(&f_mgr, dst->height, conc_thr, _BLOCKED_CONV2_FP32_H_, dst->pitch / _MATRIX_ALIGN_8B_);
    if (f_mgr == NULL) {
        return;
    }

    decx::utils::_thr_1D t1D(conc_thr);

    decx::_C2_MK64 conv2_mk_props(make_uint2(kernel->width, kernel->height), 
        src->pitch / _MATRIX_ALIGN_8B_, Wsrc / _MATRIX_ALIGN_8B_, dst->_plane,
        dst->ArrayNumber, f_mgr, src->_plane, kernel->plane);

    uint H_tmp_src;
    if (f_mgr->is_left) {
        H_tmp_src = (f_mgr->frag_num - 1) * (f_mgr->frag_len + kernel->height - 1) +
            f_mgr->frag_left_over + kernel->height - 1;
    }
    else {
        H_tmp_src = f_mgr->frag_num * (f_mgr->frag_len + kernel->height - 1);
    }

    if (decx::alloc::_host_virtual_page_malloc(&tmp_src, Wsrc * H_tmp_src * sizeof(double))) {
        decx::err::AllocateFailure(handle);
        Print_Error_Message(4, ALLOC_FAIL);
        return;
    }

    decx::conv::_conv2_MK_BC_fp64_organiser((double*)src->MatArr.ptr, tmp_src.ptr, tmp_ker.ptr, (double*)dst->MatArr.ptr,
        make_uint2(dst->pitch / _MATRIX_ALIGN_8B_, dst->height), &conv2_mk_props, &t1D);

    decx::alloc::_host_virtual_page_dealloc(&tmp_ker);
}


#endif