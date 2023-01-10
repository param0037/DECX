/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/


#ifndef _CONV2_FP64_H_
#define _CONV2_FP64_H_


#include "conv2_fp64_exec.h"
#include "../../../../classes/Matrix.h"


namespace decx
{
    namespace conv {
        static void _conv2_fp64_NB(decx::_Matrix* src, decx::_Matrix* kernel, decx::_Matrix* dst, de::DH* handle);



        static void _conv2_fp64_BC(decx::_Matrix* src, decx::_Matrix* kernel, decx::_Matrix* dst, de::DH* handle);
    }
}



static void decx::conv::_conv2_fp64_NB(decx::_Matrix* src, decx::_Matrix* kernel, decx::_Matrix* dst, de::DH* handle)
{
    decx::PtrInfo<double> tmp_ker;
    if (decx::alloc::_host_virtual_page_malloc(&tmp_ker, kernel->total_bytes)) {
        decx::err::AllocateFailure(handle);
        Print_Error_Message(4, ALLOC_FAIL);
        return;
    }
    decx::_cpy2D_plane((double*)kernel->Mat.ptr, tmp_ker.ptr, kernel->pitch, kernel->width,
        make_uint2(kernel->width, kernel->height));

    const uint conc_thr = (uint)decx::cpI.cpu_concurrency;
    decx::utils::frag_manager* f_mgr = NULL;
    _thread_dispatch_for_conv2(&f_mgr, dst->height, conc_thr, _BLOCKED_CONV2_FP64_H_, dst->pitch / _MATRIX_ALIGN_8B_);
    if (f_mgr == NULL) {
        return;
    }
    decx::utils::_thr_1D t1D(conc_thr);

    decx::_conv2_fp64_organiser((double*)src->Mat.ptr, tmp_ker.ptr, (double*)dst->Mat.ptr,
        make_uint2(dst->pitch / _MATRIX_ALIGN_8B_, dst->height), make_uint2(kernel->width, kernel->height),
        src->pitch / _MATRIX_ALIGN_8B_, dst->pitch / _MATRIX_ALIGN_8B_, &t1D, f_mgr);

    decx::alloc::_host_virtual_page_dealloc(&tmp_ker);
}



static void decx::conv::_conv2_fp64_BC(decx::_Matrix* src, decx::_Matrix* kernel, decx::_Matrix* dst, de::DH* handle)
{
    uint2 tmp_src_dims = make_uint2(decx::utils::ceil<uint>(src->width + kernel->width - 1, 8) * 8,
        src->height + kernel->height - 1);

    decx::PtrInfo<double> tmp_ker, tmp_src;
    if (decx::alloc::_host_virtual_page_malloc(&tmp_ker, kernel->total_bytes)) {
        decx::err::AllocateFailure(handle);
        Print_Error_Message(4, ALLOC_FAIL);
        return;
    }
    if (decx::alloc::_host_virtual_page_malloc(&tmp_src, (size_t)tmp_src_dims.x * (size_t)tmp_src_dims.y * sizeof(double))) {
        decx::err::AllocateFailure(handle);
        Print_Error_Message(4, ALLOC_FAIL);
        return;
    }

    decx::_cpy2D_plane<double>((double*)kernel->Mat.ptr, tmp_ker.ptr, kernel->pitch, kernel->width,
        make_uint2(kernel->width, kernel->height));

    double* start_place_src = DECX_PTR_SHF_XY<double, double>(tmp_src.ptr, kernel->height / 2, kernel->width / 2, tmp_src_dims.x);

    decx::_cpy2D_32bit_caller<double>(
        (double*)src->Mat.ptr, start_place_src, src->pitch, tmp_src_dims.x, make_uint2(src->width, src->height));

    const uint conc_thr = (uint)decx::cpI.cpu_concurrency;
    decx::utils::frag_manager* f_mgr = NULL;
    _thread_dispatch_for_conv2(&f_mgr, dst->height, conc_thr, _BLOCKED_CONV2_FP64_H_, dst->pitch / 4);
    if (f_mgr == NULL) {
        return;
    }
    decx::utils::_thr_1D t1D(conc_thr);

    decx::_conv2_fp64_organiser(tmp_src.ptr, tmp_ker.ptr, (double*)dst->Mat.ptr,
        make_uint2(dst->pitch / _MATRIX_ALIGN_8B_, dst->height), make_uint2(kernel->width, kernel->height),
        tmp_src_dims.x / _MATRIX_ALIGN_8B_, src->pitch / _MATRIX_ALIGN_8B_, &t1D, f_mgr);

    decx::alloc::_host_virtual_page_dealloc(&tmp_ker);
    decx::alloc::_host_virtual_page_dealloc(&tmp_src);
}



#endif