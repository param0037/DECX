/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/


#ifndef _CONV2_FP32_H_
#define _CONV2_FP32_H_


#include "conv2_fp32_exec.h"
#include "../../../../classes/Matrix.h"


namespace decx
{
    namespace conv {
        static void _conv2_fp32_NB(decx::_Matrix* src, decx::_Matrix* kernel, decx::_Matrix* dst, de::DH* handle);



        static void _conv2_fp32_BC(decx::_Matrix* src, decx::_Matrix* kernel, decx::_Matrix* dst, de::DH* handle);
    }
}



static void decx::conv::_conv2_fp32_NB(decx::_Matrix* src, decx::_Matrix* kernel, decx::_Matrix* dst, de::DH* handle)
{
    decx::PtrInfo<float> tmp_ker;
    if (decx::alloc::_host_virtual_page_malloc(&tmp_ker, kernel->total_bytes)) {
        decx::err::AllocateFailure(handle);
        Print_Error_Message(4, ALLOC_FAIL);
        return;
    }
    decx::_cpy2D_plane((float*)kernel->Mat.ptr, tmp_ker.ptr, kernel->pitch, kernel->width,
        make_uint2(kernel->width, kernel->height));

    const uint conc_thr = (uint)decx::cpI.cpu_concurrency;
    decx::utils::frag_manager* f_mgr = NULL;
    _thread_dispatch_for_conv2(&f_mgr, dst->height, conc_thr, _BLOCKED_CONV2_FP32_H_, dst->pitch / 8);
    if (f_mgr == NULL) {
        return;
    }
    decx::utils::_thr_1D t1D(conc_thr);

    decx::conv::_conv2_r8_fp32_organiser((float*)src->Mat.ptr, tmp_ker.ptr, (float*)dst->Mat.ptr,
        make_uint2(dst->pitch / 8, dst->height), make_uint2(kernel->width, kernel->height),
         src->pitch / 8, dst->pitch / 8, &t1D, f_mgr);
        
    decx::alloc::_host_virtual_page_dealloc(&tmp_ker);
}



static void decx::conv::_conv2_fp32_BC(decx::_Matrix* src, decx::_Matrix* kernel, decx::_Matrix* dst, de::DH* handle)
{
    uint2 tmp_src_dims = make_uint2(decx::utils::ceil<uint>(src->width + kernel->width - 1, 8) * 8,
        src->height + kernel->height - 1);

    decx::PtrInfo<float> tmp_ker, tmp_src;
    if (decx::alloc::_host_virtual_page_malloc(&tmp_ker, kernel->total_bytes)) {
        decx::err::AllocateFailure(handle);
        Print_Error_Message(4, ALLOC_FAIL);
        return;
    }
    if (decx::alloc::_host_virtual_page_malloc(&tmp_src, tmp_src_dims.x * tmp_src_dims.y * sizeof(float))) {
        decx::err::AllocateFailure(handle);
        Print_Error_Message(4, ALLOC_FAIL);
        return;
    }

    decx::_cpy2D_plane((float*)kernel->Mat.ptr, tmp_ker.ptr, kernel->pitch, kernel->width,
        make_uint2(kernel->width, kernel->height));

    float* start_place_src = DECX_PTR_SHF_XY<float, float>(tmp_src.ptr, kernel->height / 2, kernel->width / 2, tmp_src_dims.x);

    decx::_cpy2D_32bit_caller<float>(
        (float*)src->Mat.ptr, start_place_src, src->pitch, tmp_src_dims.x, make_uint2(src->width, src->height));

    const uint conc_thr = (uint)decx::cpI.cpu_concurrency;
    decx::utils::frag_manager* f_mgr = NULL;
    _thread_dispatch_for_conv2(&f_mgr, dst->height, conc_thr, _BLOCKED_CONV2_FP32_H_, dst->pitch / 8);
    if (f_mgr == NULL) {
        return;
    }
    decx::utils::_thr_1D t1D(conc_thr);

    decx::conv::_conv2_r8_fp32_organiser(tmp_src.ptr, tmp_ker.ptr, (float*)dst->Mat.ptr,
        make_uint2(dst->pitch / 8, dst->height), make_uint2(kernel->width, kernel->height),
        tmp_src_dims.x / 8, src->pitch / 8, &t1D, f_mgr);

    decx::alloc::_host_virtual_page_dealloc(&tmp_ker);
    decx::alloc::_host_virtual_page_dealloc(&tmp_src);
}

#endif