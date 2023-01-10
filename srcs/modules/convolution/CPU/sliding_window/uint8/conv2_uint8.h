/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/

#ifndef _CONV2_UINT8_H_
#define _CONV2_UINT8_H_

#include "conv2_uint8_exec.h"
#include "../../../../classes/Matrix.h"


namespace decx
{
    namespace conv {
        static void _conv2_uint8_NB(decx::_Matrix* src, decx::_Matrix* kernel, void* tmp_ker, decx::_Matrix* dst, de::DH* handle, const int _output_type);



        static void _conv2_uint8_BC(decx::_Matrix* src, decx::_Matrix* kernel, void* tmp_ker, decx::_Matrix* dst, de::DH* handle, const int _output_type);
    }
}



static void decx::conv::_conv2_uint8_NB(decx::_Matrix* src, decx::_Matrix* kernel, void * tmp_ker, decx::_Matrix* dst, de::DH* handle, const int _store_type)
{
    const uint conc_thr = (uint)decx::cpI.cpu_concurrency;
    decx::utils::frag_manager* f_mgr = NULL;
    _thread_dispatch_for_conv2(&f_mgr, dst->height, conc_thr, _BLOCKED_CONV2_UINT8_H_, dst->pitch / 8);
    if (f_mgr == NULL) {
        return;
    }
    decx::utils::_thr_1D t1D(conc_thr);

    decx::conv::_conv2_r8_uint8_organiser((double*)src->Mat.ptr, tmp_ker, (float*)dst->Mat.ptr,
        make_uint2(dst->pitch, dst->height), make_uint2(kernel->width, kernel->height),
         src->pitch, dst->pitch, &t1D, f_mgr, _store_type);
}



static void decx::conv::_conv2_uint8_BC(decx::_Matrix* src, decx::_Matrix* kernel, void *tmp_ker, decx::_Matrix* dst, de::DH* handle, const int _store_type)
{
    uint2 tmp_src_dims = make_uint2(decx::utils::ceil<uint>(src->width + kernel->width - 1, 16) * 16,
        src->height + kernel->height - 1);

    decx::PtrInfo<uint8_t> tmp_src;
    if (decx::alloc::_host_virtual_page_malloc(&tmp_src, tmp_src_dims.x * tmp_src_dims.y * sizeof(uint8_t))) {
        decx::err::AllocateFailure(handle);
        Print_Error_Message(4, ALLOC_FAIL);
        return;
    }

    uint8_t* start_place_src = DECX_PTR_SHF_XY<uint8_t, uint8_t>(tmp_src.ptr, kernel->height / 2, kernel->width / 2, tmp_src_dims.x);

    decx::_cpy2D_32bit_caller<uint8_t>(
        (uint8_t*)src->Mat.ptr, start_place_src, src->pitch, tmp_src_dims.x, make_uint2(src->width, src->height));

    const uint conc_thr = (uint)decx::cpI.cpu_concurrency;
    decx::utils::frag_manager* f_mgr = NULL;
    _thread_dispatch_for_conv2(&f_mgr, dst->height, conc_thr, _BLOCKED_CONV2_UINT8_H_, dst->pitch / 8);
    if (f_mgr == NULL) {
        return;
    }
    decx::utils::_thr_1D t1D(conc_thr);

    decx::conv::_conv2_r8_uint8_organiser((double*)tmp_src.ptr, tmp_ker, dst->Mat.ptr,
        make_uint2(dst->pitch, dst->height), make_uint2(kernel->width, kernel->height),
        tmp_src_dims.x, dst->pitch, &t1D, f_mgr, _store_type);

    decx::alloc::_host_virtual_page_dealloc(&tmp_src);
}


#endif