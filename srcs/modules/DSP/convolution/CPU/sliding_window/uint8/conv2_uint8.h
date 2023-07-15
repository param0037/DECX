/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _CONV2_UINT8_H_
#define _CONV2_UINT8_H_

#include "conv2_uint8_exec.h"
#include "../../../../classes/Matrix.h"


namespace decx
{
    namespace conv 
    {
        template <bool _print>
        static void _conv2_uint8_NB(decx::_Matrix* src, decx::_Matrix* kernel, void* tmp_ker, decx::_Matrix* dst, de::DH* handle, const int _output_type);


        template <bool _print>
        static void _conv2_uint8_BC(decx::_Matrix* src, decx::_Matrix* kernel, void* tmp_ker, decx::_Matrix* dst, de::DH* handle, const int _output_type);
    }
}


template <bool _print>
static void decx::conv::_conv2_uint8_NB(decx::_Matrix* src, decx::_Matrix* kernel, void * tmp_ker, decx::_Matrix* dst, de::DH* handle, const int _store_type)
{
    const uint conc_thr = (uint)decx::cpu::_get_permitted_concurrency();
    decx::utils::frag_manager* f_mgr = NULL;
    _thread_dispatch_for_conv2(&f_mgr, dst->Height(), conc_thr, _BLOCKED_CONV2_UINT8_H_, dst->Pitch() / 8);
    if (f_mgr == NULL) {
        return;
    }
    decx::utils::_thr_1D t1D(conc_thr);

    decx::conv::_conv2_r8_uint8_organiser((double*)src->Mat.ptr, tmp_ker, (float*)dst->Mat.ptr,
        make_uint2(dst->Pitch(), dst->Height()), make_uint2(kernel->Width(), kernel->Height()),
         src->Pitch(), dst->Pitch(), &t1D, f_mgr, _store_type);
}


template <bool _print>
static void decx::conv::_conv2_uint8_BC(decx::_Matrix* src, decx::_Matrix* kernel, void *tmp_ker, decx::_Matrix* dst, de::DH* handle, const int _store_type)
{
    uint2 tmp_src_dims = make_uint2(decx::utils::ceil<uint>(src->Width() + kernel->Width() - 1, 16) * 16,
        src->Height() + kernel->Height() - 1);

    decx::PtrInfo<uint8_t> tmp_src;
    if (decx::alloc::_host_virtual_page_malloc(&tmp_src, tmp_src_dims.x * tmp_src_dims.y * sizeof(uint8_t))) {
        decx::err::AllocateFailure<_print>(handle);
        return;
    }

    uint8_t* start_place_src = DECX_PTR_SHF_XY<uint8_t, uint8_t>(tmp_src.ptr, kernel->Height() / 2, kernel->Width() / 2, tmp_src_dims.x);

    decx::_cpy2D_anybit_caller<uint8_t>(
        (uint8_t*)src->Mat.ptr, start_place_src, src->Pitch(), tmp_src_dims.x, make_uint2(src->Width(), src->Height()));

    const uint conc_thr = (uint)decx::cpu::_get_permitted_concurrency();
    decx::utils::frag_manager* f_mgr = NULL;
    decx::_thread_dispatch_for_conv2(&f_mgr, dst->Height(), conc_thr, _BLOCKED_CONV2_UINT8_H_, dst->Pitch() / 8);
    if (f_mgr == NULL) {
        return;
    }
    decx::utils::_thr_1D t1D(conc_thr);

    decx::conv::_conv2_r8_uint8_organiser((double*)tmp_src.ptr, tmp_ker, dst->Mat.ptr,
        make_uint2(dst->Pitch(), dst->Height()), make_uint2(kernel->Width(), kernel->Height()),
        tmp_src_dims.x, dst->Pitch(), &t1D, f_mgr, _store_type);

    decx::alloc::_host_virtual_page_dealloc(&tmp_src);
}


#endif