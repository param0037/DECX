/**
*   ----------------------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ----------------------------------------------------------------------------------
* 
* This is a part of the open source project named "DECX", a high-performance scientific
* computational library. This project follows the MIT License. For more information 
* please visit https://github.com/param0037/DECX.
* 
* Copyright (c) 2021 Wayne Anderson
* 
* Permission is hereby granted, free of charge, to any person obtaining a copy of this 
* software and associated documentation files (the "Software"), to deal in the Software 
* without restriction, including without limitation the rights to use, copy, modify, 
* merge, publish, distribute, sublicense, and/or sell copies of the Software, and to 
* permit persons to whom the Software is furnished to do so, subject to the following 
* conditions:
* 
* The above copyright notice and this permission notice shall be included in all copies 
* or substantial portions of the Software.
* 
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
* INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR 
* PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE 
* FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR 
* OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
* DEALINGS IN THE SOFTWARE.
*/


#ifndef _CONV2_UINT8_H_
#define _CONV2_UINT8_H_

#include "conv2_uint8_exec.h"
#include "../../../../../classes/Matrix.h"


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
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_ALLOCATION,
            ALLOC_FAIL);
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