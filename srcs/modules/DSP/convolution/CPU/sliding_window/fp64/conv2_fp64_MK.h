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



#ifndef _CONV2_FP64_MK_H_
#define _CONV2_FP64_MK_H_


#include "conv2_fp64_exec.h"
#include "../../../../../classes/Matrix.h"
#include "../../../../../classes/MatrixArray.h"
#include "../../../../../core/thread_management/thread_arrange.h"
#include "../../../../../core/utils/fragment_arrangment.h"
#include "conv2_fp64_NB_MK_exec.h"
#include "conv2_fp64_BC_MK_exec.h"


namespace decx
{
    namespace conv 
    {
        static void _conv2_fp64_MK_NB(decx::_MatrixArray* src, decx::_MatrixArray* kernel, decx::_MatrixArray* dst, de::DH* handle);


        static void _conv2_fp64_MK_BC(decx::_MatrixArray* src, decx::_MatrixArray* kernel, decx::_MatrixArray* dst, de::DH* handle);
    }
}


static void decx::conv::_conv2_fp64_MK_NB(decx::_MatrixArray* src, decx::_MatrixArray* kernel, decx::_MatrixArray* dst, de::DH* handle)
{
    uint2 tmp_src_dims = make_uint2(decx::utils::ceil<uint>(src->Width() + kernel->Width() - 1, _MATRIX_ALIGN_8B_) * _MATRIX_ALIGN_8B_,
        src->Height() + kernel->Height() - 1);

    decx::PtrInfo<double> tmp_ker;
    if (decx::alloc::_host_virtual_page_malloc(&tmp_ker, kernel->_element_num * sizeof(double))) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_ALLOCATION,
            ALLOC_FAIL);
        return;
    }

    decx::_cpy2D_plane((double*)kernel->MatArr.ptr, tmp_ker.ptr, kernel->Pitch(), kernel->Width(),
        make_uint2(kernel->Width(), kernel->Height() * kernel->MatrixNumber()));

    const uint conc_thr = (uint)decx::cpu::_get_permitted_concurrency();
    decx::utils::frag_manager* f_mgr = NULL;
    decx::_thread_dispatch_for_conv2_fp64(&f_mgr, dst->Height(), conc_thr, _BLOCKED_CONV2_FP64_H_, dst->Pitch() / _MATRIX_ALIGN_8B_);
    if (f_mgr == NULL) {
        return;
    }
    decx::utils::_thr_1D t1D(conc_thr);

    decx::_C2_MK64 conv2_mk_props(make_uint2(kernel->Width(), kernel->Height()),
        dst->Pitch() / _MATRIX_ALIGN_8B_, src->Pitch() / _MATRIX_ALIGN_8B_,
        dst->_plane, dst->MatrixNumber(), f_mgr,
        src->_plane, kernel->plane);

    decx::conv::_conv2_NB_MK_fp64_organiser((double*)src->MatArr.ptr, tmp_ker.ptr, (double*)dst->MatArr.ptr,
        make_uint2(dst->Pitch() / _MATRIX_ALIGN_8B_, dst->Height()), &t1D, &conv2_mk_props);

    decx::alloc::_host_virtual_page_dealloc(&tmp_ker);
}



static void decx::conv::_conv2_fp64_MK_BC(decx::_MatrixArray* src, decx::_MatrixArray* kernel, decx::_MatrixArray* dst, de::DH* handle)
{
    decx::PtrInfo<double> tmp_ker;
    if (decx::alloc::_host_virtual_page_malloc(&tmp_ker, kernel->total_bytes)) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_ALLOCATION,
            ALLOC_FAIL);
        return;
    }
    // copy kernel data to a linear memory to eliminate margins in the end of every row
    decx::_cpy2D_plane((double*)kernel->MatArr.ptr, tmp_ker.ptr, kernel->Pitch(), kernel->Width(),
        make_uint2(kernel->Width(), kernel->Height() * kernel->ArrayNumber));

    decx::PtrInfo<double> tmp_src;

    const uint Wsrc = decx::utils::ceil<uint>(src->Width() + kernel->Width() - 1, _MATRIX_ALIGN_8B_) * _MATRIX_ALIGN_8B_;

    const uint conc_thr = (uint)decx::cpu::_get_permitted_concurrency();
    decx::utils::frag_manager *f_mgr = NULL;
    decx::_thread_dispatch_for_conv2_fp64(&f_mgr, dst->Height(), conc_thr, _BLOCKED_CONV2_FP32_H_, dst->Pitch() / _MATRIX_ALIGN_8B_);
    if (f_mgr == NULL) {
        return;
    }

    decx::utils::_thr_1D t1D(conc_thr);

    decx::_C2_MK64 conv2_mk_props(make_uint2(kernel->Width(), kernel->Height()), 
        src->Pitch() / _MATRIX_ALIGN_8B_, Wsrc / _MATRIX_ALIGN_8B_, dst->_plane,
        dst->ArrayNumber, f_mgr, src->_plane, kernel->plane);

    uint H_tmp_src;
    if (f_mgr->is_left) {
        H_tmp_src = (f_mgr->frag_num - 1) * (f_mgr->frag_len + kernel->Height() - 1) +
            f_mgr->frag_left_over + kernel->Height() - 1;
    }
    else {
        H_tmp_src = f_mgr->frag_num * (f_mgr->frag_len + kernel->Height() - 1);
    }

    if (decx::alloc::_host_virtual_page_malloc(&tmp_src, Wsrc * H_tmp_src * sizeof(double))) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_ALLOCATION,
            ALLOC_FAIL);
        return;
    }

    decx::conv::_conv2_MK_BC_fp64_organiser((double*)src->MatArr.ptr, tmp_src.ptr, tmp_ker.ptr, (double*)dst->MatArr.ptr,
        make_uint2(dst->Pitch() / _MATRIX_ALIGN_8B_, dst->Height()), &conv2_mk_props, &t1D);

    decx::alloc::_host_virtual_page_dealloc(&tmp_ker);
}


#endif