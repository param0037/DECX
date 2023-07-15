/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _RCP2_SQDIFF_UINT8_H_
#define _RCP2_SQDIFF_UINT8_H_


#include "rcp2_SQDIFF_uint8_exec.h"
#include "../../../../classes/Matrix.h"
#include "../../../../BLAS/basic_process/rect_and_cube/CPU/rect_copy2D_exec.h"
#include "../../../../DSP/convolution/conv_utils.h"
#include "../utils/template_configurations.h"


namespace decx
{
    namespace rcp {
        template <bool is_norm>
        static void _rcp2_SQDIFF_uint8(decx::_Matrix* src, decx::_Matrix* kernel, decx::_Matrix* dst, de::DH* handle);
    }
}


template <bool is_norm>
static void decx::rcp::_rcp2_SQDIFF_uint8(decx::_Matrix* src, decx::_Matrix* kernel, decx::_Matrix* dst, de::DH* handle)
{
    decx::PtrInfo<uint8_t> tmp_ker;
    if (decx::alloc::_host_virtual_page_malloc(&tmp_ker, kernel->Pitch() * kernel->Height())) {
        decx::err::AllocateFailure(handle);
        Print_Error_Message(4, ALLOC_FAIL);
        return;
    }
    decx::_cpy2D_plane((uint8_t*)kernel->Mat.ptr, tmp_ker.ptr, kernel->Pitch(), kernel->Width(),
        make_uint2(kernel->Width(), kernel->Height()));

    const uint conc_thr = (uint)decx::cpu::_get_permitted_concurrency();
    decx::utils::frag_manager* f_mgr = NULL;
    decx::_thread_dispatch_for_conv2(&f_mgr, dst->Height(), conc_thr, _BLOCKED_RCP2_FP32_H_, dst->Pitch() / 8);
    if (f_mgr == NULL) {
        return;
    }
    decx::utils::_thr_1D t1D(conc_thr);

    if (is_norm) {
        // get sqrt sum of kernel
        float _sqrt_k_sum = 0;
        decx::rcp::CPUK::_template_sq_sum_vec8_uint8((uint8_t*)kernel->Mat.ptr, kernel->Pitch() * kernel->Height() / 8, &_sqrt_k_sum);
        _sqrt_k_sum = sqrtf(_sqrt_k_sum);

        decx::rcp::_rcp2_SQDIFF_uint8_organiser<true>((double*)src->Mat.ptr, tmp_ker.ptr, (float*)dst->Mat.ptr, _sqrt_k_sum,
            make_uint2(dst->Pitch() / 8, dst->Height()), make_uint2(kernel->Width(), kernel->Height()),
            src->Pitch() / 8, dst->Pitch() / 8, &t1D, f_mgr);
    }
    else {
        decx::rcp::_rcp2_SQDIFF_uint8_organiser<false>((double*)src->Mat.ptr, tmp_ker.ptr, (float*)dst->Mat.ptr, 0,
            make_uint2(dst->Pitch() / 8, dst->Height()), make_uint2(kernel->Width(), kernel->Height()),
            src->Pitch() / 8, dst->Pitch() / 8, &t1D, f_mgr);
    }
    decx::alloc::_host_virtual_page_dealloc(&tmp_ker);
}

#endif