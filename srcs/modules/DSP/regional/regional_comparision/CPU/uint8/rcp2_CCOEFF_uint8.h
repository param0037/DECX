/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _RCP2_CCOEFF_UINT8_H_
#define _RCP2_CCOEFF_UINT8_H_


#include "rcp2_CCOEFF_uint8_exec.h"
#include "../../../../classes/Matrix.h"
#include "../../../../BLAS/basic_process/rect_and_cube/CPU/rect_copy2D_exec.h"
#include "../../../../DSP/convolution/conv_utils.h"
#include "../utils/template_configurations.h"
#include "../../../../basic_calculations/operators/Matrix/integral.h"


namespace decx
{
    namespace rcp {
        template <bool is_norm>
        static void _rcp2_CCOEFF_uint8(decx::_Matrix* src, decx::_Matrix* kernel, decx::_Matrix* dst, de::DH* handle);
    }
}



template <bool is_norm>
static void decx::rcp::_rcp2_CCOEFF_uint8(decx::_Matrix* src, decx::_Matrix* kernel, decx::_Matrix* dst, de::DH* handle)
{
    decx::PtrInfo<float> tmp_ker, tmp_Isrc;
    if (decx::alloc::_host_virtual_page_malloc(&tmp_ker, kernel->Pitch() * kernel->Height())) {
        decx::err::AllocateFailure(handle);
        Print_Error_Message(4, ALLOC_FAIL);
        return;
    }
    const uint2 Isrc_dims = make_uint2(decx::utils::ceil<uint>(src->Width() + 1, 8) * 8, src->Height() + 1);
    if (decx::alloc::_host_virtual_page_malloc(&tmp_Isrc, Isrc_dims.x * Isrc_dims.y * sizeof(float))) {
        decx::err::AllocateFailure(handle);
        Print_Error_Message(4, ALLOC_FAIL);
        return;
    }

    decx::rcp::CPUK::_template_normalize_uint8_cpy2D((uint8_t*)kernel->Mat.ptr, tmp_ker.ptr, kernel->Pitch() * kernel->Height() / 8,
        kernel->Pitch(), kernel->Width(), kernel->Width(), kernel->Height());

    float* begin_I_src = DECX_PTR_SHF_XY_SAME_TYPE(tmp_Isrc.ptr, 1, 1, Isrc_dims.x);
    decx::calc::_integral_caller2D_uint8_f32((uint8_t*)src->Mat.ptr,
                                             begin_I_src,
                                             make_uint2(src->Width(), src->Height()), 
                                             src->Pitch(), Isrc_dims.x);

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
        decx::rcp::CPUK::_template_sq_sum_vec8_fp32(tmp_ker.ptr, kernel->Pitch() * kernel->Height() / 8, &_sqrt_k_sum);
        _sqrt_k_sum = sqrtf(_sqrt_k_sum);

        decx::rcp::_rcp2_CCOEFF_uint8_organiser<true>((double*)src->Mat.ptr, tmp_ker.ptr, begin_I_src, (float*)dst->Mat.ptr, _sqrt_k_sum,
            make_uint2(dst->Pitch() / 8, dst->Height()), make_uint2(kernel->Width(), kernel->Height()),
            src->Pitch() / 8, Isrc_dims.x / 8, dst->Pitch() / 8, &t1D, f_mgr);
    }
    else {
        decx::rcp::_rcp2_CCOEFF_uint8_organiser<false>((double*)src->Mat.ptr, tmp_ker.ptr, begin_I_src, (float*)dst->Mat.ptr, 0,
            make_uint2(dst->Pitch() / 8, dst->Height()), make_uint2(kernel->Width(), kernel->Height()),
            src->Pitch() / 8, Isrc_dims.x / 8, dst->Pitch() / 8, &t1D, f_mgr);
    }
    decx::alloc::_host_virtual_page_dealloc(&tmp_ker);
}



#endif