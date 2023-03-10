/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/


#ifndef _RCP2_SQDIFF_FP32_H_
#define _RCP2_SQDIFF_FP32_H_


#include "rcp2_SQDIFF_fp32_exec.h"
#include "../../../../classes/Matrix.h"
#include "../../../../basic_process/rect_and_cube/CPU/rect_copy2D_exec.h"
#include "../../../../convolution/conv_utils.h"
#include "../utils/template_configurations.h"


namespace decx
{
    namespace rcp {
        template <bool is_norm>
        static void _rcp2_SQDIFF_fp32_NB(decx::_Matrix* src, decx::_Matrix* kernel, decx::_Matrix* dst, de::DH* handle);
    }
}


template <bool is_norm>
static void decx::rcp::_rcp2_SQDIFF_fp32_NB(decx::_Matrix* src, decx::_Matrix* kernel, decx::_Matrix* dst, de::DH* handle)
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
    decx::_thread_dispatch_for_conv2(&f_mgr, dst->height, conc_thr, _BLOCKED_RCP2_FP32_H_, dst->pitch / 8);
    if (f_mgr == NULL) {
        return;
    }
    decx::utils::_thr_1D t1D(conc_thr);

    if (is_norm) {
        // get sqrt sum of kernel
        float _sqrt_k_sum = 0;
        decx::rcp::CPUK::_template_sq_sum_vec8_fp32((const float*)kernel->Mat.ptr, kernel->_element_num / 8, &_sqrt_k_sum);
        _sqrt_k_sum = sqrtf(_sqrt_k_sum);

        decx::rcp::_rcp2_SQDIFF_fp32_organiser<true>((float*)src->Mat.ptr, tmp_ker.ptr, (float*)dst->Mat.ptr, _sqrt_k_sum,
            make_uint2(dst->pitch / 8, dst->height), make_uint2(kernel->width, kernel->height),
            src->pitch / 8, dst->pitch / 8, &t1D, f_mgr);
    }
    else {
        decx::rcp::_rcp2_SQDIFF_fp32_organiser<false>((float*)src->Mat.ptr, tmp_ker.ptr, (float*)dst->Mat.ptr, 0,
            make_uint2(dst->pitch / 8, dst->height), make_uint2(kernel->width, kernel->height),
            src->pitch / 8, dst->pitch / 8, &t1D, f_mgr);
    }
    decx::alloc::_host_virtual_page_dealloc(&tmp_ker);
}

#endif