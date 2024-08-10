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

#include "../common/cpu_filter2D_planner.h"
#include "../common/filter2D_kernels.h"
#include "../../../../../../common/Basic_process/extension/CPU/extend_constant.h"
#include "../../../../../../common/Basic_process/extension/CPU/extend_reflect.h"


decx::ResourceHandle decx::dsp::g_cpu_filter2D_64b;


template <> template <bool _cplxf>
void decx::dsp::cpu_Filter2D_planner<double>::
filter2D_NB_64b(decx::_Matrix* src, 
                 decx::_Matrix* kernel, 
                 decx::_Matrix* dst,
                 decx::utils::_thr_2D* t2D)
{
    decx::dsp::CPUK::conv2_NB_kernel_64b* kernel_ptr = NULL;
    if
#if __cplusplus >= 201703L
        constexpr
#endif
        (_cplxf) {
        kernel_ptr = decx::dsp::CPUK::conv2_cplxf_kernel;
    }
    else {
        kernel_ptr = decx::dsp::CPUK::conv2_fp64_kernel;
    }

    const double* src_loc = NULL;
    double* dst_loc = NULL;
    for (uint32_t i = 0; i < t2D->thread_h; ++i)
    {
        src_loc = (double*)src->Mat.ptr + i * _thread_blocking_conf._fmgrH.frag_len * src->Pitch();
        dst_loc = (double*)dst->Mat.ptr + i * _thread_blocking_conf._fmgrH.frag_len * dst->Pitch();

        for (uint32_t j = 0; j < t2D->thread_w; ++j)
        {
            const auto* thread_block = &this->_blocking_confs.ptr[i * t2D->thread_w + j];
            t2D->_async_thread[i * t2D->thread_w + j] =
                decx::cpu::register_task_default(kernel_ptr,
                    src_loc,            (double*)kernel->Mat.ptr,
                    dst_loc,            make_uint2(kernel->Width(), kernel->Height()),
                    thread_block,       src->Pitch(),
                    kernel->Pitch(),    dst->Pitch());

            src_loc += _thread_blocking_conf._fmgrW.frag_len * 4;
            dst_loc += _thread_blocking_conf._fmgrW.frag_len * 4;
        }
    }

    t2D->__sync_all_threads();
}



template <> template <bool _cplxf> void 
decx::dsp::cpu_Filter2D_planner<double>::filter2D_B_64b(decx::_Matrix* src, 
                                                        decx::_Matrix* kernel, 
                                                        decx::_Matrix* dst,
                                                        decx::utils::_thr_2D* t2D)
{
    decx::dsp::CPUK::conv2_B_kernel_64b* _kernel_ptr = NULL;

    if (this->_padding_method == de::extend_label::_EXTEND_CONSTANT_) {
        decx::bp::_extend_constant_b64_2D((double*)src->Mat.ptr,         (double*)this->_ext_src._ptr.ptr, 0,
                                          make_uint4(this->_layout_kernel.width >> 1, this->_layout_kernel.width >> 1, 0, 0), 
                                          this->_layout_src.pitch,      this->_ext_src._dims.x, 
                                          this->_layout_src.width,      this->_layout_src.height, NULL);

        if
#if __cplusplus >= 201703L
            constexpr
#endif
            (_cplxf) {
            _kernel_ptr = decx::dsp::CPUK::conv2_BC_cplxf_kernel;
        }
        else {
            _kernel_ptr = decx::dsp::CPUK::conv2_BC_fp64_kernel;
        }
    }
    else {
        decx::bp::_extend_reflect_b64_2D((double*)src->Mat.ptr,          (double*)this->_ext_src._ptr.ptr,
                                          make_uint4(this->_layout_kernel.width >> 1, this->_layout_kernel.width >> 1, 0, 0), 
                                          this->_layout_src.pitch,      this->_ext_src._dims.x, 
                                          this->_layout_src.width,      this->_layout_src.height, NULL);

        if
#if __cplusplus >= 201703L
            constexpr
#endif
            (_cplxf) {
            _kernel_ptr = decx::dsp::CPUK::conv2_BR_cplxf_kernel;
        }
        else {
            _kernel_ptr = decx::dsp::CPUK::conv2_BR_fp64_kernel;
        }
    }

    const double* src_loc = NULL;
    double* dst_loc = NULL;
    uint32_t _start_row_id = 0;

    for (uint32_t i = 0; i < t2D->thread_h; ++i)
    {
        src_loc = (double*)this->_ext_src._ptr.ptr;
        dst_loc = (double*)dst->Mat.ptr + _start_row_id * dst->Pitch();

        for (uint32_t j = 0; j < t2D->thread_w; ++j)
        {
            const auto* thread_block = &this->_blocking_confs.ptr[i * t2D->thread_w + j];
            t2D->_async_thread[i * t2D->thread_w + j] =
                decx::cpu::register_task_default(_kernel_ptr,
                    src_loc,            (double*)kernel->Mat.ptr,
                    dst_loc,            make_uint2(kernel->Width(), kernel->Height()),
                    thread_block,       this->_ext_src._dims.x,
                    kernel->Pitch(),    dst->Pitch(),
                    _start_row_id,      src->Height());

            src_loc += _thread_blocking_conf._fmgrW.frag_len * 4;
            dst_loc += _thread_blocking_conf._fmgrW.frag_len * 4;
        }
        _start_row_id += _thread_blocking_conf._fmgrH.frag_len;
    }

    t2D->__sync_all_threads();
}



template <> template <bool _cplxf> void
decx::dsp::cpu_Filter2D_planner<double>::run(decx::_Matrix* src, 
                                             decx::_Matrix* kernel, 
                                             decx::_Matrix* dst, 
                                             decx::utils::_thr_2D* t2D)
{
    if (this->_padding_method == de::extend_label::_EXTEND_NONE_) {
        this->filter2D_NB_64b<_cplxf>(src, kernel, dst, t2D);
    }
    else {
        this->filter2D_B_64b<_cplxf>(src, kernel, dst, t2D);
    }
}

template void decx::dsp::cpu_Filter2D_planner<double>::run<true>(decx::_Matrix*,
    decx::_Matrix*, decx::_Matrix*, decx::utils::_thr_2D*);

template void decx::dsp::cpu_Filter2D_planner<double>::run<false>(decx::_Matrix*,
    decx::_Matrix*, decx::_Matrix*, decx::utils::_thr_2D*);
