/*   Author : Wayne Anderson
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

#include "cpu_filter2D_planner.h"


template <>
uint2 decx::dsp::cpu_Filter2D_planner<void>::
query_dst_dims(const decx::_matrix_layout* src_layout,
              const decx::_matrix_layout* kernel_layout,
              const de::extend_label padding)
{
    uint2 res;

    if (padding == de::extend_label::_EXTEND_NONE_) {
        res = make_uint2(src_layout->width - kernel_layout->width + 1,
            src_layout->height - kernel_layout->height + 1);
    }
    else {
        res = make_uint2(src_layout->width, src_layout->height);
    }
    return res;
}


#define _CONV2_BLOCK_H_ 64
#define _CONV2_BLOCK_W_ 1       // This should be small enough to make equal load to each thread as possible

template <typename _data_type> void 
decx::dsp::cpu_Filter2D_planner<_data_type>::plan(const uint32_t concurrency, 
                                                  const decx::_matrix_layout* src_layout, 
                                                  const decx::_matrix_layout* kernel_layout,
                                                  const decx::_matrix_layout* dst_layout, 
                                                  de::DH* handle,
                                                  const de::extend_label padding)
{
    this->_layout_src = *src_layout;
    this->_layout_kernel = *kernel_layout;
    this->_layout_dst = *dst_layout;

    this->_concurrency = concurrency;

    this->_conv_dims_v1 = decx::dsp::cpu_Filter2D_planner<void>::query_dst_dims(src_layout, kernel_layout, padding);

    this->_padding_method = padding;

    decx::utils::thread2D_arrangement_advisor(&this->_thread_dist, this->_concurrency, this->_conv_dims_v1);

    constexpr uint32_t _alignment = 32 / sizeof(_data_type);

    const uint32_t conv_dim_W_v = decx::utils::ceil<uint32_t>(this->_conv_dims_v1.x, _alignment);
    decx::utils::frag_manager_gen_Nx(&this->_thread_blocking_conf._fmgrH, this->_conv_dims_v1.y, this->_thread_dist.y, _CONV2_BLOCK_H_);
    decx::utils::frag_manager_gen_Nx(&this->_thread_blocking_conf._fmgrW, conv_dim_W_v, this->_thread_dist.x, _CONV2_BLOCK_W_);

    if (decx::alloc::_host_virtual_page_malloc(&this->_blocking_confs, this->_concurrency * sizeof(decx::utils::_blocking2D_fmgrs))){
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_ALLOCATION,
            ALLOC_FAIL);
        return;
    }

    if (this->_padding_method != de::extend_label::_EXTEND_NONE_) 
    {
        const uint32_t conv_W = this->_layout_src.width + this->_layout_kernel.width - 1;
        this->_ext_src._dims = make_uint2(decx::utils::align<uint32_t>(conv_W, _alignment), 
                                          this->_layout_src.height);

        if (decx::alloc::_host_virtual_page_malloc(&this->_ext_src._ptr, this->_ext_src._dims.x * this->_ext_src._dims.y * sizeof(_data_type))) {
            decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_ALLOCATION,
                ALLOC_FAIL);
            return;
        }
    }

    for (uint32_t i = 0; i < this->_thread_dist.y; ++i)
    {
        uint2 thread_proc_dim_v = make_uint2(this->_thread_blocking_conf._fmgrW.frag_len,
            i < this->_thread_dist.y - 1 ? this->_thread_blocking_conf._fmgrH.frag_len :
                                     this->_thread_blocking_conf._fmgrH.last_frag_len);

        for (uint32_t j = 0; j < _thread_dist.x - 1; ++j){
            auto* thread_block = &this->_blocking_confs.ptr[i * _thread_dist.x + j];
            decx::utils::frag_manager_gen_from_fragLen(&thread_block->_fmgrH, thread_proc_dim_v.y, _CONV2_BLOCK_H_);
            decx::utils::frag_manager_gen_from_fragLen(&thread_block->_fmgrW, thread_proc_dim_v.x, _CONV2_BLOCK_W_);
        }
        thread_proc_dim_v.x = _thread_blocking_conf._fmgrW.last_frag_len;
        auto* thread_block = &this->_blocking_confs.ptr[(i + 1) * _thread_dist.x - 1];

        decx::utils::frag_manager_gen_from_fragLen(&thread_block->_fmgrH, thread_proc_dim_v.y, _CONV2_BLOCK_H_);
        decx::utils::frag_manager_gen_from_fragLen(&thread_block->_fmgrW, thread_proc_dim_v.x, _CONV2_BLOCK_W_);
    }
}

template void decx::dsp::cpu_Filter2D_planner<float>::plan(const uint32_t, 
const decx::_matrix_layout*, const decx::_matrix_layout*, const decx::_matrix_layout*, de::DH*, de::extend_label);

template void decx::dsp::cpu_Filter2D_planner<double>::plan(const uint32_t,
    const decx::_matrix_layout*, const decx::_matrix_layout*, const decx::_matrix_layout*, de::DH*, de::extend_label);


template <typename _data_type>
uint2 decx::dsp::cpu_Filter2D_planner<_data_type>::get_thread_dist() const {
    return this->_thread_dist;
}

template uint2 decx::dsp::cpu_Filter2D_planner<float>::get_thread_dist() const;
template uint2 decx::dsp::cpu_Filter2D_planner<double>::get_thread_dist() const;


template <typename _data_type>
bool decx::dsp::cpu_Filter2D_planner<_data_type>::
changed(const uint32_t      conc, 
        decx::_Matrix*      src, 
        decx::_Matrix*      kernel, 
        decx::_Matrix*      dst,
        de::extend_label    padding_method) const
{
    uint32_t src_dims_matched = src->Width() ^ this->_layout_src.width;
    src_dims_matched |= (src->Height() ^ this->_layout_src.height);

    uint32_t ker_dims_matched = kernel->Width() ^ this->_layout_kernel.width;
    ker_dims_matched |= (kernel->Height() ^ this->_layout_kernel.height);

    uint32_t dst_dims_matched = dst->Width() ^ this->_layout_dst.width;
    ker_dims_matched |=  (dst->Height() ^ this->_layout_dst.height);

    uint32_t padding_lable_matched = (int32_t)this->_padding_method ^ (int32_t)padding_method;

    return src_dims_matched | ker_dims_matched | padding_lable_matched |
        (this->_concurrency ^ conc);
}

template bool decx::dsp::cpu_Filter2D_planner<float>::changed(const uint32_t, decx::_Matrix* src, decx::_Matrix* kernel, decx::_Matrix* dst,
    de::extend_label padding_method) const;

template bool decx::dsp::cpu_Filter2D_planner<double>::changed(const uint32_t, decx::_Matrix* src, decx::_Matrix* kernel, decx::_Matrix* dst,
    de::extend_label padding_method) const;


template <typename _data_type>
void decx::dsp::cpu_Filter2D_planner<_data_type>::release(decx::dsp::cpu_Filter2D_planner<_data_type>* _fake_this)
{
    decx::alloc::_host_virtual_page_dealloc(&_fake_this->_blocking_confs);
}

template void decx::dsp::cpu_Filter2D_planner<float>::release(decx::dsp::cpu_Filter2D_planner<float>*);
template void decx::dsp::cpu_Filter2D_planner<double>::release(decx::dsp::cpu_Filter2D_planner<double>*);
