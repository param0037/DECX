/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#include "cuda_filter2D_planner.cuh"


template <typename _data_type> _CRSR_ void 
decx::dsp::cuda_Filter2D_planner<_data_type>::plan(const decx::_matrix_layout* src_layout, 
                                                   const decx::_matrix_layout* kernel_layout,
                                                   const de::extend_label _method, 
                                                   decx::cuda_stream* S,
                                                   de::DH* handle)
{
    this->_src_layout = src_layout;
    this->_kernel_layout = kernel_layout;

    this->_conv_border_method = _method;

    if (this->_conv_border_method == de::extend_label::_EXTEND_NONE_) {
        this->_dst_dims = make_uint2(this->_src_layout->width - this->_kernel_layout->width + 1, 
                                      this->_src_layout->height - this->_kernel_layout->height + 1);
    }
    else {
        this->_dst_dims = make_uint2(this->_src_layout->width, this->_src_layout->height);

        this->_ext_src._dims = make_uint2(decx::utils::align<uint32_t>(this->_dst_dims.x + this->_kernel_layout->width - 1, 128 / sizeof(_data_type)), 
            this->_dst_dims.y);
        if (decx::alloc::_device_malloc(&this->_ext_src._ptr, this->_ext_src._dims.x * this->_ext_src._dims.y * sizeof(_data_type), true, S)) {
            decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_CUDA_ALLOCATION,
                DEV_ALLOC_FAIL);
            return;
        }
    }

    this->_block = dim3(_CU_FILTER2D_FP32_BLOCK_X_, _CU_FILTER2D_FP32_BLOCK_Y_);
    constexpr uint8_t _proc_vec_len = sizeof(_data_type) < 4 ? (8 / sizeof(_data_type)) : (16 / sizeof(_data_type));
    this->_grid = dim3(decx::utils::ceil<uint32_t>(this->_dst_dims.x, this->_block.x * _proc_vec_len),
        decx::utils::ceil<uint32_t>(this->_dst_dims.y, this->_block.y));
}

template void decx::dsp::cuda_Filter2D_planner<float>::plan(const decx::_matrix_layout*, const decx::_matrix_layout*,
    const de::extend_label, decx::cuda_stream*, de::DH*);

template void decx::dsp::cuda_Filter2D_planner<uint8_t>::plan(const decx::_matrix_layout*, const decx::_matrix_layout*,
    const de::extend_label, decx::cuda_stream*, de::DH*);


template <typename _data_type>
uint2 decx::dsp::cuda_Filter2D_planner<_data_type>::dst_dims_req() const
{
    return this->_dst_dims;
}

template uint2 decx::dsp::cuda_Filter2D_planner<float>::dst_dims_req() const;
template uint2 decx::dsp::cuda_Filter2D_planner<uint8_t>::dst_dims_req() const;

