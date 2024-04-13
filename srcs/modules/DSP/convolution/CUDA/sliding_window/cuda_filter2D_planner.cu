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


template <typename _data_type>
decx::dsp::cuda_Filter2D_planner<_data_type>::cuda_Filter2D_planner()
{
    memset(this, 0, sizeof(decx::dsp::cuda_Filter2D_planner<_data_type>));
}

template decx::dsp::cuda_Filter2D_planner<float>::cuda_Filter2D_planner();
template decx::dsp::cuda_Filter2D_planner<uint8_t>::cuda_Filter2D_planner();


template <typename _data_type> _CRSR_ void 
decx::dsp::cuda_Filter2D_planner<_data_type>::plan(const decx::_matrix_layout* src_layout, 
                                                   const decx::_matrix_layout* kernel_layout,
                                                   const de::extend_label _method, 
                                                   const de::_DATA_TYPES_FLAGS_ output_type,
                                                   decx::cuda_stream* S,
                                                   de::DH* handle)
{
    this->_src_layout = src_layout;
    this->_kernel_layout = kernel_layout;

    this->_conv_border_method = _method;

    this->_output_type = output_type;

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
    const de::extend_label, const de::_DATA_TYPES_FLAGS_, decx::cuda_stream*, de::DH*);

template void decx::dsp::cuda_Filter2D_planner<uint8_t>::plan(const decx::_matrix_layout*, const decx::_matrix_layout*,
    const de::extend_label, const de::_DATA_TYPES_FLAGS_, decx::cuda_stream*, de::DH*);


template <typename _data_type>
bool decx::dsp::cuda_Filter2D_planner<_data_type>::changed(const decx::_matrix_layout* src_layout, const decx::_matrix_layout* kernel_layout,
    const de::extend_label extend_method, const de::_DATA_TYPES_FLAGS_ output_type) const
{
    if (_src_layout == NULL || this->_kernel_layout == NULL) {
        return true;
    }
    else {
        return this->_conv_border_method != extend_method ||
               this->_src_layout->width != src_layout->width ||
               this->_src_layout->height != src_layout->height ||
               this->_kernel_layout->width != kernel_layout->width ||
               this->_kernel_layout->height != kernel_layout->height ||
               this->_output_type != output_type;
    }
}

template bool decx::dsp::cuda_Filter2D_planner<float>::changed(const decx::_matrix_layout*, const decx::_matrix_layout*,
    const de::extend_label, const de::_DATA_TYPES_FLAGS_) const;

template bool decx::dsp::cuda_Filter2D_planner<uint8_t>::changed(const decx::_matrix_layout*, const decx::_matrix_layout*,
    const de::extend_label, const de::_DATA_TYPES_FLAGS_) const;


template <typename _data_type>
uint2 decx::dsp::cuda_Filter2D_planner<_data_type>::dst_dims_req() const
{
    return this->_dst_dims;
}

template uint2 decx::dsp::cuda_Filter2D_planner<float>::dst_dims_req() const;
template uint2 decx::dsp::cuda_Filter2D_planner<uint8_t>::dst_dims_req() const;


template <typename _data_type> bool decx::dsp::cuda_Filter2D_planner<_data_type>::
validate_kerW(const uint32_t kerW)
{
    uint32_t constexpr _proc_vec_byte = sizeof(_data_type) < 4 ? 8 : 16;
    uint32_t constexpr _max_ext_w = _CU_FILTER2D_FP32_BLOCK_X_ * _proc_vec_byte / sizeof(_data_type);
    return kerW <= _max_ext_w * 2 + 1;
}

template bool decx::dsp::cuda_Filter2D_planner<float>::validate_kerW(const uint32_t);
template bool decx::dsp::cuda_Filter2D_planner<uint8_t>::validate_kerW(const uint32_t);
//template bool decx::dsp::cuda_Filter2D_planner<double>::validate_kerW(const uint32_t);


template <typename _data_type>
void decx::dsp::cuda_Filter2D_planner<_data_type>::release(decx::dsp::cuda_Filter2D_planner<_data_type>* _fake_this)
{
    if (_fake_this->_conv_border_method != de::extend_label::_EXTEND_NONE_) {
        decx::alloc::_device_dealloc(&_fake_this->_ext_src._ptr);
    }
}

template void decx::dsp::cuda_Filter2D_planner<float>::release(decx::dsp::cuda_Filter2D_planner<float>*);
template void decx::dsp::cuda_Filter2D_planner<uint8_t>::release(decx::dsp::cuda_Filter2D_planner<uint8_t>*);


template <typename _data_type>
decx::dsp::cuda_Filter2D_planner<_data_type>::~cuda_Filter2D_planner()
{
    this->release(this);
}

template decx::dsp::cuda_Filter2D_planner<float>::~cuda_Filter2D_planner();
template decx::dsp::cuda_Filter2D_planner<uint8_t>::~cuda_Filter2D_planner();

//
//void decx::dsp::Release_CU_Filter2D_Resource()
//{
//    if (decx::dsp::_cuda_filter2D_fp32 != NULL) {
//        decx::dsp::_cuda_filter2D_fp32->release();
//    }
//    if (decx::dsp::_cuda_filter2D_u8 != NULL) {
//        decx::dsp::_cuda_filter2D_u8->release();
//    }
//}
