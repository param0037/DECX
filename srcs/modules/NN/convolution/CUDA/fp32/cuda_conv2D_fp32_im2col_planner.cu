/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#include "cuda_conv2D_fp32_im2col_planner.cuh"


_CRSR_ void 
decx::nn::cuda_conv2D_fp32_im2col_planner::plan(const decx::_tensor_layout* src_layout, 
                                                const decx::_GPU_TensorArray* kernel,
                                                const decx::_tensor_layout* dst_layout,
                                                const decx::bp::extend_label ext_method, 
                                                const uint2 strides,
                                                decx::cuda_stream* S, 
                                                de::DH* handle)
{
    this->_strides = strides;
    this->_ext_method = ext_method;

    this->_src_layout = src_layout;
    this->_kernel_layout = &kernel->get_layout();
    this->_dst_layout = dst_layout;
    this->_kernel_tensor_num = kernel->TensorNum();

    const ulong2 im2col_buf_dims = make_ulong2(decx::utils::align(dst_layout->width, 32) * dst_layout->height,
        kernel->Width() * kernel->Height() * decx::utils::align<uint32_t>(src_layout->depth, 4));

    // Allocate space for im2col buffer
    if (decx::alloc::_device_malloc(&this->_im2col_buf, im2col_buf_dims.x * im2col_buf_dims.y * sizeof(float), true, S)) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_CUDA_ALLOCATION,
            DEV_ALLOC_FAIL);
        return;
    }


}