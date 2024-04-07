/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#include "../cuda_Filter2D_planner.cuh"
#include "../filter2D_kernel.cuh"


template <> void
decx::dsp::cuda_Filter2D_planner<float>::run(decx::_GPU_Matrix* src, decx::_GPU_Matrix* kernel,
    decx::_GPU_Matrix* dst, decx::cuda_stream* S, de::DH* handle)
{
    if (this->_conv_border_method != de::extend_label::_EXTEND_NONE_) 
    {
        checkCudaErrors(cudaMemcpy2DAsync((float*)this->_ext_src._ptr.ptr + (this->_kernel_layout->width >> 1),
            _ext_src._dims.x * sizeof(float),
            src->Mat.ptr,
            this->_src_layout->pitch * sizeof(float),
            this->_src_layout->width * sizeof(float),
            this->_src_layout->height,
            cudaMemcpyDeviceToDevice,
            S->get_raw_stream_ref()));
        
        decx::dsp::GPUK::cu_filter2D_BC_fp32<32> << <this->_grid, this->_block, 0, S->get_raw_stream_ref() >> > (
            (float4*)_ext_src._ptr.ptr,
            (float*)kernel->Mat.ptr,
            (float4*)dst->Mat.ptr,
            this->_ext_src._dims.x / 4,
            dst->get_layout().pitch / 4,
            make_uint3(kernel->Width(), kernel->Height(), kernel->get_layout().pitch),
            this->_dst_dims);
    }
    else
    {
        decx::dsp::GPUK::cu_filter2D_NB_fp32<32> << <this->_grid, this->_block, 0, S->get_raw_stream_ref() >> > (
            (float4*)src->Mat.ptr,
            (float*)kernel->Mat.ptr,
            (float4*)dst->Mat.ptr,
            this->_src_layout->pitch / 4,
            dst->get_layout().pitch / 4,
            make_uint3(kernel->Width(), kernel->Height(), kernel->get_layout().pitch),
            this->_dst_dims);
    }
}