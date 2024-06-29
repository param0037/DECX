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


#include "cuda_conv2D_fp32_im2col_planner.cuh"


decx::nn::cuda_conv2D_fp32_im2col_planner* decx::nn::_conv2_fp32_planner;


template <> _CRSR_
void decx::nn::cuda_conv2D_im2col_kernel_arrange<float>::init(const decx::_GPU_TensorArray* kernel,
    decx::cuda_stream* S, de::DH* handle)
{
    this->_kernel_layout = &kernel->get_layout();

    this->_eq_kernel_dims_2D = make_uint2(
        decx::utils::align<uint32_t>(decx::utils::align<uint32_t>(kernel->Depth(), 4) * kernel->Width() * kernel->Height(), 4), 
        kernel->TensorNum());
    this->_transp_ker_dims = make_uint2(decx::utils::align<uint32_t>(this->_eq_kernel_dims_2D.y, 32),
                                              kernel->Depth() * kernel->Width() * kernel->Height());

    this->_kernel_tensor_num = kernel->TensorNum();

    if (decx::alloc::_device_malloc(&this->_shrinked_kernel, this->_eq_kernel_dims_2D.x * this->_eq_kernel_dims_2D.y * sizeof(float)) ||
        decx::alloc::_device_malloc(&this->_transposed_kernel, this->_transp_ker_dims.x * this->_transp_ker_dims.y * sizeof(float))) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_CUDA_ALLOCATION,
            DEV_ALLOC_FAIL);
        return;
    }

    this->_kernel_cpy_params = { 0 };
    this->_kernel_cpy_params.kind = cudaMemcpyDeviceToDevice;
    this->_kernel_cpy_params.extent = make_cudaExtent(kernel->Depth() * sizeof(float), kernel->Width(), kernel->Height());

    this->_kernel_cpy_params.srcPtr = make_cudaPitchedPtr(NULL, kernel->get_layout().dpitch * sizeof(float),
        kernel->Depth() * sizeof(float), kernel->get_layout().wpitch);
    this->_kernel_cpy_params.dstPtr = make_cudaPitchedPtr(this->_shrinked_kernel.ptr, kernel->Depth() * sizeof(float),
        kernel->Depth() * sizeof(float), kernel->Width());
}


template <>
void decx::nn::cuda_conv2D_im2col_kernel_arrange<float>::arrange_kernel(const decx::_GPU_TensorArray* kernel,
    decx::cuda_stream* S)
{
    this->_kernel_cpy_params.dstPtr.ptr = this->_shrinked_kernel.ptr;
    for (uint32_t i = 0; i < kernel->TensorNum(); ++i) 
    {
        this->_kernel_cpy_params.srcPtr.ptr = kernel->TensptrArr.ptr[i];
        checkCudaErrors(cudaMemcpy3DAsync(&this->_kernel_cpy_params, S->get_raw_stream_ref()));
        this->_kernel_cpy_params.dstPtr.ptr = (float*)this->_kernel_cpy_params.dstPtr.ptr + _eq_kernel_dims_2D.x;
    }
    
    decx::bp::transpose2D_b4((float2*)this->_shrinked_kernel.ptr, 
                             (float2*)this->_transposed_kernel.ptr, 
                             make_uint2(this->_eq_kernel_dims_2D.y, this->_transp_ker_dims.y),
                             this->_eq_kernel_dims_2D.x, 
                             this->_transp_ker_dims.x, S);
}



decx::nn::cuda_conv2D_fp32_im2col_planner::cuda_conv2D_fp32_im2col_planner()
{
    memset(this, 0, sizeof(decx::nn::cuda_conv2D_fp32_im2col_planner));
}


bool decx::nn::cuda_conv2D_fp32_im2col_planner::changed(const decx::_tensor_layout* src_layout,
                                                        const decx::_GPU_TensorArray* kernel,
                                                        const de::extend_label ext_method, 
                                                        const uint2 strides) const
{
    if (this->_src_layout != NULL) {
        bool unmatched_src = src_layout->width != this->_src_layout->width || src_layout->height != this->_src_layout->height
            || src_layout->depth != this->_src_layout->depth;

        const decx::_tensor_layout* _kernel_layout = this->_kernel_manager._kernel_layout;
        bool unmatched_kernel = kernel->get_layout().width != _kernel_layout->width || _kernel_layout->height != _kernel_layout->height
            || kernel->get_layout().depth != _kernel_layout->depth || kernel->TensorNum() != this->_kernel_manager._kernel_tensor_num;

        bool unmatched_strides = *((uint64_t*)&strides) != *((uint64_t*)&this->_strides);

        bool unmatched_method = ext_method != this->_ext_method;

        return unmatched_src || unmatched_kernel || unmatched_strides || unmatched_method;
    }
    else {
        return true;
    }
}



template <>
void decx::nn::cuda_conv2D_im2col_kernel_arrange<float>::release()
{
    decx::alloc::_device_dealloc(&this->_shrinked_kernel);
    decx::alloc::_device_dealloc(&this->_transposed_kernel);
}


void decx::nn::cuda_conv2D_fp32_im2col_planner::_kernel_launch_config(const uint32_t _proc_idx,
    const uint32_t _proc_h)
{
    decx::nn::cuda_conv2D_im2col_kernel_params* _ptr = this->_params_array.get_ptr(_proc_idx);
    
    _ptr->_proc_H = _proc_h;
    _ptr->_im2col_bufW = this->_I2C_wpitch * _proc_h;

    // Plan for the CUDA kernel configureations
    if (this->_src_layout->dpitch == 4)
    {
        constexpr uint32_t STG_block_dimx = _IM2COL_GET_STG_BLOCKDIM_X_(_IM2COL_D4N_FP32_BLOCK_X_, 4);
        constexpr uint32_t STG_block_dimy = _IM2COL_GET_STG_BLOCKDIM_Y_(_IM2COL_D4N_FP32_BLOCK_Y_);

        _ptr->_block_i2c = dim3(_IM2COL_D4N_FP32_BLOCK_X_, _IM2COL_D4N_FP32_BLOCK_Y_);
        _ptr->_grid_i2c = dim3(decx::utils::ceil<uint32_t>(this->_dst_dims.y, STG_block_dimx),
                               decx::utils::ceil<uint32_t>(_ptr->_proc_H, STG_block_dimy),
                               this->_kernel_manager._kernel_layout->height);

        _ptr->_block_gemm = dim3(_IM2COL_GEMM_FP32_BLOCK_X_, _IM2COL_GEMM_FP32_BLOCK_Y_);
        _ptr->_grid_gemm = dim3(decx::utils::ceil<uint32_t>(this->_dst_dims.y, STG_block_dimx),
                                decx::utils::ceil<uint32_t>(_ptr->_proc_H, _IM2COL_GEMM_FP32_BLOCK_Y_),
                                decx::utils::ceil<uint32_t>(this->_kernel_manager._kernel_tensor_num, 4));
    }
    else if (this->_src_layout->dpitch == 8) {
        constexpr uint32_t STG_block_dimx = _IM2COL_GET_STG_BLOCKDIM_X_(_IM2COL_D4N_FP32_BLOCK_X_, 8);
        constexpr uint32_t STG_block_dimy = _IM2COL_GET_STG_BLOCKDIM_Y_(_IM2COL_D4N_FP32_BLOCK_Y_);

        _ptr->_block_i2c = dim3(_IM2COL_D4N_FP32_BLOCK_X_, _IM2COL_D4N_FP32_BLOCK_Y_);
        _ptr->_grid_i2c = dim3(decx::utils::ceil<uint32_t>(this->_dst_dims.y, STG_block_dimx),
            decx::utils::ceil<uint32_t>(_ptr->_proc_H, STG_block_dimy),
            this->_kernel_manager._kernel_layout->height);

        _ptr->_block_gemm = dim3(_IM2COL_GEMM_FP32_BLOCK_X_, _IM2COL_GEMM_FP32_BLOCK_Y_);
        _ptr->_grid_gemm = dim3(decx::utils::ceil<uint32_t>(this->_dst_dims.y, STG_block_dimx),
            decx::utils::ceil<uint32_t>(_ptr->_proc_H, _IM2COL_GEMM_FP32_BLOCK_Y_),
            decx::utils::ceil<uint32_t>(this->_kernel_manager._kernel_tensor_num, 4));
    }
    else if (this->_src_layout->dpitch == 12) {
        constexpr uint32_t STG_block_dimx = _IM2COL_GET_STG_BLOCKDIM_X_(_IM2COL_D12_FP32_BLOCK_X_, 12);
        constexpr uint32_t STG_block_dimy = _IM2COL_GET_STG_BLOCKDIM_Y_(_IM2COL_D12_FP32_BLOCK_Y_);

        _ptr->_block_i2c = dim3(_IM2COL_D12_FP32_BLOCK_X_, _IM2COL_D12_FP32_BLOCK_Y_);
        _ptr->_grid_i2c = dim3(decx::utils::ceil<uint32_t>(this->_dst_dims.y, STG_block_dimx),
            decx::utils::ceil<uint32_t>(_ptr->_proc_H, STG_block_dimy),
            this->_kernel_manager._kernel_layout->height);

        _ptr->_block_gemm = dim3(_IM2COL_GEMM_FP32_BLOCK_X_, _IM2COL_GEMM_FP32_BLOCK_Y_);
        _ptr->_grid_gemm = dim3(decx::utils::ceil<uint32_t>(this->_dst_dims.y, STG_block_dimx),
            decx::utils::ceil<uint32_t>(_ptr->_proc_H, _IM2COL_GEMM_FP32_BLOCK_Y_),
            decx::utils::ceil<uint32_t>(this->_kernel_manager._kernel_tensor_num, 4));
    }
    else {
        constexpr uint32_t STG_block_dimx = _IM2COL_GET_STG_BLOCKDIM_X_(_IM2COL_D4N_FP32_BLOCK_X_, 16);
        constexpr uint32_t STG_block_dimy = _IM2COL_GET_STG_BLOCKDIM_Y_(_IM2COL_D4N_FP32_BLOCK_Y_);

        _ptr->_block_i2c = dim3(_IM2COL_D4N_FP32_BLOCK_X_, _IM2COL_D4N_FP32_BLOCK_Y_);
        _ptr->_grid_i2c = dim3(decx::utils::ceil<uint32_t>(this->_dst_dims.y, STG_block_dimx),
            decx::utils::ceil<uint32_t>(_ptr->_proc_H, STG_block_dimy),
            this->_kernel_manager._kernel_layout->height);

        _ptr->_block_gemm = dim3(_IM2COL_GEMM_FP32_BLOCK_X_, _IM2COL_GEMM_FP32_BLOCK_Y_);
        _ptr->_grid_gemm = dim3(decx::utils::ceil<uint32_t>(this->_dst_dims.y, STG_block_dimx),
            decx::utils::ceil<uint32_t>(_ptr->_proc_H, _IM2COL_GEMM_FP32_BLOCK_Y_),
            decx::utils::ceil<uint32_t>(this->_kernel_manager._kernel_tensor_num, 4));
    }
}


_CRSR_ void 
decx::nn::cuda_conv2D_fp32_im2col_planner::plan(const decx::_tensor_layout* src_layout, 
                                                const decx::_GPU_TensorArray* kernel,
                                                const de::extend_label ext_method, 
                                                const uint2 strides,
                                                decx::cuda_stream* S, 
                                                de::DH* handle)
{
    this->_strides = strides;
    this->_ext_method = ext_method;

    this->_kernel_manager.init(kernel, S, handle);
    Check_Runtime_Error(handle);
    
    this->_src_layout = src_layout;

    if (this->_ext_method == de::extend_label::_EXTEND_NONE_) {
        this->_dst_dims = make_uint3(this->_kernel_manager._kernel_tensor_num,
                                     (src_layout->width - kernel->Width() + 1) / strides.x,
                                     (src_layout->height - kernel->Height() + 1) / strides.y);
    }
    else if (this->_ext_method == de::extend_label::_EXTEND_CONSTANT_) {
        this->_dst_dims = make_uint3(this->_kernel_manager._kernel_tensor_num,
                                     src_layout->width / strides.x,
                                     src_layout->height / strides.y);
    }
    else {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_ErrorFlag, MEANINGLESS_FLAG);
        return;
    }
    
    this->_I2C_wpitch = decx::utils::align(this->_dst_dims.y, 32);

    const uint32_t I2C_kernel_len = kernel->Width() * kernel->Height() * /*decx::utils::align<uint32_t>(src_layout->depth, 4)*/kernel->Depth();
    const uint64_t I2C_WD_size = this->_I2C_wpitch * I2C_kernel_len * sizeof(float);
    const uint32_t procH = _MAX_IM2COL_TILE_SIZE_ / I2C_WD_size;

    decx::utils::frag_manager _conv_div_info;
    decx::utils::frag_manager_gen_from_fragLen(&_conv_div_info, this->_dst_dims.z, procH);

    this->_im2col_buf_alloc = make_ulong2(this->_I2C_wpitch * _conv_div_info.frag_len,
                                          I2C_kernel_len);
    
    // Allocate space for im2col buffer
    if (decx::alloc::_device_malloc(&this->_im2col_buf, 
        this->_im2col_buf_alloc.x * this->_im2col_buf_alloc.y * sizeof(float), true, S))
    {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_CUDA_ALLOCATION,
            DEV_ALLOC_FAIL);
        return;
    }
    
    this->_wpitchsrc_proc_v1 = this->_src_layout->wpitch;

    // Copy data from src to _ext_src_buf if method == BC
    if (this->_ext_method == de::extend_label::_EXTEND_CONSTANT_) {
        // Allocate buffer for _ext_src_buf
        this->_ext_src_buf._dims = make_uint2(decx::utils::align<uint32_t>(src_layout->width + kernel->Width() - 1, 8),
                                              src_layout->height);

        this->_wpitchsrc_proc_v1 = this->_ext_src_buf._dims.x;

        if (decx::alloc::_device_malloc(&this->_ext_src_buf._ptr, this->_ext_src_buf._dims.x * this->_ext_src_buf._dims.y * src_layout->dpitch * sizeof(float),
            true, S)) {
            decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_CUDA_ALLOCATION, DEV_ALLOC_FAIL);
            return;
        }
    }

    this->_params_array.define_capacity(_conv_div_info.frag_num);
    for (uint32_t i = 0; i < _conv_div_info.frag_num - 1; ++i) {
        this->_params_array.emplace_back();
        this->_kernel_launch_config(i, _conv_div_info.frag_len);
    }
    this->_params_array.emplace_back();
    this->_kernel_launch_config(_conv_div_info.frag_num - 1, _conv_div_info.is_left ? _conv_div_info.frag_left_over : _conv_div_info.frag_len);

}



void decx::nn::cuda_conv2D_fp32_im2col_planner::_cpy_src_ext(decx::_GPU_Tensor* src,
    decx::cuda_stream* S) const
{
    if (this->_ext_method == de::extend_label::_EXTEND_CONSTANT_)
    {
        float4* _cpy_start = this->_ext_src_buf._ptr.ptr +
            (this->_kernel_manager._kernel_layout->width >> 1) * (src->get_layout().dpitch / 4);

        checkCudaErrors(cudaMemcpy2DAsync(_cpy_start,
            this->_ext_src_buf._dims.x * src->get_layout().dpitch * sizeof(float),
            src->Tens.ptr, src->get_layout().dp_x_wp * sizeof(float),
            src->Width() * src->get_layout().dpitch * sizeof(float),
            src->Height(),
            cudaMemcpyDeviceToDevice,
            S->get_raw_stream_ref()));
    }
}


void _CRSR_
decx::nn::cuda_conv2D_fp32_im2col_planner::run_single_frag_NB(const uint32_t _proc_idx,
    decx::cuda_stream* S)
{
    decx::nn::cuda_conv2D_im2col_kernel_params* _ptr = this->_params_array.get_ptr(_proc_idx);
    const decx::_tensor_layout* _kernel_layout = this->_kernel_manager._kernel_layout;

    switch (this->_src_layout->dpitch)
    {
    case 4:
        decx::nn::GPUK::cu_im2col_DP4_NB_fp32 << <_ptr->_grid_i2c, _ptr->_block_i2c, 0, S->get_raw_stream_ref() >> > (
            (float4*)_ptr->_src_loc,         (float4*)this->_im2col_buf.ptr,
            make_uint2(this->_dst_dims.y, _ptr->_proc_H),  
            make_uint3(_kernel_layout->width, _kernel_layout->height, _kernel_layout->depth),
            this->_strides,                 decx::utils::align<uint32_t>(this->_dst_dims.y, 32),
            this->_wpitchsrc_proc_v1, _ptr->_im2col_bufW);
        break;

    case 8:
        decx::nn::GPUK::cu_im2col_DP8_NB_fp32 << <_ptr->_grid_i2c, _ptr->_block_i2c, 0, S->get_raw_stream_ref() >> > (
            (float4*)_ptr->_src_loc, (float2*)this->_im2col_buf.ptr,
            make_uint2(this->_dst_dims.y, _ptr->_proc_H),
            make_uint3(_kernel_layout->width, _kernel_layout->height, _kernel_layout->depth),
            this->_strides, decx::utils::align<uint32_t>(this->_dst_dims.y, 32),
            this->_wpitchsrc_proc_v1, _ptr->_im2col_bufW);
        break;

    case 12:
        decx::nn::GPUK::cu_im2col_DP12_NB_fp32 << <_ptr->_grid_i2c, _ptr->_block_i2c, 0, S->get_raw_stream_ref() >> > (
            (float4*)_ptr->_src_loc, (float2*)this->_im2col_buf.ptr,
            make_uint2(this->_dst_dims.y, _ptr->_proc_H),
            make_uint3(_kernel_layout->width, _kernel_layout->height, _kernel_layout->depth),
            this->_strides, decx::utils::align<uint32_t>(this->_dst_dims.y, 32),
            this->_wpitchsrc_proc_v1, _ptr->_im2col_bufW);
        break;

    case 16:
        decx::nn::GPUK::cu_im2col_DP16_NB_fp32 << <_ptr->_grid_i2c, _ptr->_block_i2c, 0, S->get_raw_stream_ref() >> > (
            (float4*)_ptr->_src_loc, (float*)this->_im2col_buf.ptr,
            make_uint2(this->_dst_dims.y, _ptr->_proc_H),
            make_uint3(_kernel_layout->width, _kernel_layout->height, _kernel_layout->depth),
            this->_strides, decx::utils::align<uint32_t>(this->_dst_dims.y, 32),
            this->_wpitchsrc_proc_v1,       _ptr->_im2col_bufW);
        break;

    default:
        break;
    }

    decx::nn::GPUK::cu_im2col_GEMM_fp32 << <_ptr->_grid_gemm, _ptr->_block_gemm, 0, S->get_raw_stream_ref() >> > (
        (float4*)this->_im2col_buf.ptr,                             (float4*)this->_kernel_manager._transposed_kernel.ptr,        
        (float4*)_ptr->_dst_loc,                                    this->_dst_layout->dpitch,               
        decx::utils::align<uint32_t>(this->_dst_dims.y, 32),        this->_dst_layout->wpitch,               
        _kernel_layout->depth * _kernel_layout->width * _kernel_layout->height,
        make_uint2(this->_dst_dims.y, _ptr->_proc_H));
}



void decx::nn::cuda_conv2D_fp32_im2col_planner::_flush_im2col_buf(decx::cuda_stream* S, const bool _is_top)
{
    const uint32_t _flush_EQdst_rows = this->_kernel_manager._kernel_layout->height / 2;
    const uint32_t _flush_EQi2c_cols = _flush_EQdst_rows * this->_I2C_wpitch;

    if (_is_top) {
        checkCudaErrors(cudaMemset2DAsync(this->_im2col_buf.ptr, 
                                          this->_im2col_buf_alloc.x * sizeof(float),
                                          0, 
                                          _flush_EQi2c_cols * sizeof(float), 
                                          this->_im2col_buf_alloc.y, 
                                          S->get_raw_stream_ref()));
    }
    else {
        checkCudaErrors(cudaMemset2DAsync((float*)this->_im2col_buf.ptr + 
            (this->_params_array.back()->_proc_H - _flush_EQdst_rows) * this->_I2C_wpitch,
                                          this->_params_array.back()->_im2col_bufW * sizeof(float),
                                          0, 
                                          _flush_EQi2c_cols * sizeof(float), 
                                          this->_im2col_buf_alloc.y, 
                                          S->get_raw_stream_ref()));
    }
}



template <bool _boundless_T, bool _boundless_B> void _CRSR_
decx::nn::cuda_conv2D_fp32_im2col_planner::run_single_frag_BC(const uint32_t _proc_idx,
                                                              decx::cuda_stream* S)
{
    decx::nn::cuda_conv2D_im2col_kernel_params* _ptr = this->_params_array.get_ptr(_proc_idx);
    const decx::_tensor_layout* _kernel_layout = this->_kernel_manager._kernel_layout;

    switch (this->_src_layout->dpitch)
    {
    case 4:
        decx::nn::GPUK::cu_im2col_DP4_BC_fp32
            <_boundless_T, _boundless_B> << <_ptr->_grid_i2c, _ptr->_block_i2c, 0, S->get_raw_stream_ref() >> > (
            (float4*)_ptr->_src_loc,    
            (float4*)this->_im2col_buf.ptr,
            make_uint2(this->_dst_dims.y, _ptr->_proc_H),
            make_uint3(_kernel_layout->width, _kernel_layout->height, _kernel_layout->depth),
            this->_strides,                 
            decx::utils::align<uint32_t>(this->_dst_dims.y, 32),
            this->_wpitchsrc_proc_v1,            
            _ptr->_im2col_bufW);
        break;
    
    case 8:
        decx::nn::GPUK::cu_im2col_DP8_BC_fp32
            <_boundless_T, _boundless_B> << <_ptr->_grid_i2c, _ptr->_block_i2c, 0, S->get_raw_stream_ref() >> > (
            (float4*)_ptr->_src_loc,    
            (float2*)this->_im2col_buf.ptr,
            make_uint2(this->_dst_dims.y, _ptr->_proc_H),
            make_uint3(_kernel_layout->width, _kernel_layout->height, _kernel_layout->depth),
            this->_strides,                 
            decx::utils::align<uint32_t>(this->_dst_dims.y, 32),
            this->_wpitchsrc_proc_v1,            
            _ptr->_im2col_bufW);
        break;

    case 12:
        decx::nn::GPUK::cu_im2col_DP12_BC_fp32
            <_boundless_T, _boundless_B> << <_ptr->_grid_i2c, _ptr->_block_i2c, 0, S->get_raw_stream_ref() >> > (
            (float4*)_ptr->_src_loc,    
            (float2*)this->_im2col_buf.ptr,
            make_uint2(this->_dst_dims.y, _ptr->_proc_H),
            make_uint3(_kernel_layout->width, _kernel_layout->height, _kernel_layout->depth),
            this->_strides,
            decx::utils::align<uint32_t>(this->_dst_dims.y, 32),
            this->_wpitchsrc_proc_v1,            
            _ptr->_im2col_bufW);
        break;

    case 16:
        decx::nn::GPUK::cu_im2col_DP16_BC_fp32
            <_boundless_T, _boundless_B> << <_ptr->_grid_i2c, _ptr->_block_i2c, 0, S->get_raw_stream_ref() >> > (
            (float4*)_ptr->_src_loc,    
            (float*)this->_im2col_buf.ptr,
            make_uint2(this->_dst_dims.y, _ptr->_proc_H),
            make_uint3(_kernel_layout->width, _kernel_layout->height, _kernel_layout->depth),
            this->_strides,                 
            decx::utils::align<uint32_t>(this->_dst_dims.y, 32),
            this->_wpitchsrc_proc_v1,            
            _ptr->_im2col_bufW);
        break;

    default:
        break;
    }

    decx::nn::GPUK::cu_im2col_GEMM_fp32 << <_ptr->_grid_gemm, _ptr->_block_gemm, 0, S->get_raw_stream_ref() >> > (
        (float4*)this->_im2col_buf.ptr,                             (float4*)this->_kernel_manager._transposed_kernel.ptr,        
        (float4*)_ptr->_dst_loc,                                    this->_dst_layout->dpitch,               
        decx::utils::align<uint32_t>(this->_dst_dims.y, 32),        this->_dst_layout->wpitch,               
        _kernel_layout->depth * _kernel_layout->width * _kernel_layout->height,
        make_uint2(this->_dst_dims.y, _ptr->_proc_H));


}

template void _CRSR_ decx::nn::cuda_conv2D_fp32_im2col_planner::run_single_frag_BC<false, false>(const uint32_t, decx::cuda_stream*);
template void _CRSR_ decx::nn::cuda_conv2D_fp32_im2col_planner::run_single_frag_BC<true, false>(const uint32_t, decx::cuda_stream*);
template void _CRSR_ decx::nn::cuda_conv2D_fp32_im2col_planner::run_single_frag_BC<false, true>(const uint32_t, decx::cuda_stream*);


void _CRSR_
decx::nn::cuda_conv2D_fp32_im2col_planner::run(decx::_GPU_Tensor* src, decx::_GPU_TensorArray* kernel,
    decx::_GPU_Tensor* dst, decx::cuda_stream* S, de::DH* handle)
{
    // Kernel data arrangement
    this->_kernel_manager.arrange_kernel(kernel, S);
    // End of kernel data arrangement

    // Copy the data from source tensor
    this->_cpy_src_ext(src, S);

    const uint64_t _src_dp_x_wp = this->_ext_method == de::_EXTEND_NONE_ ?
        this->_src_layout->dp_x_wp :
        this->_ext_src_buf._dims.x * src->get_layout().dpitch;

    float* _src_loc = this->_ext_method == de::_EXTEND_NONE_ ?
        (float*)src->Tens.ptr :
        (float*)(this->_ext_src_buf._ptr.ptr) - _src_dp_x_wp * (this->_kernel_manager._kernel_layout->height >> 1);

    float* _dst_loc = (float*)dst->Tens.ptr;

    if (this->_params_array.size() > 1) {
        for (uint32_t i = 0; i < this->_params_array.size(); ++i)
        {
            if (i == 0 || i == this->_params_array.size() - 1) {
                this->_flush_im2col_buf(S, i == 0);
            }
            //checkCudaErrors(cudaMemsetAsync(this->_im2col_buf.ptr, 0, this->_im2col_buf_alloc.x * this->_im2col_buf_alloc.y * sizeof(float), S->get_raw_stream_ref()));

            this->_params_array[i]._src_loc = _src_loc;
            this->_params_array[i]._dst_loc = _dst_loc;

            if (i == 0 && this->_ext_method == de::_EXTEND_CONSTANT_) {
                this->run_single_frag_BC<false, true>(i, S);
            }
            else if (i == this->_params_array.size() - 1 && this->_ext_method == de::_EXTEND_CONSTANT_) {
                this->run_single_frag_BC<true, false>(i, S);
            }
            else {
                this->run_single_frag_NB(i, S);
            }
            _src_loc += _src_dp_x_wp * (this->_params_array[i]._proc_H * this->_strides.y);
            _dst_loc += this->_dst_layout->dp_x_wp * this->_params_array[i]._proc_H;
        }
    }
    else {
        this->_params_array[0]._src_loc = _src_loc;
        this->_params_array[0]._dst_loc = _dst_loc;
        if (this->_ext_method == de::extend_label::_EXTEND_CONSTANT_) {
            this->run_single_frag_BC<false, false>(0, S);
        }
        else {
            this->run_single_frag_NB(0, S);
        }
    }
}


void decx::nn::cuda_conv2D_fp32_im2col_planner::update_dst_layout(const decx::_tensor_layout* dst_layout)
{
    this->_dst_layout = dst_layout;
}



const uint3& decx::nn::cuda_conv2D_fp32_im2col_planner::dst_dims_query() const
{
    return this->_dst_dims;
}


void decx::nn::cuda_conv2D_fp32_im2col_planner::release()
{
    decx::alloc::_device_dealloc(&this->_ext_src_buf._ptr);
    if (this->_ext_method == de::extend_label::_EXTEND_CONSTANT_) {
        decx::alloc::_device_dealloc(&this->_ext_src_buf._ptr);
    }

    this->_params_array.clear();

    this->_kernel_manager.release();
}


void decx::nn::InitCUDAConv2DResource()
{
    decx::nn::_conv2_fp32_planner = NULL;
}
