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
    this->_kernel_cpy_params.dstPtr = make_cudaPitchedPtr(this->_shrinked_kernel.ptr, decx::utils::align<uint32_t>(kernel->Depth(), 4) * sizeof(float),
        kernel->Depth() * sizeof(float), kernel->Width());
}


template <>
void decx::nn::cuda_conv2D_im2col_kernel_arrange<float>::arrange_kernel(const decx::_GPU_TensorArray* kernel,
    decx::cuda_stream* S)
{
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


template <>
void decx::nn::cuda_conv2D_im2col_kernel_arrange<float>::release()
{
    decx::alloc::_device_dealloc(&this->_shrinked_kernel);
    decx::alloc::_device_dealloc(&this->_transposed_kernel);
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

    this->_im2col_buf_dims = make_ulong2(decx::utils::align(this->_dst_dims.y, 32) * this->_dst_dims.z,
        kernel->Width() * kernel->Height() * decx::utils::align<uint32_t>(src_layout->depth, 4));

    // Allocate space for im2col buffer
    if (decx::alloc::_device_malloc(&this->_im2col_buf, 
        this->_im2col_buf_dims.x * this->_im2col_buf_dims.y * sizeof(float), true, S)) 
    {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_CUDA_ALLOCATION,
            DEV_ALLOC_FAIL);
        return;
    }

    // Plan for the CUDA kernel configureations
    if (src_layout->dpitch == 4)
    {
        constexpr uint32_t STG_block_dimx = _IM2COL_GET_STG_BLOCKDIM_X_(_IM2COL_D4_FP32_BLOCK_X_, 4);
        constexpr uint32_t STG_block_dimy = _IM2COL_GET_STG_BLOCKDIM_Y_(_IM2COL_D4_FP32_BLOCK_Y_);

        this->_block_i2c = dim3(_IM2COL_D4_FP32_BLOCK_X_, _IM2COL_D4_FP32_BLOCK_Y_);
        this->_grid_i2c = dim3(decx::utils::ceil<uint32_t>(this->_dst_dims.y, STG_block_dimx),
                               decx::utils::ceil<uint32_t>(this->_dst_dims.z, STG_block_dimy),
                               kernel->Height());

        this->_block_gemm = dim3(_IM2COL_GEMM_FP32_BLOCK_X_, _IM2COL_GEMM_FP32_BLOCK_Y_);
        this->_grid_gemm = dim3(decx::utils::ceil<uint32_t>(this->_dst_dims.y, STG_block_dimx),
                                decx::utils::ceil<uint32_t>(this->_dst_dims.z, _IM2COL_GEMM_FP32_BLOCK_Y_),
                                decx::utils::ceil<uint32_t>(kernel->TensorNum(), 4));
    } 
    else if (src_layout->dpitch == 8) {
        constexpr uint32_t STG_block_dimx = _IM2COL_GET_STG_BLOCKDIM_X_(_IM2COL_D8_FP32_BLOCK_X_, 8);
        constexpr uint32_t STG_block_dimy = _IM2COL_GET_STG_BLOCKDIM_Y_(_IM2COL_D8_FP32_BLOCK_Y_);

        this->_block_i2c = dim3(_IM2COL_D8_FP32_BLOCK_X_, _IM2COL_D8_FP32_BLOCK_Y_);
        this->_grid_i2c = dim3(decx::utils::ceil<uint32_t>(this->_dst_dims.y, STG_block_dimx),
                               decx::utils::ceil<uint32_t>(this->_dst_dims.z, STG_block_dimy),
                               kernel->Height());

        this->_block_gemm = dim3(_IM2COL_GEMM_FP32_BLOCK_X_, _IM2COL_GEMM_FP32_BLOCK_Y_);
        this->_grid_gemm = dim3(decx::utils::ceil<uint32_t>(this->_dst_dims.y, STG_block_dimx),
                                decx::utils::ceil<uint32_t>(this->_dst_dims.z, _IM2COL_GEMM_FP32_BLOCK_Y_),
                                decx::utils::ceil<uint32_t>(kernel->TensorNum(), 4));
    }
    else {
        constexpr uint32_t STG_block_dimx = _IM2COL_GET_STG_BLOCKDIM_X_(_IM2COL_D8_FP32_BLOCK_X_, 16);
        constexpr uint32_t STG_block_dimy = _IM2COL_GET_STG_BLOCKDIM_Y_(_IM2COL_D8_FP32_BLOCK_Y_);

        this->_block_i2c = dim3(_IM2COL_D8_FP32_BLOCK_X_, _IM2COL_D8_FP32_BLOCK_Y_);
        this->_grid_i2c = dim3(decx::utils::ceil<uint32_t>(this->_dst_dims.y, STG_block_dimx),
                               decx::utils::ceil<uint32_t>(this->_dst_dims.z, STG_block_dimy),
                               kernel->Height());

        this->_block_gemm = dim3(_IM2COL_GEMM_FP32_BLOCK_X_, _IM2COL_GEMM_FP32_BLOCK_Y_);
        this->_grid_gemm = dim3(decx::utils::ceil<uint32_t>(this->_dst_dims.y, STG_block_dimx),
                                decx::utils::ceil<uint32_t>(this->_dst_dims.z, _IM2COL_GEMM_FP32_BLOCK_Y_),
                                decx::utils::ceil<uint32_t>(kernel->TensorNum(), 4));
    }

    // Copy data from src to _ext_src_buf if method == BC
    if (this->_ext_method == de::extend_label::_EXTEND_CONSTANT_) {
        // Allocate buffer for _ext_src_buf
        this->_ext_src_buf._dims = make_uint2(decx::utils::align<uint32_t>(src_layout->width + kernel->Width() - 1, 8),
                                              src_layout->height);

        if (decx::alloc::_device_malloc(&this->_ext_src_buf._ptr, this->_ext_src_buf._dims.x * this->_ext_src_buf._dims.y * src_layout->dpitch * sizeof(float),
            true, S)) {
            decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_CUDA_ALLOCATION, DEV_ALLOC_FAIL);
            return;
        }
    }
}



void decx::nn::cuda_conv2D_fp32_im2col_planner::_cpy_src_ext(decx::_GPU_Tensor* src,
    decx::cuda_stream* S) const
{
    if (this->_ext_method == de::extend_label::_EXTEND_CONSTANT_)
    {
        float4* _cpy_start = this->_ext_src_buf._ptr.ptr +
            (this->_kernel_manager._kernel_layout->width >> 1) * (src->get_layout().depth / 4);

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
decx::nn::cuda_conv2D_fp32_im2col_planner::run(decx::_GPU_Tensor* src, decx::_GPU_TensorArray* kernel, 
    decx::_GPU_Tensor* dst, decx::cuda_stream* S, de::DH* handle)
{
    // Kernel data arrangement
    this->_kernel_manager.arrange_kernel(kernel, S);
    // End of kernel data arrangement

    // Copy the data from source tensor
    this->_cpy_src_ext(src, S);

    if (this->_ext_method == de::extend_label::_EXTEND_NONE_)
    {
        switch (src->get_layout().dpitch)
        {
        case 4:
            decx::nn::GPUK::cu_im2col_DP4_NB_fp32 << <this->_grid_i2c, this->_block_i2c, 0, S->get_raw_stream_ref() >> > (
                (float4*)src->Tens.ptr,         (float4*)this->_im2col_buf.ptr,
                make_uint2(this->_dst_dims.y, this->_dst_dims.z),  make_uint2(kernel->Width(), kernel->Height()),
                this->_strides,                 decx::utils::align<uint32_t>(this->_dst_dims.y, 32),
                src->get_layout().wpitch,       this->_im2col_buf_dims.x);
            break;
    
        case 8:
            decx::nn::GPUK::cu_im2col_DP8_NB_fp32 << <this->_grid_i2c, this->_block_i2c, 0, S->get_raw_stream_ref() >> > (
                (float4*)src->Tens.ptr,         (float2*)this->_im2col_buf.ptr,       
                make_uint2(this->_dst_dims.y, this->_dst_dims.z), make_uint2(kernel->Width(), kernel->Height()),
                this->_strides,                 decx::utils::align<uint32_t>(this->_dst_dims.y, 32),
                src->get_layout().wpitch,       this->_im2col_buf_dims.x);
            break;

        case 16:
            decx::nn::GPUK::cu_im2col_DP16_NB_fp32 << <this->_grid_i2c, this->_block_i2c, 0, S->get_raw_stream_ref() >> > (
                (float4*)src->Tens.ptr,         (float*)this->_im2col_buf.ptr,       
                make_uint2(this->_dst_dims.y, this->_dst_dims.z), make_uint2(kernel->Width(), kernel->Height()),
                this->_strides,                 decx::utils::align<uint32_t>(this->_dst_dims.y, 32),
                src->get_layout().wpitch,       this->_im2col_buf_dims.x);
            break;

        default:
            break;
        }
    }
    else {
        switch (src->get_layout().dpitch)
        {
        case 4:
            decx::nn::GPUK::cu_im2col_DP4_BC_fp32 << <this->_grid_i2c, this->_block_i2c, 0, S->get_raw_stream_ref() >> > (
                this->_ext_src_buf._ptr.ptr,    (float4*)this->_im2col_buf.ptr,
                make_uint2(this->_dst_dims.y, this->_dst_dims.z), make_uint2(kernel->Width(), kernel->Height()),
                this->_strides,                 decx::utils::align<uint32_t>(this->_dst_dims.y, 32),
                this->_ext_src_buf._dims.x,     this->_im2col_buf_dims.x);
            break;
    
        case 8:
            decx::nn::GPUK::cu_im2col_DP8_BC_fp32 << <this->_grid_i2c, this->_block_i2c, 0, S->get_raw_stream_ref() >> > (
                this->_ext_src_buf._ptr.ptr,    (float2*)this->_im2col_buf.ptr,
                make_uint2(this->_dst_dims.y, this->_dst_dims.z), make_uint2(kernel->Width(), kernel->Height()),
                this->_strides,                 decx::utils::align<uint32_t>(this->_dst_dims.y, 32),
                this->_ext_src_buf._dims.x,     this->_im2col_buf_dims.x);
            break;

        case 16:
            decx::nn::GPUK::cu_im2col_DP16_BC_fp32 << <this->_grid_i2c, this->_block_i2c, 0, S->get_raw_stream_ref() >> > (
                this->_ext_src_buf._ptr.ptr,    (float*)this->_im2col_buf.ptr,
                make_uint2(this->_dst_dims.y, this->_dst_dims.z), make_uint2(kernel->Width(), kernel->Height()),
                this->_strides,                 decx::utils::align<uint32_t>(this->_dst_dims.y, 32),
                this->_ext_src_buf._dims.x,     this->_im2col_buf_dims.x);
            break;

        default:
            break;
        }
    }

    decx::nn::GPUK::cu_im2col_GEMM_fp32 << <this->_grid_gemm, this->_block_gemm, 0, S->get_raw_stream_ref() >> > (
        (float4*)this->_im2col_buf.ptr,                             (float4*)this->_kernel_manager._transposed_kernel.ptr,        
        (float4*)dst->Tens.ptr,                                     dst->get_layout().dpitch,               
        decx::utils::align<uint32_t>(this->_dst_dims.y, 32),        dst->get_layout().wpitch,               
        kernel->Depth() * kernel->Width() * kernel->Height(),       make_uint2(this->_dst_dims.y, this->_dst_dims.z));
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

    this->_kernel_manager.release();
}
