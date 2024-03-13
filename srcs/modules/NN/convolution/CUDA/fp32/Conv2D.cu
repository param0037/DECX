/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#include "im2col_fp32.cuh"
#include "im2col_GEMM_fp32.cuh"
#include "../../../../classes/GPU_Tensor.h"
#include "../../../../classes/GPU_TensorArray.h"
#include "../../../../BLAS/basic_process/transpose/CUDA/transpose_kernels.cuh"


namespace de
{
    namespace nn {
        namespace cuda
        {
            _DECX_API_ de::DH Conv2D(de::GPU_Tensor& src, de::GPU_TensorArray& kernel, de::GPU_Tensor& dst,
                const de::Point2D strides = { 1, 1 });
        }
    }
}


namespace decx
{
    namespace nn {
        _CRSR_ static void conv2D_im2col_fp32(decx::_GPU_Tensor* src, decx::_GPU_TensorArray* krenel, decx::_GPU_Tensor* dst, 
            const uint2 strides, de::DH* handle);


        _CRSR_ static void conv2D_im2col_BC_fp32(decx::_GPU_Tensor* src, decx::_GPU_TensorArray* krenel, decx::_GPU_Tensor* dst,
            const uint2 strides, de::DH* handle);
    }
}


_CRSR_ static void decx::nn::conv2D_im2col_fp32(decx::_GPU_Tensor* src, decx::_GPU_TensorArray* kernel, decx::_GPU_Tensor* dst, 
    const uint2 strides, de::DH* handle)
{
    const uint2 dst_dims = make_uint2((src->Width() - kernel->Width() + 1) / strides.x, 
                                      (src->Height() - kernel->Height() + 1) / strides.y);
    
    const ulong2 im2col_buf_dims = make_ulong2(decx::utils::align(dst_dims.x, 32) * dst_dims.y,
        kernel->Width() * kernel->Height() * decx::utils::align<uint32_t>(src->Depth(), 4));
    
    decx::cuda_stream* S = NULL;
    decx::cuda_event* E = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (S == NULL) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_CUDA_STREAM,
            CUDA_STREAM_ACCESS_FAIL);
        return;
    }
    if (E == NULL) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_CUDA_EVENT,
            CUDA_EVENT_ACCESS_FAIL);
        return;
    }
    
    decx::PtrInfo<void> _im2col_buf;
    if (decx::alloc::_device_malloc(&_im2col_buf, im2col_buf_dims.x * im2col_buf_dims.y * sizeof(float), true, S)) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_CUDA_ALLOCATION,
            DEV_ALLOC_FAIL);
        return;
    }

    // Kernel data arrangement
    decx::PtrInfo<void> _shrinked_kernel, _transposed_kernel;
    const uint2 _eq_kernel_dims_2D = make_uint2(
        decx::utils::align<uint32_t>(decx::utils::align<uint32_t>(kernel->Depth(), 4) * kernel->Width() * kernel->Height(), 4), 
        kernel->TensorNum());
    const uint2 _transp_ker_dims = make_uint2(decx::utils::align<uint32_t>(_eq_kernel_dims_2D.y, 32),
                                              kernel->Depth() * kernel->Width() * kernel->Height());
    if (decx::alloc::_device_malloc(&_shrinked_kernel, _eq_kernel_dims_2D.x * _eq_kernel_dims_2D.y * sizeof(float)) ||
        decx::alloc::_device_malloc(&_transposed_kernel, _transp_ker_dims.x * _transp_ker_dims.y * sizeof(float))) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_CUDA_ALLOCATION,
            DEV_ALLOC_FAIL);
        return;
    }

    cudaMemcpy3DParms params = { 0 };
    params.kind = cudaMemcpyDeviceToDevice;
    params.extent = make_cudaExtent(kernel->Depth() * sizeof(float), kernel->Width(), kernel->Height());

    params.srcPtr = make_cudaPitchedPtr(kernel->TensptrArr.ptr[0], kernel->get_layout().dpitch * sizeof(float),
                                        kernel->Depth() * sizeof(float), kernel->get_layout().wpitch);
    params.dstPtr = make_cudaPitchedPtr(_shrinked_kernel.ptr, decx::utils::align<uint32_t>(kernel->Depth(), 4) * sizeof(float),
                                        kernel->Depth() * sizeof(float), kernel->Width());
    for (uint32_t i = 0; i < kernel->TensorNum(); ++i) 
    {
        params.srcPtr.ptr = kernel->TensptrArr.ptr[i];
        checkCudaErrors(cudaMemcpy3DAsync(&params, S->get_raw_stream_ref()));
        params.dstPtr.ptr = (float*)params.dstPtr.ptr + _eq_kernel_dims_2D.x;
    }
    
    decx::bp::transpose2D_b4((float2*)_shrinked_kernel.ptr, (float2*)_transposed_kernel.ptr, make_uint2(_eq_kernel_dims_2D.y, _transp_ker_dims.y),
        _eq_kernel_dims_2D.x, _transp_ker_dims.x, S);
    // End of kernel data arrangement


    dim3 block_1, grid_1;

    if (src->get_layout().dpitch == 4)
    {
        constexpr uint32_t STG_block_dimx = _IM2COL_GET_STG_BLOCKDIM_X_(_IM2COL_D4_FP32_BLOCK_X_, 4);
        constexpr uint32_t STG_block_dimy = _IM2COL_GET_STG_BLOCKDIM_Y_(_IM2COL_D4_FP32_BLOCK_Y_);

        dim3 block(_IM2COL_D4_FP32_BLOCK_X_, _IM2COL_D4_FP32_BLOCK_Y_);
        dim3 grid(decx::utils::ceil<uint32_t>(dst_dims.x, STG_block_dimx),
                  decx::utils::ceil<uint32_t>(dst_dims.y, STG_block_dimy),
                  kernel->Height());

        decx::nn::GPUK::cu_im2col_DP4_NB_fp32 << <grid, block, 0, S->get_raw_stream_ref() >> > ((float4*)src->Tens.ptr,
            (float4*)_im2col_buf.ptr,
            dst_dims,
            make_uint2(kernel->Width(), kernel->Height()),
            strides,
            decx::utils::align<uint32_t>(dst_dims.x, 32),
            src->get_layout().wpitch,
            im2col_buf_dims.x);

        block_1 = dim3(_IM2COL_GEMM_FP32_BLOCK_X_, _IM2COL_GEMM_FP32_BLOCK_Y_);
        grid_1 = dim3(decx::utils::ceil<uint32_t>(dst_dims.x, STG_block_dimx),
                    decx::utils::ceil<uint32_t>(dst_dims.y, _IM2COL_GEMM_FP32_BLOCK_Y_),
                    decx::utils::ceil<uint32_t>(kernel->Depth(), 4));
    }
    else {
        constexpr uint32_t STG_block_dimx = _IM2COL_GET_STG_BLOCKDIM_X_(_IM2COL_D8_FP32_BLOCK_X_, 8);
        constexpr uint32_t STG_block_dimy = _IM2COL_GET_STG_BLOCKDIM_Y_(_IM2COL_D8_FP32_BLOCK_Y_);

        dim3 block(_IM2COL_D8_FP32_BLOCK_X_, _IM2COL_D8_FP32_BLOCK_Y_);
        dim3 grid(decx::utils::ceil<uint32_t>(dst_dims.x, STG_block_dimx),
                  decx::utils::ceil<uint32_t>(dst_dims.y, STG_block_dimy),
                  kernel->Height());

        decx::nn::GPUK::cu_im2col_DP8_NB_fp32 << <grid, block, 0, S->get_raw_stream_ref() >> > ((float4*)src->Tens.ptr,
            (float2*)_im2col_buf.ptr,
            dst_dims,
            make_uint2(kernel->Width(), kernel->Height()),
            strides,
            decx::utils::align<uint32_t>(dst_dims.x, 32),
            src->get_layout().wpitch,
            im2col_buf_dims.x);

        block_1 = dim3(_IM2COL_GEMM_FP32_BLOCK_X_, _IM2COL_GEMM_FP32_BLOCK_Y_);
        grid_1 = dim3(decx::utils::ceil<uint32_t>(dst_dims.x, STG_block_dimx),
            decx::utils::ceil<uint32_t>(dst_dims.y, _IM2COL_GEMM_FP32_BLOCK_Y_),
            decx::utils::ceil<uint32_t>(kernel->TensorNum(), 4));
    }
    
    decx::nn::GPUK::cu_im2col_GEMM_fp32 << <grid_1, block_1, 0, S->get_raw_stream_ref() >> > ((float4*)_im2col_buf.ptr,
        (float4*)_transposed_kernel.ptr, 
        (float4*)dst->Tens.ptr,
        dst->get_layout().dpitch,
        decx::utils::align<uint32_t>(dst_dims.x, 32),
        dst->get_layout().wpitch,
        kernel->Depth() * kernel->Width() * kernel->Height(),
        dst_dims);

    E->event_record(S);
    E->synchronize();

    E->detach();
    S->detach();

    decx::alloc::_device_dealloc(&_im2col_buf);
    decx::alloc::_device_dealloc(&_shrinked_kernel);
    decx::alloc::_device_dealloc(&_transposed_kernel);
}



_CRSR_ static void decx::nn::conv2D_im2col_BC_fp32(decx::_GPU_Tensor* src, decx::_GPU_TensorArray* kernel, decx::_GPU_Tensor* dst, 
    const uint2 strides, de::DH* handle)
{
    const uint2 dst_dims = make_uint2((src->Width()) / strides.x, 
                                      (src->Height()) / strides.y);
    
    const ulong2 im2col_buf_dims = make_ulong2(decx::utils::align(dst_dims.x, 32) * dst_dims.y,
        kernel->Width() * kernel->Height() * decx::utils::align<uint32_t>(src->Depth(), 4));
    
    decx::cuda_stream* S = NULL;
    decx::cuda_event* E = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (S == NULL) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_CUDA_STREAM,
            CUDA_STREAM_ACCESS_FAIL);
        return;
    }
    if (E == NULL) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_CUDA_EVENT,
            CUDA_EVENT_ACCESS_FAIL);
        return;
    }
    
    const uint2 _ext_src_dims = make_uint2(decx::utils::align<uint32_t>(src->Width() + kernel->Width() - 1, 8),
                                           src->Height());
    decx::PtrInfo<float4> _ext_src_buf;
    if (decx::alloc::_device_malloc(&_ext_src_buf, _ext_src_dims.x * _ext_src_dims.y * src->get_layout().dpitch * sizeof(float), 
        true, S)) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_CUDA_ALLOCATION,
            DEV_ALLOC_FAIL);
        return;
    }
    float4* _cpy_start = _ext_src_buf.ptr + (kernel->Width() >> 1) * (src->get_layout().depth / 4);
    checkCudaErrors(cudaMemcpy2DAsync(_cpy_start, _ext_src_dims.x * src->get_layout().dpitch * sizeof(float),
        src->Tens.ptr, src->get_layout().dp_x_wp * sizeof(float),
        src->Width() * src->get_layout().dpitch * sizeof(float),
        src->Height(),
        cudaMemcpyDeviceToDevice,
        S->get_raw_stream_ref()));

    decx::PtrInfo<void> _im2col_buf;
    if (decx::alloc::_device_malloc(&_im2col_buf, im2col_buf_dims.x * im2col_buf_dims.y * sizeof(float), true, S)) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_CUDA_ALLOCATION,
            DEV_ALLOC_FAIL);
        return;
    }

    // Kernel data arrangement
    decx::PtrInfo<void> _shrinked_kernel, _transposed_kernel;
    const uint2 _eq_kernel_dims_2D = make_uint2(
        decx::utils::align<uint32_t>(decx::utils::align<uint32_t>(kernel->Depth(), 4) * kernel->Width() * kernel->Height(), 4), 
        kernel->TensorNum());
    const uint2 _transp_ker_dims = make_uint2(decx::utils::align<uint32_t>(_eq_kernel_dims_2D.y, 32),
                                              kernel->Depth() * kernel->Width() * kernel->Height());
    if (decx::alloc::_device_malloc(&_shrinked_kernel, _eq_kernel_dims_2D.x * _eq_kernel_dims_2D.y * sizeof(float)) ||
        decx::alloc::_device_malloc(&_transposed_kernel, _transp_ker_dims.x * _transp_ker_dims.y * sizeof(float))) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_CUDA_ALLOCATION,
            DEV_ALLOC_FAIL);
        return;
    }

    cudaMemcpy3DParms params = { 0 };
    params.kind = cudaMemcpyDeviceToDevice;
    params.extent = make_cudaExtent(kernel->Depth() * sizeof(float), kernel->Width(), kernel->Height());

    params.srcPtr = make_cudaPitchedPtr(kernel->TensptrArr.ptr[0], kernel->get_layout().dpitch * sizeof(float),
                                        kernel->Depth() * sizeof(float), kernel->get_layout().wpitch);
    params.dstPtr = make_cudaPitchedPtr(_shrinked_kernel.ptr, decx::utils::align<uint32_t>(kernel->Depth(), 4) * sizeof(float),
                                        kernel->Depth() * sizeof(float), kernel->Width());
    for (uint32_t i = 0; i < kernel->TensorNum(); ++i) 
    {
        params.srcPtr.ptr = kernel->TensptrArr.ptr[i];
        checkCudaErrors(cudaMemcpy3DAsync(&params, S->get_raw_stream_ref()));
        params.dstPtr.ptr = (float*)params.dstPtr.ptr + _eq_kernel_dims_2D.x;
    }
    
    decx::bp::transpose2D_b4((float2*)_shrinked_kernel.ptr, (float2*)_transposed_kernel.ptr, make_uint2(_eq_kernel_dims_2D.y, _transp_ker_dims.y),
        _eq_kernel_dims_2D.x, _transp_ker_dims.x, S);
    // End of kernel data arrangement

    dim3 block_1, grid_1;

    if (src->get_layout().dpitch == 4)
    {
        constexpr uint32_t STG_block_dimx = _IM2COL_GET_STG_BLOCKDIM_X_(_IM2COL_D4_FP32_BLOCK_X_, 4);
        constexpr uint32_t STG_block_dimy = _IM2COL_GET_STG_BLOCKDIM_Y_(_IM2COL_D4_FP32_BLOCK_Y_);

        dim3 block(_IM2COL_D4_FP32_BLOCK_X_, _IM2COL_D4_FP32_BLOCK_Y_);
        dim3 grid(decx::utils::ceil<uint32_t>(dst_dims.x, STG_block_dimx),
                  decx::utils::ceil<uint32_t>(dst_dims.y, STG_block_dimy),
                  kernel->Height());

        decx::nn::GPUK::cu_im2col_DP4_BC_fp32 << <grid, block, 0, S->get_raw_stream_ref() >> > (_ext_src_buf.ptr,
            (float4*)_im2col_buf.ptr,
            dst_dims,
            make_uint2(kernel->Width(), kernel->Height()),
            strides,
            decx::utils::align<uint32_t>(dst_dims.x, 32),
            _ext_src_dims.x,
            im2col_buf_dims.x);

        block_1 = dim3(_IM2COL_GEMM_FP32_BLOCK_X_, _IM2COL_GEMM_FP32_BLOCK_Y_);
        grid_1 = dim3(decx::utils::ceil<uint32_t>(dst_dims.x, STG_block_dimx),
                    decx::utils::ceil<uint32_t>(dst_dims.y, _IM2COL_GEMM_FP32_BLOCK_Y_),
                    decx::utils::ceil<uint32_t>(kernel->Depth(), 4));
    }
    else {
        constexpr uint32_t STG_block_dimx = _IM2COL_GET_STG_BLOCKDIM_X_(_IM2COL_D8_FP32_BLOCK_X_, 8);
        constexpr uint32_t STG_block_dimy = _IM2COL_GET_STG_BLOCKDIM_Y_(_IM2COL_D8_FP32_BLOCK_Y_);

        dim3 block(_IM2COL_D8_FP32_BLOCK_X_, _IM2COL_D8_FP32_BLOCK_Y_);
        dim3 grid(decx::utils::ceil<uint32_t>(dst_dims.x, STG_block_dimx),
                  decx::utils::ceil<uint32_t>(dst_dims.y, STG_block_dimy),
                  kernel->Height());

        decx::nn::GPUK::cu_im2col_DP8_BC_fp32 << <grid, block, 0, S->get_raw_stream_ref() >> > (_ext_src_buf.ptr,
            (float2*)_im2col_buf.ptr,
            dst_dims,
            make_uint2(kernel->Width(), kernel->Height()),
            strides,
            decx::utils::align<uint32_t>(dst_dims.x, 32),
            _ext_src_dims.x,
            im2col_buf_dims.x);

        block_1 = dim3(_IM2COL_GEMM_FP32_BLOCK_X_, _IM2COL_GEMM_FP32_BLOCK_Y_);
        grid_1 = dim3(decx::utils::ceil<uint32_t>(dst_dims.x, STG_block_dimx),
            decx::utils::ceil<uint32_t>(dst_dims.y, _IM2COL_GEMM_FP32_BLOCK_Y_),
            decx::utils::ceil<uint32_t>(kernel->TensorNum(), 4));
    }
    
    decx::nn::GPUK::cu_im2col_GEMM_fp32 << <grid_1, block_1, 0, S->get_raw_stream_ref() >> > ((float4*)_im2col_buf.ptr,
        (float4*)_transposed_kernel.ptr, 
        (float4*)dst->Tens.ptr,
        dst->get_layout().dpitch,
        decx::utils::align<uint32_t>(dst_dims.x, 32),
        dst->get_layout().wpitch,
        kernel->Depth() * kernel->Width() * kernel->Height(),
        dst_dims);

    E->event_record(S);
    E->synchronize();

    E->detach();
    S->detach();

    decx::alloc::_device_dealloc(&_ext_src_buf);
    decx::alloc::_device_dealloc(&_im2col_buf);
    decx::alloc::_device_dealloc(&_shrinked_kernel);
    decx::alloc::_device_dealloc(&_transposed_kernel);
}


_DECX_API_ void decx_cudaMemcpy_D2H(de::GPU_Tensor& src, de::Tensor& dst, const uint64_t size)
{
    decx::_GPU_Tensor* _src = dynamic_cast<decx::_GPU_Tensor*>(&src);
    decx::_Tensor* _dst = dynamic_cast<decx::_Tensor*>(&dst);

    checkCudaErrors(cudaMemcpy(_dst->Tens.ptr, _src->Tens.ptr, size, cudaMemcpyDeviceToHost));
}


_DECX_API_ void decx_cudaMemcpy_H2D(de::GPU_Tensor& src, de::Tensor& dst, const uint64_t size)
{
    decx::_GPU_Tensor* _src = dynamic_cast<decx::_GPU_Tensor*>(&src);
    decx::_Tensor* _dst = dynamic_cast<decx::_Tensor*>(&dst);

    checkCudaErrors(cudaMemcpy(_src->Tens.ptr, _dst->Tens.ptr, size, cudaMemcpyHostToDevice));
}


_DECX_API_ de::DH de::nn::cuda::Conv2D(de::GPU_Tensor& src, de::GPU_TensorArray& kernel, de::GPU_Tensor& dst,
    const de::Point2D strides)
{
    de::DH handle;

    decx::_GPU_Tensor* _src = dynamic_cast<decx::_GPU_Tensor*>(&src);
    decx::_GPU_TensorArray* _kernel = dynamic_cast<decx::_GPU_TensorArray*>(&kernel);
    decx::_GPU_Tensor* _dst = dynamic_cast<decx::_GPU_Tensor*>(&dst);
    
    decx::nn::conv2D_im2col_BC_fp32(_src, _kernel, _dst, make_uint2(strides.x, strides.y), &handle);

    return handle;
}