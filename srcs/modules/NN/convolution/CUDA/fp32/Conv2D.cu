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
#include "../../../../classes/GPU_Tensor.h"
#include "../../../../classes/GPU_TensorArray.h"


namespace de
{
    namespace nn {
        namespace cuda
        {
            _DECX_API_ de::DH Conv2D(de::GPU_Tensor& src, de::GPU_TensorArray& kernel, de::GPU_Tensor& dst);
        }
    }
}


namespace decx
{
    namespace nn {
        _CRSR_
        static void conv2D_im2col_fp32(decx::_GPU_Tensor* src, decx::_GPU_TensorArray* krenel, decx::_GPU_Tensor* dst, de::DH* handle);
    }
}


_CRSR_
static void decx::nn::conv2D_im2col_fp32(decx::_GPU_Tensor* src, decx::_GPU_TensorArray* kernel, decx::_GPU_Tensor* dst, de::DH* handle)
{
    const ulong2 im2col_buf_dims = make_ulong2(dst->get_layout().wpitch * dst->Height(),
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
    
    decx::PtrInfo<void> im2col_buf;
    if (decx::alloc::_device_malloc(&im2col_buf, im2col_buf_dims.x * im2col_buf_dims.y * sizeof(float), true, S)) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_CUDA_ALLOCATION,
            DEV_ALLOC_FAIL);
        return;
    }
    
    constexpr uint32_t STG_block_dimx = _IM2COL_GET_THREAD_PER_ROW_(4);
    constexpr uint32_t STG_block_dimy = _IM2COL_GET_STG_BLOCKDIM_Y_(STG_block_dimx);

    dim3 block(_IM2COL_FP32_BLOCK_X_, _IM2COL_FP32_BLOCK_Y_);
    dim3 grid(decx::utils::ceil<uint32_t>(dst->Width(), STG_block_dimx * sizeof(float4) / sizeof(float)),
              decx::utils::ceil<uint32_t>(dst->Height(), STG_block_dimy),
              kernel->Height());

    decx::nn::GPUK::cu_im2col_NB_fp32_divKH << <grid, block, 0, S->get_raw_stream_ref() >> > ((float4*)src->Tens.ptr,
        (float4*)im2col_buf.ptr,
        make_uint2(dst->Width(), dst->Height()),
        make_uint2(kernel->Width(), /*kernel->Height()*/1),
        src->get_layout().dpitch,
        src->get_layout().wpitch,
        im2col_buf_dims.x);
    
    E->event_record(S);
    E->synchronize();

    E->detach();
    S->detach();
}


_DECX_API_ de::DH de::nn::cuda::Conv2D(de::GPU_Tensor& src, de::GPU_TensorArray& kernel, de::GPU_Tensor& dst)
{
    de::DH handle;

    decx::_GPU_Tensor* _src = dynamic_cast<decx::_GPU_Tensor*>(&src);
    decx::_GPU_TensorArray* _kernel = dynamic_cast<decx::_GPU_TensorArray*>(&kernel);
    decx::_GPU_Tensor* _dst = dynamic_cast<decx::_GPU_Tensor*>(&dst);
    
    decx::nn::conv2D_im2col_fp32(_src, _kernel, _dst, &handle);

    return handle;
}