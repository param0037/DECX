/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#include "cuda_summing.cuh"



de::DH de::cuda::Sum_fp32(de::GPU_Vector& src, float* res)
{
    decx::_GPU_Vector* _src = dynamic_cast<decx::_GPU_Vector*>(&src);

    de::DH handle;
    decx::bp::_sum_vec4_fp32_1D((float*)_src->Vec.ptr, res, _src->_length / 4, &handle);

    decx::err::Success(&handle);
    return handle;
}


de::DH de::cuda::Sum_fp16(de::GPU_Vector& src, de::Half* res)
{
    decx::_GPU_Vector* _src = dynamic_cast<decx::_GPU_Vector*>(&src);

    de::DH handle;

    const size_t dev_len = _src->_length / 4;
    decx::PtrInfo<float4> dev_tmp1, dev_tmp2;
    const size_t tmp_dev_size = decx::utils::ceil<size_t>(dev_len / 2, 512);

    decx::cuda_stream* S = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        decx::err::CUDA_Stream_access_fail(&handle);
        Print_Error_Message(4, CUDA_STREAM_ACCESS_FAIL);
        return handle;
    }

    if (decx::alloc::_device_malloc(&dev_tmp1, tmp_dev_size * sizeof(float4), true, S)) {
        decx::err::AllocateFailure(&handle);
        Print_Error_Message(4, ALLOC_FAIL);
        return handle;
    }
    if (decx::alloc::_device_malloc(&dev_tmp2, tmp_dev_size * sizeof(float4), true, S)) {
        decx::err::AllocateFailure(&handle);
        Print_Error_Message(4, ALLOC_FAIL);
        return handle;
    }

    decx::alloc::MIF<float4> dev_A, dev_B;
    dev_A.mem = dev_tmp1.ptr;
    dev_B.mem = dev_tmp2.ptr;

    decx::bp::_sum_fp16((float4*)_src->Vec.ptr, &dev_A, &dev_B, dev_len, tmp_dev_size, S);

    float4* ans = new float4();
    half2_8* ans_half2_8_ptr = reinterpret_cast<half2_8*>(ans);
    float _ans = 0;
    if (dev_A.leading) {
        checkCudaErrors(cudaMemcpyAsync(
            ans, dev_A.mem, sizeof(float4), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));
    }
    else {
        checkCudaErrors(cudaMemcpyAsync(
            ans, dev_B.mem, sizeof(float4), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));
    }

    checkCudaErrors(cudaDeviceSynchronize());

    decx::_final_vec8_sum_fp16(ans_half2_8_ptr, &_ans);
    *((__half*)res) = __float2half(_ans);

    decx::alloc::_device_dealloc(&dev_tmp1);
    decx::alloc::_device_dealloc(&dev_tmp2);
    delete ans;
    S->detach();

    decx::err::Success(&handle);
    return handle;
}


de::DH de::cuda::Sum_fp32(de::GPU_Matrix& src, float* res)
{
    decx::_GPU_Matrix* _src = dynamic_cast<decx::_GPU_Matrix*>(&src);

    de::DH handle;
    decx::bp::_sum_vec4_fp32_1D((float*)_src->Mat.ptr, res, _src->Pitch() * _src->Height() / 4, &handle);

    decx::err::Success(&handle);
    return handle;
}



de::DH de::cuda::Sum_fp16(de::GPU_Matrix& src, de::Half* res)
{
    decx::_GPU_Matrix* _src = dynamic_cast<decx::_GPU_Matrix*>(&src);

    de::DH handle;

    const size_t dev_len = _src->Pitch() * _src->Height() / 4;
    decx::PtrInfo<float4> dev_tmp1, dev_tmp2;
    const size_t tmp_dev_size = decx::utils::ceil<size_t>(dev_len / 2, 512);

    decx::cuda_stream* S = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        decx::err::CUDA_Stream_access_fail(&handle);
        Print_Error_Message(4, CUDA_STREAM_ACCESS_FAIL);
        return handle;
    }

    if (decx::alloc::_device_malloc(&dev_tmp1, tmp_dev_size * sizeof(float4), true, S)) {
        decx::err::AllocateFailure(&handle);
        Print_Error_Message(4, ALLOC_FAIL);
        return handle;
    }
    if (decx::alloc::_device_malloc(&dev_tmp2, tmp_dev_size * sizeof(float4), true, S)) {
        decx::err::AllocateFailure(&handle);
        Print_Error_Message(4, ALLOC_FAIL);
        return handle;
    }

    decx::alloc::MIF<float4> dev_A, dev_B;
    dev_A.mem = dev_tmp1.ptr;
    dev_B.mem = dev_tmp2.ptr;

    decx::bp::_sum_fp16((float4*)_src->Mat.ptr, &dev_A, &dev_B, dev_len, tmp_dev_size, S);

    float4* ans = new float4();
    half2_8* ans_half2_8_ptr = reinterpret_cast<half2_8*>(ans);
    float _ans = 0;
    if (dev_A.leading) {
        checkCudaErrors(cudaMemcpyAsync(
            ans, dev_A.mem, sizeof(float4), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));
    }
    else {
        checkCudaErrors(cudaMemcpyAsync(
            ans, dev_B.mem, sizeof(float4), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));
    }

    checkCudaErrors(cudaDeviceSynchronize());

    decx::_final_vec8_sum_fp16(ans_half2_8_ptr, &_ans);
    *((__half*)res) = __float2half(_ans);

    decx::alloc::_device_dealloc(&dev_tmp1);
    decx::alloc::_device_dealloc(&dev_tmp2);
    delete ans;
    S->detach();

    decx::err::Success(&handle);
    return handle;
}