/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/


#include "dot_fp32.h"


_DECX_API_
de::DH decx::cuda::Dot_fp32(de::GPU_Vector& A, de::GPU_Vector& B, float* res)
{
    decx::_GPU_Vector* _A = dynamic_cast<decx::_GPU_Vector*>(&A);
    decx::_GPU_Vector* _B = dynamic_cast<decx::_GPU_Vector*>(&B);

    de::DH handle;
    if (!decx::cuP.is_init) {
        Print_Error_Message(4, CUDA_NOT_INIT);
        decx::err::CUDA_Not_init(&handle);
        return handle;
    }

    if (_A->length != _B->length) {
        decx::MDim_Not_Matching(&handle);
        Print_Error_Message(4, MAT_DIM_NOT_MATCH);
        return handle;
    }

    const size_t dev_len = decx::utils::ceil<size_t>(_A->length, 8) * 2;
    const size_t dev_tmp_size = decx::utils::ceil<size_t>(dev_len, REDUCTION_BLOCK_SIZE) * sizeof(float4);

    decx::cuda_stream* S = NULL;
    S = decx::CStream.stream_accessor_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        decx::err::CUDA_Stream_access_fail(&handle);
        Print_Error_Message(4, CUDA_STREAM_ACCESS_FAIL);
        return handle;
    }

    decx::PtrInfo<float4> dev_tmp1, dev_tmp2;
    if (decx::alloc::_device_malloc(&dev_tmp1, dev_tmp_size, true, S)) {
        decx::err::AllocateFailure(&handle);
        Print_Error_Message(4, ALLOC_FAIL);
        return handle;
    }
    if (decx::alloc::_device_malloc(&dev_tmp2, dev_tmp_size, true, S)) {
        decx::err::AllocateFailure(&handle);
        Print_Error_Message(4, ALLOC_FAIL);
        return handle;
    }

    decx::alloc::MIF<float4> dev_A, dev_B;
    dev_A.mem = dev_tmp1.ptr;
    dev_B.mem = dev_tmp2.ptr;

    decx::dot::Kdot_fp32((float4*)_A->Vec.ptr, (float4*)_B->Vec.ptr, &dev_A, &dev_B, dev_len,
        dev_tmp_size / sizeof(float4), S);

    float4 ans;
    if (dev_A.leading) {
        checkCudaErrors(cudaMemcpyAsync(
            &ans, dev_A.mem, sizeof(float4), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));
    }
    else {
        checkCudaErrors(cudaMemcpyAsync(
            &ans, dev_B.mem, sizeof(float4), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));
    }

    decx::_final_vec4_sum<float, float4>(&ans, res);

    checkCudaErrors(cudaDeviceSynchronize());

    decx::alloc::_device_dealloc(&dev_tmp1);
    decx::alloc::_device_dealloc(&dev_tmp2);
    S->detach();

    return handle;
}



_DECX_API_
de::DH decx::cuda::Dot_fp32(de::GPU_Matrix& A, de::GPU_Matrix& B, float* res)
{
    decx::_GPU_Matrix* _A = dynamic_cast<decx::_GPU_Matrix*>(&A);
    decx::_GPU_Matrix* _B = dynamic_cast<decx::_GPU_Matrix*>(&B);

    de::DH handle;
    if (!decx::cuP.is_init) {
        Print_Error_Message(4, CUDA_NOT_INIT);
        decx::err::CUDA_Not_init(&handle);
        return handle;
    }

    if (_A->width != _B->width || _A->height != _B->height) {
        decx::MDim_Not_Matching(&handle);
        Print_Error_Message(4, MAT_DIM_NOT_MATCH);
        return handle;
    }

    const size_t dev_len = decx::utils::ceil<size_t>(_A->_element_num, 8) * 2;
    const size_t dev_tmp_size = decx::utils::ceil<size_t>(dev_len, REDUCTION_BLOCK_SIZE) * sizeof(float4);

    decx::cuda_stream* S = NULL;
    S = decx::CStream.stream_accessor_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        decx::err::CUDA_Stream_access_fail(&handle);
        Print_Error_Message(4, CUDA_STREAM_ACCESS_FAIL);
        return handle;
    }

    decx::PtrInfo<float4> dev_tmp1, dev_tmp2;
    if (decx::alloc::_device_malloc(&dev_tmp1, dev_tmp_size, true, S)) {
        decx::err::AllocateFailure(&handle);
        Print_Error_Message(4, ALLOC_FAIL);
        return handle;
    }
    if (decx::alloc::_device_malloc(&dev_tmp2, dev_tmp_size, true, S)) {
        decx::err::AllocateFailure(&handle);
        Print_Error_Message(4, ALLOC_FAIL);
        return handle;
    }

    decx::alloc::MIF<float4> dev_A, dev_B;
    dev_A.mem = dev_tmp1.ptr;
    dev_B.mem = dev_tmp2.ptr;

    decx::dot::Kdot_fp32((float4*)_A->Mat.ptr, (float4*)_B->Mat.ptr, &dev_A, &dev_B, dev_len,
        dev_tmp_size / sizeof(float4), S);

    float4 ans;
    if (dev_A.leading) {
        checkCudaErrors(cudaMemcpyAsync(
            &ans, dev_A.mem, sizeof(float4), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));
    }
    else {
        checkCudaErrors(cudaMemcpyAsync(
            &ans, dev_B.mem, sizeof(float4), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));
    }

    decx::_final_vec4_sum<float, float4>(&ans, res);

    checkCudaErrors(cudaDeviceSynchronize());

    decx::alloc::_device_dealloc(&dev_tmp1);
    decx::alloc::_device_dealloc(&dev_tmp2);
    S->detach();

    return handle;
}



de::DH decx::cuda::Dot_fp32(de::GPU_Tensor& A, de::GPU_Tensor& B, float* res)
{
    decx::_GPU_Tensor* _A = dynamic_cast<decx::_GPU_Tensor*>(&A);
    decx::_GPU_Tensor* _B = dynamic_cast<decx::_GPU_Tensor*>(&B);

    de::DH handle;
    if (!decx::cuP.is_init) {
        Print_Error_Message(4, CUDA_NOT_INIT);
        decx::err::CUDA_Not_init(&handle);
        return handle;
    }

    if (_A->width != _B->width || _A->height != _B->height || _A->depth != _B->depth) {
        decx::MDim_Not_Matching(&handle);
        Print_Error_Message(4, MAT_DIM_NOT_MATCH);
        return handle;
    }

    const size_t dev_len = decx::utils::ceil<size_t>(_A->_element_num, 8) * 2;
    const size_t dev_tmp_size = decx::utils::ceil<size_t>(dev_len, REDUCTION_BLOCK_SIZE) * sizeof(float4);

    decx::cuda_stream* S = NULL;
    S = decx::CStream.stream_accessor_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        decx::err::CUDA_Stream_access_fail(&handle);
        Print_Error_Message(4, CUDA_STREAM_ACCESS_FAIL);
        return handle;
    }

    decx::PtrInfo<float4> dev_tmp1, dev_tmp2;
    if (decx::alloc::_device_malloc(&dev_tmp1, dev_tmp_size, true, S)) {
        decx::err::AllocateFailure(&handle);
        Print_Error_Message(4, ALLOC_FAIL);
        return handle;
    }
    if (decx::alloc::_device_malloc(&dev_tmp2, dev_tmp_size, true, S)) {
        decx::err::AllocateFailure(&handle);
        Print_Error_Message(4, ALLOC_FAIL);
        return handle;
    }

    decx::alloc::MIF<float4> dev_A, dev_B;
    dev_A.mem = dev_tmp1.ptr;
    dev_B.mem = dev_tmp2.ptr;

    decx::dot::Kdot_fp32((float4*)_A->Tens.ptr, (float4*)_B->Tens.ptr, &dev_A, &dev_B, dev_len,
        dev_tmp_size / sizeof(float4), S);

    float4 ans;
    if (dev_A.leading) {
        checkCudaErrors(cudaMemcpyAsync(
            &ans, dev_A.mem, sizeof(float4), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));
    }
    else {
        checkCudaErrors(cudaMemcpyAsync(
            &ans, dev_B.mem, sizeof(float4), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));
    }

    decx::_final_vec4_sum<float, float4>(&ans, res);

    checkCudaErrors(cudaDeviceSynchronize());

    decx::alloc::_device_dealloc(&dev_tmp1);
    decx::alloc::_device_dealloc(&dev_tmp2);
    S->detach();

    return handle;
}