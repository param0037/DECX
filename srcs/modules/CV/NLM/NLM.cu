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


#include "../../classes/Matrix.h"
#include "../../handles/decx_handles.h"
#include "NLM_BGR.cuh"
#include "NLM_gray.cuh"
#include "NLM_BGR_keep_alpha.cuh"


namespace de
{
    namespace vis
    {
        namespace cuda
        {
            _DECX_API_ de::DH NLM_RGB(de::Matrix& src, de::Matrix& dst, uint search_window_radius, uint template_window_radius, float h);


            _DECX_API_ de::DH NLM_RGB_keep_alpha(de::Matrix& src, de::Matrix& dst, uint search_window_radius, uint template_window_radius, float h);


            _DECX_API_ de::DH NLM_Gray(de::Matrix& src, de::Matrix& dst, uint search_window_radius, uint template_window_radius, float h);
        }
    }
}



namespace decx
{
    namespace vis
    {
        void NLM_RGB_r16(decx::_Matrix* src, decx::_Matrix* dst, uint search_window_radius, uint template_window_radius, float h);


        void NLM_RGB_r16_keep_alpha(decx::_Matrix* src, decx::_Matrix* dst, uint search_window_radius, uint template_window_radius, float h);


        void NLM_RGB_r8(decx::_Matrix* src, decx::_Matrix* dst, uint search_window_radius, uint template_window_radius, float h);


        void NLM_RGB_r8_keep_alpha(decx::_Matrix* src, decx::_Matrix* dst, uint search_window_radius, uint template_window_radius, float h);


        void NLM_gray_r16(decx::_Matrix* src, decx::_Matrix* dst, uint search_window_radius, uint template_window_radius, float h);


        void NLM_gray_r8(decx::_Matrix* src, decx::_Matrix* dst, uint search_window_radius, uint template_window_radius, float h);
    }
}


void decx::vis::NLM_RGB_r16(decx::_Matrix* src, decx::_Matrix* dst, uint search_window_radius, uint template_window_radius, float h)
{
    const uint2 kernel_shift = make_uint2(16 - (search_window_radius + template_window_radius),
        16 - (search_window_radius + template_window_radius));

    const uint eq_ker_len = (search_window_radius * 2 + 1) * (search_window_radius * 2 + 1),
        eq_Wker = search_window_radius * 2 + 1;

    const uint2 dst_buf_dim = make_uint2(decx::utils::ceil<uint>(dst->Width(), 64) * 64,
        decx::utils::ceil<uint>(dst->Height(), 16) * 16);

    const uint2 _work_space_dim = make_uint2(dst_buf_dim.x + 32, dst_buf_dim.y + 32);

    decx::cuda_stream* S = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        
        return;
    }

    decx::PtrInfo<float4> WS_buffer, dst_buffer;
    if (decx::alloc::_device_malloc(&WS_buffer, _work_space_dim.x * _work_space_dim.y * sizeof(uchar4), true, S)) {
        
        exit(-1);
    }
    if (decx::alloc::_device_malloc(&dst_buffer, dst_buf_dim.x * dst_buf_dim.y * sizeof(uchar4), true, S)) {
        
        exit(-1);
    }

    checkCudaErrors(cudaMemcpy2DAsync((uchar4*)WS_buffer.ptr + _work_space_dim.x * 16 + 16,
        _work_space_dim.x * sizeof(uchar4),
        src->Mat.ptr, src->Pitch() * sizeof(uchar4),
        src->Width() * sizeof(uchar4), src->Height(), cudaMemcpyHostToDevice, S->get_raw_stream_ref()));

    const dim3 block(16, 16);
    const dim3 grid(dst_buf_dim.y / 16, dst_buf_dim.x / 64);

    switch (template_window_radius)
    {
    case 1:
        cu_NLM_r16_BGR_N3x3 << <grid, block, 0, S->get_raw_stream_ref() >> > (
            WS_buffer.ptr, dst_buffer.ptr, _work_space_dim.x / 4, dst_buf_dim.x / 4,
            eq_ker_len, eq_Wker, kernel_shift, h * h);
        break;

    case 2:
        cu_NLM_r16_BGR_N5x5 << <grid, block, 0, S->get_raw_stream_ref() >> > (
            WS_buffer.ptr, dst_buffer.ptr, _work_space_dim.x / 4, dst_buf_dim.x / 4,
            eq_ker_len, eq_Wker, kernel_shift, h * h);
        break;

    default:
        break;
    }

    checkCudaErrors(cudaMemcpy2DAsync(dst->Mat.ptr, dst->Pitch() * sizeof(uchar4),
        dst_buffer.ptr, dst_buf_dim.x * sizeof(uchar4), dst->Width() * sizeof(uchar4), dst->Height(),
        cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));

    checkCudaErrors(cudaDeviceSynchronize());
    S->detach();

    decx::alloc::_device_dealloc(&WS_buffer);
    decx::alloc::_device_dealloc(&dst_buffer);
}



void decx::vis::NLM_RGB_r16_keep_alpha(decx::_Matrix* src, decx::_Matrix* dst, uint search_window_radius, uint template_window_radius, float h)
{
    const uint2 kernel_shift = make_uint2(16 - (search_window_radius + template_window_radius),
        16 - (search_window_radius + template_window_radius));

    const uint eq_ker_len = (search_window_radius * 2 + 1) * (search_window_radius * 2 + 1),
        eq_Wker = search_window_radius * 2 + 1;

    const uint2 dst_buf_dim = make_uint2(decx::utils::ceil<uint>(dst->Width(), 64) * 64,
        decx::utils::ceil<uint>(dst->Height(), 16) * 16);

    const uint2 _work_space_dim = make_uint2(dst_buf_dim.x + 32, dst_buf_dim.y + 32);

    decx::cuda_stream* S = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        
        return;
    }

    decx::PtrInfo<float4> WS_buffer, dst_buffer;
    if (decx::alloc::_device_malloc(&WS_buffer, _work_space_dim.x * _work_space_dim.y * sizeof(uchar4), true, S)) {
        
        exit(-1);
    }
    if (decx::alloc::_device_malloc(&dst_buffer, dst_buf_dim.x * dst_buf_dim.y * sizeof(uchar4), true, S)) {
        
        exit(-1);
    }

    checkCudaErrors(cudaMemcpy2DAsync((uchar4*)WS_buffer.ptr + _work_space_dim.x * 16 + 16,
        _work_space_dim.x * sizeof(uchar4),
        src->Mat.ptr, src->Pitch() * sizeof(uchar4),
        src->Width() * sizeof(uchar4), src->Height(), cudaMemcpyHostToDevice, S->get_raw_stream_ref()));

    const dim3 block(16, 16);
    const dim3 grid(dst_buf_dim.y / 16, dst_buf_dim.x / 64);

    switch (template_window_radius)
    {
    case 1:
        cu_NLM_r16_BGR_KPAL_N3x3 << <grid, block, 0, S->get_raw_stream_ref() >> > (
            WS_buffer.ptr, dst_buffer.ptr, _work_space_dim.x / 4, dst_buf_dim.x / 4,
            eq_ker_len, eq_Wker, kernel_shift, h * h);
        break;

    case 2:
        cu_NLM_r16_BGR_KPAL_N5x5 << <grid, block, 0, S->get_raw_stream_ref() >> > (
            WS_buffer.ptr, dst_buffer.ptr, _work_space_dim.x / 4, dst_buf_dim.x / 4,
            eq_ker_len, eq_Wker, kernel_shift, h * h);
        break;

    default:
        break;
    }

    checkCudaErrors(cudaMemcpy2DAsync(dst->Mat.ptr, dst->Pitch() * sizeof(uchar4),
        dst_buffer.ptr, dst_buf_dim.x * sizeof(uchar4), dst->Width() * sizeof(uchar4), dst->Height(),
        cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));

    checkCudaErrors(cudaDeviceSynchronize());
    S->detach();

    decx::alloc::_device_dealloc(&WS_buffer);
    decx::alloc::_device_dealloc(&dst_buffer);
}



void decx::vis::NLM_RGB_r8(decx::_Matrix* src, decx::_Matrix* dst, uint search_window_radius, uint template_window_radius, float h)
{
    const uint2 kernel_shift = make_uint2(8 - (search_window_radius + template_window_radius),
        8 - (search_window_radius + template_window_radius));

    const uint eq_ker_len = (search_window_radius * 2 + 1) * (search_window_radius * 2 + 1),
        eq_Wker = search_window_radius * 2 + 1;

    const uint2 dst_buf_dim = make_uint2(decx::utils::ceil<uint>(dst->Width(), 64) * 64,
        decx::utils::ceil<uint>(dst->Height(), 16) * 16);

    const uint2 _work_space_dim = make_uint2(dst_buf_dim.x + 16, dst_buf_dim.y + 16);

    decx::cuda_stream* S = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        
        return;
    }

    decx::PtrInfo<float4> WS_buffer, dst_buffer;
    if (decx::alloc::_device_malloc(&WS_buffer, _work_space_dim.x * _work_space_dim.y * sizeof(uchar4), true, S)) {
        
        exit(-1);
    }
    if (decx::alloc::_device_malloc(&dst_buffer, dst_buf_dim.x * dst_buf_dim.y * sizeof(uchar4), true, S)) {
        
        exit(-1);
    }

    checkCudaErrors(cudaMemcpy2DAsync((uchar4*)WS_buffer.ptr + _work_space_dim.x * 8 + 8,
        _work_space_dim.x * sizeof(uchar4),
        src->Mat.ptr, src->Pitch() * sizeof(uchar4),
        src->Width() * sizeof(uchar4), src->Height(), cudaMemcpyHostToDevice, S->get_raw_stream_ref()));

    const dim3 block(16, 16);
    const dim3 grid(dst_buf_dim.y / 16, dst_buf_dim.x / 64);

    switch (template_window_radius)
    {
    case 1:
        cu_NLM_r8_BGR_N3x3 << <grid, block, 0, S->get_raw_stream_ref() >> > (
            WS_buffer.ptr, dst_buffer.ptr, _work_space_dim.x / 4, dst_buf_dim.x / 4,
            eq_ker_len, eq_Wker, kernel_shift, h * h);
        break;

    case 2:
        cu_NLM_r8_BGR_N5x5 << <grid, block, 0, S->get_raw_stream_ref() >> > (
            WS_buffer.ptr, dst_buffer.ptr, _work_space_dim.x / 4, dst_buf_dim.x / 4,
            eq_ker_len, eq_Wker, kernel_shift, h * h);
        break;

    default:
        break;
    }
    
    checkCudaErrors(cudaMemcpy2DAsync(dst->Mat.ptr, dst->Pitch() * sizeof(uchar4),
        dst_buffer.ptr, dst_buf_dim.x * sizeof(uchar4), dst->Width() * sizeof(uchar4), dst->Height(),
        cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));

    checkCudaErrors(cudaDeviceSynchronize());
    S->detach();

    decx::alloc::_device_dealloc(&WS_buffer);
    decx::alloc::_device_dealloc(&dst_buffer);
}




void decx::vis::NLM_RGB_r8_keep_alpha(decx::_Matrix* src, decx::_Matrix* dst, uint search_window_radius, uint template_window_radius, float h)
{
    const uint2 kernel_shift = make_uint2(8 - (search_window_radius + template_window_radius),
        8 - (search_window_radius + template_window_radius));

    const uint eq_ker_len = (search_window_radius * 2 + 1) * (search_window_radius * 2 + 1),
        eq_Wker = search_window_radius * 2 + 1;

    const uint2 dst_buf_dim = make_uint2(decx::utils::ceil<uint>(dst->Width(), 64) * 64,
        decx::utils::ceil<uint>(dst->Height(), 16) * 16);

    const uint2 _work_space_dim = make_uint2(dst_buf_dim.x + 16, dst_buf_dim.y + 16);

    decx::cuda_stream* S = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        
        return;
    }

    decx::PtrInfo<float4> WS_buffer, dst_buffer;
    if (decx::alloc::_device_malloc(&WS_buffer, _work_space_dim.x * _work_space_dim.y * sizeof(uchar4), true, S)) {
        
        exit(-1);
    }
    if (decx::alloc::_device_malloc(&dst_buffer, dst_buf_dim.x * dst_buf_dim.y * sizeof(uchar4), true, S)) {
        
        exit(-1);
    }

    checkCudaErrors(cudaMemcpy2DAsync((uchar4*)WS_buffer.ptr + _work_space_dim.x * 8 + 8,
        _work_space_dim.x * sizeof(uchar4),
        src->Mat.ptr, src->Pitch() * sizeof(uchar4),
        src->Width() * sizeof(uchar4), src->Height(), cudaMemcpyHostToDevice, S->get_raw_stream_ref()));

    const dim3 block(16, 16);
    const dim3 grid(dst_buf_dim.y / 16, dst_buf_dim.x / 64);

    switch (template_window_radius)
    {
    case 1:
        cu_NLM_r8_BGR_KPAL_N3x3 << <grid, block, 0, S->get_raw_stream_ref() >> > (
            WS_buffer.ptr, dst_buffer.ptr, _work_space_dim.x / 4, dst_buf_dim.x / 4,
            eq_ker_len, eq_Wker, kernel_shift, h * h);
        break;

    case 2:
        cu_NLM_r8_BGR_KPAL_N5x5 << <grid, block, 0, S->get_raw_stream_ref() >> > (
            WS_buffer.ptr, dst_buffer.ptr, _work_space_dim.x / 4, dst_buf_dim.x / 4,
            eq_ker_len, eq_Wker, kernel_shift, h * h);
        break;

    default:
        break;
    }

    checkCudaErrors(cudaMemcpy2DAsync(dst->Mat.ptr, dst->Pitch() * sizeof(uchar4),
        dst_buffer.ptr, dst_buf_dim.x * sizeof(uchar4), dst->Width() * sizeof(uchar4), dst->Height(),
        cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));

    checkCudaErrors(cudaDeviceSynchronize());
    S->detach();

    decx::alloc::_device_dealloc(&WS_buffer);
    decx::alloc::_device_dealloc(&dst_buffer);
}




void decx::vis::NLM_gray_r16(decx::_Matrix* src, decx::_Matrix* dst, uint search_window_radius, uint template_window_radius, float h)
{
    const uint2 kernel_shift = make_uint2(16 - (search_window_radius + template_window_radius),
        16 - (search_window_radius + template_window_radius));

    const uint eq_ker_len = (search_window_radius * 2 + 1) * (search_window_radius * 2 + 1),
        eq_Wker = search_window_radius * 2 + 1;

    const uint2 dst_buf_dim = make_uint2(decx::utils::ceil<uint>(dst->Width(), 256) * 256,
        decx::utils::ceil<uint>(dst->Height(), 16) * 16);

    const uint2 _work_space_dim = make_uint2(dst_buf_dim.x + 32, dst_buf_dim.y + 32);

    decx::cuda_stream* S = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        
        return;
    }

    decx::PtrInfo<float4> WS_buffer, dst_buffer;
    if (decx::alloc::_device_malloc(&WS_buffer, _work_space_dim.x * _work_space_dim.y * sizeof(uchar), true, S)) {
        
        exit(-1);
    }
    if (decx::alloc::_device_malloc(&dst_buffer, dst_buf_dim.x * dst_buf_dim.y * sizeof(uchar), true, S)) {
        
        exit(-1);
    }

    checkCudaErrors(cudaMemcpy2DAsync((uchar*)WS_buffer.ptr + _work_space_dim.x * 16 + 16,
        _work_space_dim.x * sizeof(uchar),
        src->Mat.ptr, src->Pitch() * sizeof(uchar),
        src->Width() * sizeof(uchar), src->Height(), cudaMemcpyHostToDevice, S->get_raw_stream_ref()));

    const dim3 block(16, 16);
    const dim3 grid(dst_buf_dim.y / 16, dst_buf_dim.x / 256);

    switch (template_window_radius)
    {
    case 1:
        cu_NLM_r16_gray_N3x3 << <grid, block, 0, S->get_raw_stream_ref() >> > (
            WS_buffer.ptr, dst_buffer.ptr, _work_space_dim.x / 16, dst_buf_dim.x / 16,
            eq_ker_len, eq_Wker, kernel_shift, h * h);
        break;

    case 2:
        cu_NLM_r16_gray_N5x5 << <grid, block, 0, S->get_raw_stream_ref() >> > (
            WS_buffer.ptr, dst_buffer.ptr, _work_space_dim.x / 16, dst_buf_dim.x / 16,
            eq_ker_len, eq_Wker, kernel_shift, h * h);
        break;
        
    default:
        break;
    }

    checkCudaErrors(cudaMemcpy2DAsync(dst->Mat.ptr, dst->Pitch() * sizeof(uchar),
        dst_buffer.ptr, dst_buf_dim.x * sizeof(uchar), dst->Width() * sizeof(uchar), dst->Height(),
        cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));

    checkCudaErrors(cudaDeviceSynchronize()); 
    S->detach();

    decx::alloc::_device_dealloc(&WS_buffer);
    decx::alloc::_device_dealloc(&dst_buffer);
}



void decx::vis::NLM_gray_r8(decx::_Matrix* src, decx::_Matrix* dst, uint search_window_radius, uint template_window_radius, float h)
{
    const uint2 kernel_shift = make_uint2(8 - (search_window_radius + template_window_radius),
        8 - (search_window_radius + template_window_radius));

    const uint eq_ker_len = (search_window_radius * 2 + 1) * (search_window_radius * 2 + 1),
        eq_Wker = search_window_radius * 2 + 1;

    const uint2 dst_buf_dim = make_uint2(decx::utils::ceil<uint>(dst->Width(), 256) * 256,
        decx::utils::ceil<uint>(dst->Height(), 16) * 16);

    const uint2 _work_space_dim = make_uint2(dst_buf_dim.x + 16, dst_buf_dim.y + 16);

    decx::cuda_stream* S = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        
        return;
    }

    decx::PtrInfo<float4> WS_buffer, dst_buffer;
    if (decx::alloc::_device_malloc(&WS_buffer, _work_space_dim.x * _work_space_dim.y * sizeof(uchar), true, S)) {
        
        exit(-1);
    }
    if (decx::alloc::_device_malloc(&dst_buffer, dst_buf_dim.x * dst_buf_dim.y * sizeof(uchar), true, S)) {
        
        exit(-1);
    }

    checkCudaErrors(cudaMemcpy2DAsync((uchar*)WS_buffer.ptr + _work_space_dim.x * 8 + 8,
        _work_space_dim.x * sizeof(uchar),
        src->Mat.ptr, src->Pitch() * sizeof(uchar),
        src->Width() * sizeof(uchar), src->Height(), cudaMemcpyHostToDevice, S->get_raw_stream_ref()));

    const dim3 block(16, 16);
    const dim3 grid(dst_buf_dim.y / 16, dst_buf_dim.x / 256);

    switch (template_window_radius)
    {
    case 1:
        cu_NLM_r8_gray_N3x3 << <grid, block, 0, S->get_raw_stream_ref() >> > (
            WS_buffer.ptr, dst_buffer.ptr, _work_space_dim.x / 16, dst_buf_dim.x / 16,
            eq_ker_len, eq_Wker, kernel_shift, h * h);
        break;

    case 2:
        cu_NLM_r8_gray_N5x5 << <grid, block, 0, S->get_raw_stream_ref() >> > (
            WS_buffer.ptr, dst_buffer.ptr, _work_space_dim.x / 16, dst_buf_dim.x / 16,
            eq_ker_len, eq_Wker, kernel_shift, h * h);
        break;

    default:
        break;
    }

    checkCudaErrors(cudaMemcpy2DAsync(dst->Mat.ptr, dst->Pitch() * sizeof(uchar),
        dst_buffer.ptr, dst_buf_dim.x * sizeof(uchar), dst->Width() * sizeof(uchar), dst->Height(),
        cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));

    checkCudaErrors(cudaDeviceSynchronize());
    S->detach();

    decx::alloc::_device_dealloc(&WS_buffer);
    decx::alloc::_device_dealloc(&dst_buffer);
}




de::DH de::vis::cuda::NLM_RGB(de::Matrix& src, de::Matrix& dst, uint search_window_radius, uint template_window_radius, float h)
{
    de::DH handle;

    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CUDA_not_init, CUDA_NOT_INIT);
        exit(-1);
    }

    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);
    decx::_Matrix* _dst = dynamic_cast<decx::_Matrix*>(&dst);

    if (search_window_radius + template_window_radius > 8) {
        decx::vis::NLM_RGB_r16(_src, _dst, search_window_radius, template_window_radius, h);
    }
    else {
        decx::vis::NLM_RGB_r8(_src, _dst, search_window_radius, template_window_radius, h);
    }
    
    return handle;
}



de::DH de::vis::cuda::NLM_RGB_keep_alpha(de::Matrix& src, de::Matrix& dst, uint search_window_radius, uint template_window_radius, float h)
{
    de::DH handle;

    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CUDA_not_init, CUDA_NOT_INIT);
        exit(-1);
    }

    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);
    decx::_Matrix* _dst = dynamic_cast<decx::_Matrix*>(&dst);

    if (search_window_radius + template_window_radius > 8) {
        decx::vis::NLM_RGB_r16_keep_alpha(_src, _dst, search_window_radius, template_window_radius, h);
    }
    else {
        decx::vis::NLM_RGB_r8_keep_alpha(_src, _dst, search_window_radius, template_window_radius, h);
    }

    return handle;
}



de::DH de::vis::cuda::NLM_Gray(de::Matrix& src, de::Matrix& dst, uint search_window_radius, uint template_window_radius, float h)
{
    de::DH handle;

    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CUDA_not_init, CUDA_NOT_INIT);
        exit(-1);
    }

    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);
    decx::_Matrix* _dst = dynamic_cast<decx::_Matrix*>(&dst);

    if (search_window_radius + template_window_radius > 8) {
        decx::vis::NLM_gray_r16(_src, _dst, search_window_radius, template_window_radius, h);
    }
    else {
        decx::vis::NLM_gray_r8(_src, _dst, search_window_radius, template_window_radius, h);
    }

    return handle;
}