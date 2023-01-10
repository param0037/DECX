/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/


#include "../cv_classes/cv_classes.h"
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

    const uint2 dst_buf_dim = make_uint2(decx::utils::ceil<uint>(dst->width, 64) * 64,
        decx::utils::ceil<uint>(dst->height, 16) * 16);

    const uint2 _work_space_dim = make_uint2(dst_buf_dim.x + 32, dst_buf_dim.y + 32);

    decx::cuda_stream* S = NULL;
    S = decx::CStream.stream_accessor_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        Print_Error_Message(4, CUDA_STREAM_ACCESS_FAIL);
        return;
    }

    decx::PtrInfo<float4> WS_buffer, dst_buffer;
    if (decx::alloc::_device_malloc(&WS_buffer, _work_space_dim.x * _work_space_dim.y * sizeof(uchar4), true, S)) {
        Print_Error_Message(4, ALLOC_FAIL);
        exit(-1);
    }
    if (decx::alloc::_device_malloc(&dst_buffer, dst_buf_dim.x * dst_buf_dim.y * sizeof(uchar4), true, S)) {
        Print_Error_Message(4, ALLOC_FAIL);
        exit(-1);
    }

    checkCudaErrors(cudaMemcpy2DAsync((uchar4*)WS_buffer.ptr + _work_space_dim.x * 16 + 16,
        _work_space_dim.x * sizeof(uchar4),
        src->Mat.ptr, src->pitch * sizeof(uchar4),
        src->width * sizeof(uchar4), src->height, cudaMemcpyHostToDevice, S->get_raw_stream_ref()));

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

    checkCudaErrors(cudaMemcpy2DAsync(dst->Mat.ptr, dst->pitch * sizeof(uchar4),
        dst_buffer.ptr, dst_buf_dim.x * sizeof(uchar4), dst->width * sizeof(uchar4), dst->height,
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

    const uint2 dst_buf_dim = make_uint2(decx::utils::ceil<uint>(dst->width, 64) * 64,
        decx::utils::ceil<uint>(dst->height, 16) * 16);

    const uint2 _work_space_dim = make_uint2(dst_buf_dim.x + 32, dst_buf_dim.y + 32);

    decx::cuda_stream* S = NULL;
    S = decx::CStream.stream_accessor_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        Print_Error_Message(4, CUDA_STREAM_ACCESS_FAIL);
        return;
    }

    decx::PtrInfo<float4> WS_buffer, dst_buffer;
    if (decx::alloc::_device_malloc(&WS_buffer, _work_space_dim.x * _work_space_dim.y * sizeof(uchar4), true, S)) {
        Print_Error_Message(4, ALLOC_FAIL);
        exit(-1);
    }
    if (decx::alloc::_device_malloc(&dst_buffer, dst_buf_dim.x * dst_buf_dim.y * sizeof(uchar4), true, S)) {
        Print_Error_Message(4, ALLOC_FAIL);
        exit(-1);
    }

    checkCudaErrors(cudaMemcpy2DAsync((uchar4*)WS_buffer.ptr + _work_space_dim.x * 16 + 16,
        _work_space_dim.x * sizeof(uchar4),
        src->Mat.ptr, src->pitch * sizeof(uchar4),
        src->width * sizeof(uchar4), src->height, cudaMemcpyHostToDevice, S->get_raw_stream_ref()));

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

    checkCudaErrors(cudaMemcpy2DAsync(dst->Mat.ptr, dst->pitch * sizeof(uchar4),
        dst_buffer.ptr, dst_buf_dim.x * sizeof(uchar4), dst->width * sizeof(uchar4), dst->height,
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

    const uint2 dst_buf_dim = make_uint2(decx::utils::ceil<uint>(dst->width, 64) * 64,
        decx::utils::ceil<uint>(dst->height, 16) * 16);

    const uint2 _work_space_dim = make_uint2(dst_buf_dim.x + 16, dst_buf_dim.y + 16);

    decx::cuda_stream* S = NULL;
    S = decx::CStream.stream_accessor_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        Print_Error_Message(4, CUDA_STREAM_ACCESS_FAIL);
        return;
    }

    decx::PtrInfo<float4> WS_buffer, dst_buffer;
    if (decx::alloc::_device_malloc(&WS_buffer, _work_space_dim.x * _work_space_dim.y * sizeof(uchar4), true, S)) {
        Print_Error_Message(4, ALLOC_FAIL);
        exit(-1);
    }
    if (decx::alloc::_device_malloc(&dst_buffer, dst_buf_dim.x * dst_buf_dim.y * sizeof(uchar4), true, S)) {
        Print_Error_Message(4, ALLOC_FAIL);
        exit(-1);
    }

    checkCudaErrors(cudaMemcpy2DAsync((uchar4*)WS_buffer.ptr + _work_space_dim.x * 8 + 8,
        _work_space_dim.x * sizeof(uchar4),
        src->Mat.ptr, src->pitch * sizeof(uchar4),
        src->width * sizeof(uchar4), src->height, cudaMemcpyHostToDevice, S->get_raw_stream_ref()));

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
    
    checkCudaErrors(cudaMemcpy2DAsync(dst->Mat.ptr, dst->pitch * sizeof(uchar4),
        dst_buffer.ptr, dst_buf_dim.x * sizeof(uchar4), dst->width * sizeof(uchar4), dst->height,
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

    const uint2 dst_buf_dim = make_uint2(decx::utils::ceil<uint>(dst->width, 64) * 64,
        decx::utils::ceil<uint>(dst->height, 16) * 16);

    const uint2 _work_space_dim = make_uint2(dst_buf_dim.x + 16, dst_buf_dim.y + 16);

    decx::cuda_stream* S = NULL;
    S = decx::CStream.stream_accessor_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        Print_Error_Message(4, CUDA_STREAM_ACCESS_FAIL);
        return;
    }

    decx::PtrInfo<float4> WS_buffer, dst_buffer;
    if (decx::alloc::_device_malloc(&WS_buffer, _work_space_dim.x * _work_space_dim.y * sizeof(uchar4), true, S)) {
        Print_Error_Message(4, ALLOC_FAIL);
        exit(-1);
    }
    if (decx::alloc::_device_malloc(&dst_buffer, dst_buf_dim.x * dst_buf_dim.y * sizeof(uchar4), true, S)) {
        Print_Error_Message(4, ALLOC_FAIL);
        exit(-1);
    }

    checkCudaErrors(cudaMemcpy2DAsync((uchar4*)WS_buffer.ptr + _work_space_dim.x * 8 + 8,
        _work_space_dim.x * sizeof(uchar4),
        src->Mat.ptr, src->pitch * sizeof(uchar4),
        src->width * sizeof(uchar4), src->height, cudaMemcpyHostToDevice, S->get_raw_stream_ref()));

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

    checkCudaErrors(cudaMemcpy2DAsync(dst->Mat.ptr, dst->pitch * sizeof(uchar4),
        dst_buffer.ptr, dst_buf_dim.x * sizeof(uchar4), dst->width * sizeof(uchar4), dst->height,
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

    const uint2 dst_buf_dim = make_uint2(decx::utils::ceil<uint>(dst->width, 256) * 256,
        decx::utils::ceil<uint>(dst->height, 16) * 16);

    const uint2 _work_space_dim = make_uint2(dst_buf_dim.x + 32, dst_buf_dim.y + 32);

    decx::cuda_stream* S = NULL;
    S = decx::CStream.stream_accessor_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        Print_Error_Message(4, CUDA_STREAM_ACCESS_FAIL);
        return;
    }

    decx::PtrInfo<float4> WS_buffer, dst_buffer;
    if (decx::alloc::_device_malloc(&WS_buffer, _work_space_dim.x * _work_space_dim.y * sizeof(uchar), true, S)) {
        Print_Error_Message(4, ALLOC_FAIL);
        exit(-1);
    }
    if (decx::alloc::_device_malloc(&dst_buffer, dst_buf_dim.x * dst_buf_dim.y * sizeof(uchar), true, S)) {
        Print_Error_Message(4, ALLOC_FAIL);
        exit(-1);
    }

    checkCudaErrors(cudaMemcpy2DAsync((uchar*)WS_buffer.ptr + _work_space_dim.x * 16 + 16,
        _work_space_dim.x * sizeof(uchar),
        src->Mat.ptr, src->pitch * sizeof(uchar),
        src->width * sizeof(uchar), src->height, cudaMemcpyHostToDevice, S->get_raw_stream_ref()));

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

    checkCudaErrors(cudaMemcpy2DAsync(dst->Mat.ptr, dst->pitch * sizeof(uchar),
        dst_buffer.ptr, dst_buf_dim.x * sizeof(uchar), dst->width * sizeof(uchar), dst->height,
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

    const uint2 dst_buf_dim = make_uint2(decx::utils::ceil<uint>(dst->width, 256) * 256,
        decx::utils::ceil<uint>(dst->height, 16) * 16);

    const uint2 _work_space_dim = make_uint2(dst_buf_dim.x + 16, dst_buf_dim.y + 16);

    decx::cuda_stream* S = NULL;
    S = decx::CStream.stream_accessor_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        Print_Error_Message(4, CUDA_STREAM_ACCESS_FAIL);
        return;
    }

    decx::PtrInfo<float4> WS_buffer, dst_buffer;
    if (decx::alloc::_device_malloc(&WS_buffer, _work_space_dim.x * _work_space_dim.y * sizeof(uchar), true, S)) {
        Print_Error_Message(4, ALLOC_FAIL);
        exit(-1);
    }
    if (decx::alloc::_device_malloc(&dst_buffer, dst_buf_dim.x * dst_buf_dim.y * sizeof(uchar), true, S)) {
        Print_Error_Message(4, ALLOC_FAIL);
        exit(-1);
    }

    checkCudaErrors(cudaMemcpy2DAsync((uchar*)WS_buffer.ptr + _work_space_dim.x * 8 + 8,
        _work_space_dim.x * sizeof(uchar),
        src->Mat.ptr, src->pitch * sizeof(uchar),
        src->width * sizeof(uchar), src->height, cudaMemcpyHostToDevice, S->get_raw_stream_ref()));

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

    checkCudaErrors(cudaMemcpy2DAsync(dst->Mat.ptr, dst->pitch * sizeof(uchar),
        dst_buffer.ptr, dst_buf_dim.x * sizeof(uchar), dst->width * sizeof(uchar), dst->height,
        cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));

    checkCudaErrors(cudaDeviceSynchronize());
    S->detach();

    decx::alloc::_device_dealloc(&WS_buffer);
    decx::alloc::_device_dealloc(&dst_buffer);
}




de::DH de::vis::cuda::NLM_RGB(de::Matrix& src, de::Matrix& dst, uint search_window_radius, uint template_window_radius, float h)
{
    de::DH handle;

    if (!decx::cuP.is_init) {
        decx::Not_init(&handle);
        Print_Error_Message(4, CUDA_NOT_INIT);
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

    if (!decx::cuP.is_init) {
        decx::Not_init(&handle);
        Print_Error_Message(4, CUDA_NOT_INIT);
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

    if (!decx::cuP.is_init) {
        decx::Not_init(&handle);
        Print_Error_Message(4, CUDA_NOT_INIT);
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