/**
*    ---------------------------------------------------------------------
*    Author : Wayne Anderson
*    Date   : 2021.04.16
*    ---------------------------------------------------------------------
*    This is a part of the open source program named "DECX", copyright c Wayne,
*    2021.04.16
*/

#ifndef _DEV_CONV2_BORDER_IGNORED_FP16_H_
#define _DEV_CONV2_BORDER_IGNORED_FP16_H_

#include "../../../../core/basic.h"
#include "hconv2_kernel_callers.h"
#include "../../../../classes/GPU_Matrix.h"
#include "../../../conv_utils.h"


using decx::_Matrix;
using decx::alloc::MIF;
using decx::_GPU_Matrix;


namespace decx
{
    /*\
    * 8 x 8
    * This function complete a comand of a convolutipon kernel, which is in type of border-ignored
    * In order to save the device memory, just allocate memory of suitable size
    */
    static void _dev_Conv2_NB_R8x8_fp16(decx::_GPU_Matrix* src, decx::_GPU_Matrix* kernel, decx::_GPU_Matrix* dst, 
        de::DH* handle, const int flag);


    /*\
    * 16 x 16
    * This function complete a comand of a convolutipon kernel, which is in type of border-ignored
    * In order to save the device memory, just allocate memory of suitable size
    */
    static void _dev_Conv2_NB_R16x16_fp16(decx::_GPU_Matrix* src, decx::_GPU_Matrix* kernel, decx::_GPU_Matrix* dst, 
        de::DH* handle, const int flag);


    /*\
    * 8 x 16
    * This function complete a comand of a convolutipon kernel, which is in type of border-ignored
    * In order to save the device memory, just allocate memory of suitable size
    */
    static void _dev_Conv2_NB_R8x16_fp16(decx::_GPU_Matrix* src, decx::_GPU_Matrix* kernel, decx::_GPU_Matrix* dst, 
        de::DH* handle, const int flag);


    /*\
    * 16 x 8
    * This function complete a comand of a convolutipon kernel, which is in type of border-ignored
    * In order to save the device memory, just allocate memory of suitable size
    */
    static void _dev_Conv2_NB_R16x8_fp16(decx::_GPU_Matrix* src, decx::_GPU_Matrix* kernel, decx::_GPU_Matrix* dst, 
        de::DH* handle, const int flag);



    static void dev_hConv2_border_ignore(decx::_GPU_Matrix& src, decx::_GPU_Matrix& kernel, decx::_GPU_Matrix& dst, 
        de::DH* handle, const int flag);
}




static void decx::_dev_Conv2_NB_R8x8_fp16(decx::_GPU_Matrix* src, decx::_GPU_Matrix* kernel, decx::_GPU_Matrix* dst, 
    de::DH* handle, const int flag)
{
    float4* Dsrc,
        * Ddst;

    int2 Dsrc_alloc_dim,
        Ddst_alloc_dim,
        ker_dim;

    ker_dim.x = kernel->width;            ker_dim.y = kernel->height;
    // first, allocate the memory according to R8 alignments
    Ddst_alloc_dim.x = decx::utils::ceil<int>(dst->width, 64) * bounded_kernel_R8 * 2;
    Ddst_alloc_dim.y = decx::utils::ceil<int>(dst->height, 16) * bounded_kernel_R8 * 2;

    Dsrc_alloc_dim.x = Ddst_alloc_dim.x + bounded_kernel_R8 / 4;        // bounded_kernel_R8 * 2 / 4
    Dsrc_alloc_dim.y = Ddst_alloc_dim.y + bounded_kernel_R8 * 2;

    size_t dev_src_size = static_cast<size_t>(Dsrc_alloc_dim.x) * static_cast<size_t>(Dsrc_alloc_dim.y);
    size_t dev_dst_size = static_cast<size_t>(Ddst_alloc_dim.x) * static_cast<size_t>(Ddst_alloc_dim.y);


    decx::PtrInfo<float4> dev_tmp_1, dev_tmp_2;
    if (decx::alloc::_device_malloc(&dev_tmp_1, dev_src_size * sizeof(float4))) {
        Print_Error_Message(4, DEV_ALLOC_FAIL);
        decx::err::device_AllocateFailure(handle);
        return;
    }
    if (decx::alloc::_device_malloc(&dev_tmp_2, dev_dst_size * sizeof(float4))) {
        Print_Error_Message(4, DEV_ALLOC_FAIL);
        decx::err::device_AllocateFailure(handle);
        return;
    }

    Dsrc = dev_tmp_1.ptr;
    Ddst = dev_tmp_2.ptr;

    decx::cuda_stream* S = NULL;
    S = decx::CStream.stream_accessor_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        Print_Error_Message(4, CUDA_STREAM_ACCESS_FAIL);
        decx::err::CUDA_Stream_access_fail(handle);
        return;
    }

    uint offset_lin = 0, offset_ker = 0;

    for (int i = 0; i < kernel->height; ++i) {
        cudaMemcpyToSymbolAsync(Const_Mem,
            (de::Half*)kernel->Mat.ptr + offset_ker,
            kernel->width * sizeof(de::Half),
            offset_lin * sizeof(de::Half), cudaMemcpyDeviceToDevice, S->get_raw_stream_ref());

        offset_lin += kernel->width;
        offset_ker += kernel->pitch;
    }

    if (ker_dim.x == (bounded_kernel_R8 * 2 + 1) && ker_dim.y == (bounded_kernel_R8 * 2 + 1)) {
        checkCudaErrors(cudaMemcpy2DAsync(Dsrc,
            Dsrc_alloc_dim.x * sizeof(float4),
            src->Mat.ptr,
            src->pitch * sizeof(de::Half),
            src->width * sizeof(de::Half),
            src->height,
            cudaMemcpyDeviceToDevice,
            S->get_raw_stream_ref()));                            // copy the datas of src from host to device

        hconv2_kernel_exact8x8(Dsrc, Ddst, Dsrc_alloc_dim, Ddst_alloc_dim, ker_dim, S, flag);
    }
    else {
        int2 src_diff;
        src_diff.x = bounded_kernel_R8 - ker_dim.y / 2;                src_diff.y = bounded_kernel_R8 - ker_dim.x / 2;
        checkCudaErrors(cudaMemcpy2DAsync(
            reinterpret_cast<de::Half*>(Dsrc) + src_diff.x * Dsrc_alloc_dim.x * 4 + src_diff.y,
            Dsrc_alloc_dim.x * sizeof(float4),
            src->Mat.ptr,
            src->pitch * sizeof(de::Half),
            src->width * sizeof(de::Half),
            src->height,
            cudaMemcpyDeviceToDevice,
            S->get_raw_stream_ref()));                            // copy the datas of src from host to device

        hconv2_kernel_within8x8(Dsrc, Ddst, src_diff, Dsrc_alloc_dim, Ddst_alloc_dim, ker_dim, S, flag);
    }

    checkCudaErrors(cudaMemcpy2DAsync(dst->Mat.ptr,        // copy the datas of src from device to host
        dst->pitch * sizeof(de::Half),
        Ddst,
        Ddst_alloc_dim.x * sizeof(float4),
        dst->width * sizeof(de::Half),
        dst->height,
        cudaMemcpyDeviceToHost,
        S->get_raw_stream_ref()));

    checkCudaErrors(cudaDeviceSynchronize());

    decx::alloc::_device_dealloc(&dev_tmp_1);
    decx::alloc::_device_dealloc(&dev_tmp_2);
    S->detach();
}


    

static void decx::_dev_Conv2_NB_R16x16_fp16(decx::_GPU_Matrix* src, decx::_GPU_Matrix* kernel, decx::_GPU_Matrix* dst, 
    de::DH* handle, const int flag)
{
    float4* Dsrc,
        * Ddst;

    int2 Dsrc_alloc_dim,
        Ddst_alloc_dim,
        ker_dim;

    ker_dim.x = kernel->width;            ker_dim.y = kernel->height;
    // first, allocate the memory according to R8 alignments
    Ddst_alloc_dim.x = decx::utils::ceil<int>(dst->width, 64) * 16;
    Ddst_alloc_dim.y = decx::utils::ceil<int>(dst->height, 16) * 16;

    Dsrc_alloc_dim.x = Ddst_alloc_dim.x + bounded_kernel_R16 / 4;        // bounded_kernel_R16 * 2 / 4
    Dsrc_alloc_dim.y = Ddst_alloc_dim.y + bounded_kernel_R16 * 2;

    size_t dev_src_size = static_cast<size_t>(Dsrc_alloc_dim.x) * static_cast<size_t>(Dsrc_alloc_dim.y);
    size_t dev_dst_size = static_cast<size_t>(Ddst_alloc_dim.x) * static_cast<size_t>(Ddst_alloc_dim.y);


    decx::PtrInfo<float4> dev_tmp_1, dev_tmp_2;
    if (decx::alloc::_device_malloc(&dev_tmp_1, dev_src_size * sizeof(float4))) {
        Print_Error_Message(4, DEV_ALLOC_FAIL);
        decx::err::device_AllocateFailure(handle);
        return;
    }
    if (decx::alloc::_device_malloc(&dev_tmp_2, dev_dst_size * sizeof(float4))) {
        Print_Error_Message(4, DEV_ALLOC_FAIL);
        decx::err::device_AllocateFailure(handle);
        return;
    }

    Dsrc = dev_tmp_1.ptr;
    Ddst = dev_tmp_2.ptr;

    decx::cuda_stream* S = NULL;
    S = decx::CStream.stream_accessor_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        Print_Error_Message(4, CUDA_STREAM_ACCESS_FAIL);
        decx::err::CUDA_Stream_access_fail(handle);
        return;
    }

    uint offset_lin = 0, offset_ker = 0;

    for (int i = 0; i < kernel->height; ++i) {
        cudaMemcpyToSymbolAsync(Const_Mem,
            (de::Half*)kernel->Mat.ptr + offset_ker,
            kernel->width * sizeof(de::Half),
            offset_lin * sizeof(de::Half), cudaMemcpyDeviceToDevice, S->get_raw_stream_ref());

        offset_lin += kernel->width;
        offset_ker += kernel->pitch;
    }

    if (ker_dim.x == (bounded_kernel_R16 * 2 + 1) && ker_dim.y == (bounded_kernel_R16 * 2 + 1)) {
        checkCudaErrors(cudaMemcpy2DAsync(Dsrc,
            Dsrc_alloc_dim.x * sizeof(float4),
            src->Mat.ptr,
            src->pitch * sizeof(de::Half),
            src->width * sizeof(de::Half),
            src->height,
            cudaMemcpyDeviceToDevice,
            S->get_raw_stream_ref()));                            // copy the datas of src from host to device

        hconv2_kernel_exact16x16(Dsrc, Ddst, Dsrc_alloc_dim, Ddst_alloc_dim, ker_dim, S, flag);
    }
    else {
        int2 src_diff;
        src_diff.x = bounded_kernel_R16 - ker_dim.y / 2;                src_diff.y = bounded_kernel_R16 - ker_dim.x / 2;
        checkCudaErrors(cudaMemcpy2DAsync(
            reinterpret_cast<de::Half*>(Dsrc) + Dsrc_alloc_dim.x * ker_dim.y * 2 + ker_dim.x / 2,
            Dsrc_alloc_dim.x * sizeof(float4),
            src->Mat.ptr,
            src->pitch * sizeof(de::Half),
            src->width * sizeof(de::Half),
            src->height,
            cudaMemcpyDeviceToDevice,
            S->get_raw_stream_ref()));                            // copy the datas of src from host to device

        hconv2_kernel_within16x16(Dsrc, Ddst, src_diff, Dsrc_alloc_dim, Ddst_alloc_dim, ker_dim, S, flag);
    }

    checkCudaErrors(cudaMemcpy2DAsync(dst->Mat.ptr,        // copy the datas of src from device to host
        dst->pitch * sizeof(de::Half),
        Ddst,
        Ddst_alloc_dim.x * sizeof(float4),
        dst->width * sizeof(de::Half),
        dst->height,
        cudaMemcpyDeviceToDevice,
        S->get_raw_stream_ref()));

    checkCudaErrors(cudaDeviceSynchronize());

    decx::alloc::_device_dealloc(&dev_tmp_1);
    decx::alloc::_device_dealloc(&dev_tmp_2);
    S->detach();
}




static void decx::_dev_Conv2_NB_R8x16_fp16(decx::_GPU_Matrix* src, decx::_GPU_Matrix* kernel, decx::_GPU_Matrix* dst, 
    de::DH* handle, const int flag)
{
    float4* Dsrc,
        * Ddst;

    int2 Dsrc_alloc_dim,
        Ddst_alloc_dim,
        ker_dim;

    ker_dim.x = kernel->width;            ker_dim.y = kernel->height;
    // first, allocate the memory according to R8 alignments
    Ddst_alloc_dim.x = decx::utils::ceil<int>(dst->width, 64) * 16;
    Ddst_alloc_dim.y = decx::utils::ceil<int>(dst->height, 16) * 16;

    Dsrc_alloc_dim.x = Ddst_alloc_dim.x + bounded_kernel_R16 / 4;        // bounded_kernel_R16 * 2 / 4
    Dsrc_alloc_dim.y = Ddst_alloc_dim.y + bounded_kernel_R8 * 2;

    size_t dev_src_size = static_cast<size_t>(Dsrc_alloc_dim.x) * static_cast<size_t>(Dsrc_alloc_dim.y);
    size_t dev_dst_size = static_cast<size_t>(Ddst_alloc_dim.x) * static_cast<size_t>(Ddst_alloc_dim.y);

    decx::PtrInfo<float4> dev_tmp_1, dev_tmp_2;
    if (decx::alloc::_device_malloc(&dev_tmp_1, dev_src_size * sizeof(float4))) {
        Print_Error_Message(4, DEV_ALLOC_FAIL);
        decx::err::device_AllocateFailure(handle);
        return;
    }
    if (decx::alloc::_device_malloc(&dev_tmp_2, dev_dst_size * sizeof(float4))) {
        Print_Error_Message(4, DEV_ALLOC_FAIL);
        decx::err::device_AllocateFailure(handle);
        return;
    }

    Dsrc = dev_tmp_1.ptr;
    Ddst = dev_tmp_2.ptr;

    decx::cuda_stream* S = NULL;
    S = decx::CStream.stream_accessor_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        Print_Error_Message(4, CUDA_STREAM_ACCESS_FAIL);
        decx::err::CUDA_Stream_access_fail(handle);
        return;
    }

    uint offset_lin = 0, offset_ker = 0;

    for (int i = 0; i < kernel->height; ++i) {
        cudaMemcpyToSymbolAsync(Const_Mem,
            (de::Half*)kernel->Mat.ptr + offset_ker,
            kernel->width * sizeof(de::Half),
            offset_lin * sizeof(de::Half), cudaMemcpyDeviceToDevice, S->get_raw_stream_ref());

        offset_lin += kernel->width;
        offset_ker += kernel->pitch;
    }

    if (ker_dim.x == (bounded_kernel_R16 * 2 + 1) && ker_dim.y == (bounded_kernel_R8 * 2 + 1)) {
        checkCudaErrors(cudaMemcpy2DAsync(Dsrc,
            Dsrc_alloc_dim.x * sizeof(float4),
            src->Mat.ptr,
            src->pitch * sizeof(de::Half),
            src->width * sizeof(de::Half),
            src->height,
            cudaMemcpyDeviceToDevice,
            S->get_raw_stream_ref()));                            // copy the datas of src from host to device

        hconv2_kernel_exact8x16(Dsrc, Ddst, Dsrc_alloc_dim, Ddst_alloc_dim, ker_dim, S, flag);
    }
    else {
        int2 src_diff;
        src_diff.x = bounded_kernel_R8 - ker_dim.y / 2;                src_diff.y = bounded_kernel_R16 - ker_dim.x / 2;
        checkCudaErrors(cudaMemcpy2DAsync(
            reinterpret_cast<de::Half*>(Dsrc) + src_diff.x * Dsrc_alloc_dim.x * 4 + src_diff.y,
            Dsrc_alloc_dim.x * sizeof(float4),
            src->Mat.ptr,
            src->pitch * sizeof(de::Half),
            src->width * sizeof(de::Half),
            src->height,
            cudaMemcpyDeviceToDevice,
            S->get_raw_stream_ref()));                            // copy the datas of src from host to device

        hconv2_kernel_within8x16(Dsrc, Ddst, src_diff, Dsrc_alloc_dim, Ddst_alloc_dim, ker_dim, S, flag);
    }

    checkCudaErrors(cudaMemcpy2DAsync(dst->Mat.ptr,        // copy the datas of src from device to host
        dst->pitch * sizeof(de::Half),
        Ddst,
        Ddst_alloc_dim.x * sizeof(float4),
        dst->width * sizeof(de::Half),
        dst->height,
        cudaMemcpyDeviceToDevice,
        S->get_raw_stream_ref()));

    checkCudaErrors(cudaDeviceSynchronize());

    decx::alloc::_device_dealloc(&dev_tmp_1);
    decx::alloc::_device_dealloc(&dev_tmp_2);
    S->detach();
}




static void decx::_dev_Conv2_NB_R16x8_fp16(decx::_GPU_Matrix* src, decx::_GPU_Matrix* kernel, decx::_GPU_Matrix* dst,
    de::DH* handle, const int flag)
{
    float4* Dsrc,
        * Ddst;

    int2 Dsrc_alloc_dim,
        Ddst_alloc_dim,
        ker_dim;

    ker_dim.x = kernel->width;            ker_dim.y = kernel->height;
    // first, allocate the memory according to R8 alignments
    Ddst_alloc_dim.x = decx::utils::ceil<int>(dst->width, 64) * 16;
    Ddst_alloc_dim.y = decx::utils::ceil<int>(dst->height, 16) * 16;

    Dsrc_alloc_dim.x = Ddst_alloc_dim.x + bounded_kernel_R8 / 4;        // bounded_kernel_R16 * 2 / 4
    Dsrc_alloc_dim.y = Ddst_alloc_dim.y + bounded_kernel_R16 * 2;

    size_t dev_src_size = static_cast<size_t>(Dsrc_alloc_dim.x) * static_cast<size_t>(Dsrc_alloc_dim.y);
    size_t dev_dst_size = static_cast<size_t>(Ddst_alloc_dim.x) * static_cast<size_t>(Ddst_alloc_dim.y);

    decx::PtrInfo<float4> dev_tmp_1, dev_tmp_2;
    if (decx::alloc::_device_malloc(&dev_tmp_1, dev_src_size * sizeof(float4))) {
        Print_Error_Message(4, DEV_ALLOC_FAIL);
        decx::err::device_AllocateFailure(handle);
        return;
    }
    if (decx::alloc::_device_malloc(&dev_tmp_2, dev_dst_size * sizeof(float4))) {
        Print_Error_Message(4, DEV_ALLOC_FAIL);
        decx::err::device_AllocateFailure(handle);
        return;
    }

    Dsrc = dev_tmp_1.ptr;
    Ddst = dev_tmp_2.ptr;

    decx::cuda_stream* S = NULL;
    S = decx::CStream.stream_accessor_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        Print_Error_Message(4, CUDA_STREAM_ACCESS_FAIL);
        decx::err::CUDA_Stream_access_fail(handle);
        return;
    }

    uint offset_lin = 0, offset_ker = 0;

    for (int i = 0; i < kernel->height; ++i) {
        cudaMemcpyToSymbolAsync(Const_Mem,
            (de::Half*)kernel->Mat.ptr + offset_ker,
            kernel->width * sizeof(de::Half),
            offset_lin * sizeof(de::Half), cudaMemcpyDeviceToDevice, S->get_raw_stream_ref());

        offset_lin += kernel->width;
        offset_ker += kernel->pitch;
    }

    if (ker_dim.x == (bounded_kernel_R8 * 2 + 1) && ker_dim.y == (bounded_kernel_R16 * 2 + 1)) {
        checkCudaErrors(cudaMemcpy2DAsync(Dsrc,
            Dsrc_alloc_dim.x * sizeof(float4),
            src->Mat.ptr,
            src->pitch * sizeof(de::Half),
            src->width * sizeof(de::Half),
            src->height,
            cudaMemcpyDeviceToDevice,
            S->get_raw_stream_ref()));                            // copy the datas of src from host to device

        hconv2_kernel_exact16x8(Dsrc, Ddst, Dsrc_alloc_dim, Ddst_alloc_dim, ker_dim, S, flag);
    }
    else {
        int2 src_diff;
        src_diff.x = bounded_kernel_R16 - ker_dim.y / 2;                src_diff.y = bounded_kernel_R8 - ker_dim.x / 2;
        checkCudaErrors(cudaMemcpy2DAsync(
            reinterpret_cast<de::Half*>(Dsrc) + src_diff.x * Dsrc_alloc_dim.x * 4 + src_diff.y,
            Dsrc_alloc_dim.x * sizeof(float4),
            src->Mat.ptr,
            src->pitch * sizeof(de::Half),
            src->width * sizeof(de::Half),
            src->height,
            cudaMemcpyDeviceToDevice,
            S->get_raw_stream_ref()));                            // copy the datas of src from host to device

        hconv2_kernel_within16x8(Dsrc, Ddst, src_diff, Dsrc_alloc_dim, Ddst_alloc_dim, ker_dim, S, flag);
    }

    checkCudaErrors(cudaMemcpy2DAsync(dst->Mat.ptr,        // copy the datas of src from device to host
        dst->pitch * sizeof(de::Half),
        Ddst,
        Ddst_alloc_dim.x * sizeof(float4),
        dst->width * sizeof(de::Half),
        dst->height,
        cudaMemcpyDeviceToDevice,
        S->get_raw_stream_ref()));

    checkCudaErrors(cudaDeviceSynchronize());

    decx::alloc::_device_dealloc(&dev_tmp_1);
    decx::alloc::_device_dealloc(&dev_tmp_2);
    S->detach();
}




static void decx::dev_hConv2_border_ignore(decx::_GPU_Matrix& src, decx::_GPU_Matrix& kernel, decx::_GPU_Matrix& dst, 
    de::DH* handle, const int flag)
{
    int2 half_ker_dim;
    half_ker_dim.x = kernel.width / 2;                half_ker_dim.y = kernel.height / 2;

    /*const uint2 dst_dim = make_uint2(src.width - (half_ker_dim.x * 2),
                                     src.height - (half_ker_dim.y * 2));*/

    dst.re_construct(src.type, src.width - (half_ker_dim.x * 2), src.height - (half_ker_dim.y * 2));

    if (half_ker_dim.x < bounded_kernel_R8 + 1) {
        if (half_ker_dim.y < bounded_kernel_R8 + 1) {
            decx::_dev_Conv2_NB_R8x8_fp16(&src, &kernel, &dst, handle, flag);
        }
        else {
            decx::_dev_Conv2_NB_R16x8_fp16(&src, &kernel, &dst, handle, flag);
        }
    }
    else {
        if (half_ker_dim.y < bounded_kernel_R8 + 1) {
            decx::_dev_Conv2_NB_R8x16_fp16(&src, &kernel, &dst, handle, flag);
        }
        else {
            decx::_dev_Conv2_NB_R16x16_fp16(&src, &kernel, &dst, handle, flag);
        }
    }
}


#endif        //    #ifndef _DEV_CONV2_BORDER_IGNORED_H_