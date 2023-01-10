/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/

#ifndef _CONV2_BORDER_CONST_FP32_H_
#define _CONV2_BORDER_CONST_FP32_H_

#include "../../../../core/basic.h"
#include "../../../../classes/Matrix.h"
#include "sconv2_kernel_callers.h"


//using decx::_Matrix;
using decx::alloc::MIF;


namespace decx
{
    /*\
    * 8 x 8
    * This function complete a comand of a convolutipon kernel, which is in type of border-ignored
    * In order to save the device memory, just allocate memory of suitable size
    */
    static void _Conv2_BC_R8x8(decx::_Matrix* src, decx::_Matrix* kernel, decx::_Matrix* dst, de::DH* handle);


    /*\
    * 8 x 16 (h x w)
    * This function complete a comand of a convolutipon kernel, which is in type of border-ignored
    * In order to save the device memory, just allocate memory of suitable size
    */
    static void _Conv2_BC_R8x16(decx::_Matrix* src, decx::_Matrix* kernel, decx::_Matrix* dst, de::DH* handle);


    /*\
    * 16 x 8 (h x w)
    * This function complete a comand of a convolutipon kernel, which is in type of border-ignored
    * In order to save the device memory, just allocate memory of suitable size
    */
    static void _Conv2_BC_R16x8(decx::_Matrix* src, decx::_Matrix* kernel, decx::_Matrix* dst, de::DH* handle);


    /*\
    * 16 x 16
    * This function complete a comand of a convolutipon kernel, which is in type of border-ignored
    * In order to save the device memory, just allocate memory of suitable size
    */
    static void _Conv2_BC_R16x16(decx::_Matrix* src, decx::_Matrix* kernel, decx::_Matrix* dst, de::DH* handle);


    static void sConv2_border_zero(decx::_Matrix& src, decx::_Matrix& kernel, decx::_Matrix& dst, de::DH* handle);
}




static
void decx::_Conv2_BC_R8x8(decx::_Matrix* src, decx::_Matrix* kernel, decx::_Matrix* dst, de::DH *handle)
{
    float4* Dsrc,
        *Ddst;

    int2 Dsrc_alloc_dim,
        Ddst_alloc_dim,
        ker_dim;
    
    ker_dim.x = kernel->width;            ker_dim.y = kernel->height;
    // first, allocate the memory according to R8 alignments
    Ddst_alloc_dim.x = decx::utils::ceil<int>(dst->width, 64) * bounded_kernel_R8 * 2;
    Ddst_alloc_dim.y = decx::utils::ceil<int>(dst->height, 16) * bounded_kernel_R8 * 2;

    Dsrc_alloc_dim.x = Ddst_alloc_dim.x + bounded_kernel_R8 / 2;        // bounded_kernel_R8 * 2 / 4
    Dsrc_alloc_dim.y = Ddst_alloc_dim.y + bounded_kernel_R8 * 2;
    
    size_t dev_src_size = static_cast<size_t>(Dsrc_alloc_dim.x) * static_cast<size_t>(Dsrc_alloc_dim.y);
    size_t dev_dst_size = static_cast<size_t>(Ddst_alloc_dim.x) * static_cast<size_t>(Ddst_alloc_dim.y);

    decx::PtrInfo<float4> dev_tmp1, dev_tmp2;
    decx::cuda_stream* S = NULL;

    S = decx::CStream.stream_accessor_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        Print_Error_Message(4, CUDA_STREAM_ACCESS_FAIL);
        decx::err::CUDA_Stream_access_fail(handle);
        return;
    }

    if (decx::alloc::_device_malloc(&dev_tmp1, dev_src_size * sizeof(float4), S)) {
        Print_Error_Message(4, ALLOC_FAIL);
        decx::err::AllocateFailure(handle);
        return;
    }
    if (decx::alloc::_device_malloc(&dev_tmp2, dev_dst_size * sizeof(float4), S)) {
        Print_Error_Message(4, ALLOC_FAIL);
        decx::err::AllocateFailure(handle);
        return;
    }

    Dsrc = dev_tmp1.ptr;
    Ddst = dev_tmp2.ptr;
    uint offset_lin = 0, offset_ker = 0;
    
    for (int i = 0; i < kernel->height; ++i) {
        cudaMemcpyToSymbolAsync(Const_Mem,
            (float*)kernel->Mat.ptr + offset_ker,
            kernel->width * sizeof(float),
            offset_lin * sizeof(float), cudaMemcpyHostToDevice, S->get_raw_stream_ref());

        offset_lin += kernel->width;
        offset_ker += kernel->pitch;
    }

    checkCudaErrors(cudaMemcpy2DAsync(
        reinterpret_cast<float*>(Dsrc) + Dsrc_alloc_dim.x * bounded_kernel_R8 * 4 + bounded_kernel_R8,
        Dsrc_alloc_dim.x * sizeof(float4),
        src->Mat.ptr,
        src->pitch * sizeof(float),
        src->width * sizeof(float),
        src->height,
        cudaMemcpyHostToDevice,
        S->get_raw_stream_ref()));                            // copy the datas of src from host to device

    if (ker_dim.x == (bounded_kernel_R8 * 2 + 1) && ker_dim.y == (bounded_kernel_R8 * 2 + 1)) {
        decx::sconv2_kernel_exact8x8(Dsrc, Ddst, Dsrc_alloc_dim, Ddst_alloc_dim, ker_dim, S);
    }
    else {
        int2 src_diff;
        src_diff.x = bounded_kernel_R8 - ker_dim.y / 2;                src_diff.y = bounded_kernel_R8 - ker_dim.x / 2;
        
        decx::sconv2_kernel_within8x8(Dsrc, Ddst, src_diff, Dsrc_alloc_dim, Ddst_alloc_dim, ker_dim, S);
    }
    
    checkCudaErrors(cudaMemcpy2DAsync(dst->Mat.ptr,        // copy the datas of src from device to host
        dst->pitch * sizeof(float),
        Ddst,
        Ddst_alloc_dim.x * sizeof(float4),
        dst->width * sizeof(float),
        dst->height,
        cudaMemcpyDeviceToHost,
        S->get_raw_stream_ref()));

    checkCudaErrors(cudaDeviceSynchronize());

    decx::alloc::_device_dealloc(&dev_tmp1);
    decx::alloc::_device_dealloc(&dev_tmp2);
    S->detach();
}




static
void decx::_Conv2_BC_R8x16(decx::_Matrix* src, decx::_Matrix* kernel, decx::_Matrix* dst, de::DH *handle)
{
    float4* Dsrc,
        *Ddst;

    int2 Dsrc_alloc_dim,
        Ddst_alloc_dim,
        ker_dim;
    
    ker_dim.x = kernel->width;            ker_dim.y = kernel->height;
    // first, allocate the memory according to R8 alignments
    Ddst_alloc_dim.x = decx::utils::ceil<int>(dst->width, 64) * 16;
    Ddst_alloc_dim.y = decx::utils::ceil<int>(dst->height, 16) * 16;

    Dsrc_alloc_dim.x = Ddst_alloc_dim.x + bounded_kernel_R16 / 2;        // bounded_kernel_R16 * 2 / 4
    Dsrc_alloc_dim.y = Ddst_alloc_dim.y + bounded_kernel_R8 * 2;
    
    size_t dev_src_size = static_cast<size_t>(Dsrc_alloc_dim.x) * static_cast<size_t>(Dsrc_alloc_dim.y);
    size_t dev_dst_size = static_cast<size_t>(Ddst_alloc_dim.x) * static_cast<size_t>(Ddst_alloc_dim.y);

    decx::PtrInfo<float4> dev_tmp1, dev_tmp2;
    decx::cuda_stream* S = NULL;

    S = decx::CStream.stream_accessor_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        Print_Error_Message(4, CUDA_STREAM_ACCESS_FAIL);
        decx::err::CUDA_Stream_access_fail(handle);
        return;
    }

    if (decx::alloc::_device_malloc(&dev_tmp1, dev_src_size * sizeof(float4), S)) {
        Print_Error_Message(4, ALLOC_FAIL);
        decx::err::AllocateFailure(handle);
        return;
    }
    if (decx::alloc::_device_malloc(&dev_tmp2, dev_dst_size * sizeof(float4), S)) {
        Print_Error_Message(4, ALLOC_FAIL);
        decx::err::AllocateFailure(handle);
        return;
    }

    Dsrc = dev_tmp1.ptr;
    Ddst = dev_tmp2.ptr;

    uint offset_lin = 0, offset_ker = 0;
    
    for (int i = 0; i < kernel->height; ++i) {
        cudaMemcpyToSymbolAsync(Const_Mem,
            (float*)kernel->Mat.ptr + offset_ker,
            kernel->width * sizeof(float),
            offset_lin * sizeof(float), cudaMemcpyHostToDevice, S->get_raw_stream_ref());

        offset_lin += kernel->width;
        offset_ker += kernel->pitch;
    }

    checkCudaErrors(cudaMemcpy2DAsync(
        reinterpret_cast<float*>(Dsrc) + bounded_kernel_R8 * Dsrc_alloc_dim.x * 4 + bounded_kernel_R16,
        Dsrc_alloc_dim.x * sizeof(float4),
        src->Mat.ptr,
        src->pitch * sizeof(float),
        src->width * sizeof(float),
        src->height,
        cudaMemcpyHostToDevice,
        S->get_raw_stream_ref()));                            // copy the datas of src from host to device

    if (ker_dim.x == (bounded_kernel_R16 * 2 + 1) && ker_dim.y == (bounded_kernel_R8 * 2 + 1)) {
        decx::sconv2_kernel_exact8x16(Dsrc, Ddst, Dsrc_alloc_dim, Ddst_alloc_dim, ker_dim, S);
    }
    else {
        int2 src_diff;
        src_diff.x = bounded_kernel_R8 - ker_dim.y / 2;                src_diff.y = bounded_kernel_R16 - ker_dim.x / 2;
        
        decx::sconv2_kernel_within8x16(Dsrc, Ddst, src_diff, Dsrc_alloc_dim, Ddst_alloc_dim, ker_dim, S);
    }
    
    checkCudaErrors(cudaMemcpy2DAsync(dst->Mat.ptr,        // copy the datas of src from device to host
        dst->pitch * sizeof(float),
        Ddst,
        Ddst_alloc_dim.x * sizeof(float4),
        dst->width * sizeof(float),
        dst->height,
        cudaMemcpyDeviceToHost,
        S->get_raw_stream_ref()));

    checkCudaErrors(cudaDeviceSynchronize());

    decx::alloc::_device_dealloc(&dev_tmp1);
    decx::alloc::_device_dealloc(&dev_tmp2);
    S->detach();
}





static
void decx::_Conv2_BC_R16x8(decx::_Matrix* src, decx::_Matrix* kernel, decx::_Matrix* dst, de::DH *handle)
{
    float4* Dsrc,
        *Ddst;

    int2 Dsrc_alloc_dim,
        Ddst_alloc_dim,
        ker_dim;
    
    ker_dim.x = kernel->width;            ker_dim.y = kernel->height;
    // first, allocate the memory according to R8 alignments
    Ddst_alloc_dim.x = decx::utils::ceil<int>(dst->width, 64) * 16;
    Ddst_alloc_dim.y = decx::utils::ceil<int>(dst->height, 16) * 16;

    Dsrc_alloc_dim.x = Ddst_alloc_dim.x + bounded_kernel_R8 / 2;        // bounded_kernel_R16 * 2 / 4
    Dsrc_alloc_dim.y = Ddst_alloc_dim.y + bounded_kernel_R16 * 2;
    
    size_t dev_src_size = static_cast<size_t>(Dsrc_alloc_dim.x) * static_cast<size_t>(Dsrc_alloc_dim.y);
    size_t dev_dst_size = static_cast<size_t>(Ddst_alloc_dim.x) * static_cast<size_t>(Ddst_alloc_dim.y);

    decx::PtrInfo<float4> dev_tmp1, dev_tmp2;
    decx::cuda_stream* S = NULL;

    S = decx::CStream.stream_accessor_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        Print_Error_Message(4, CUDA_STREAM_ACCESS_FAIL);
        decx::err::CUDA_Stream_access_fail(handle);
        return;
    }

    if (decx::alloc::_device_malloc(&dev_tmp1, dev_src_size * sizeof(float4), S)) {
        Print_Error_Message(4, ALLOC_FAIL);
        decx::err::AllocateFailure(handle);
        return;
    }
    if (decx::alloc::_device_malloc(&dev_tmp2, dev_dst_size * sizeof(float4), S)) {
        Print_Error_Message(4, ALLOC_FAIL);
        decx::err::AllocateFailure(handle);
        return;
    }

    Dsrc = dev_tmp1.ptr;
    Ddst = dev_tmp2.ptr;

    uint offset_lin = 0, offset_ker = 0;
    
    for (uint i = 0; i < kernel->height; ++i) {
        cudaMemcpyToSymbolAsync(Const_Mem,
            (float*)kernel->Mat.ptr + offset_ker,
            kernel->width * sizeof(float),
            offset_lin * sizeof(float), cudaMemcpyHostToDevice, S->get_raw_stream_ref());

        offset_lin += kernel->width;
        offset_ker += kernel->pitch;
    }

    checkCudaErrors(cudaMemcpy2DAsync(
        reinterpret_cast<float*>(Dsrc) + bounded_kernel_R16 * Dsrc_alloc_dim.x * 4 + bounded_kernel_R8,
        Dsrc_alloc_dim.x * sizeof(float4),
        src->Mat.ptr,
        src->pitch * sizeof(float),
        src->width * sizeof(float),
        src->height,
        cudaMemcpyHostToDevice,
        S->get_raw_stream_ref()));                            // copy the datas of src from host to device

    if (ker_dim.x == (bounded_kernel_R8 * 2 + 1) && ker_dim.y == (bounded_kernel_R16 * 2 + 1)) {
        decx::sconv2_kernel_exact16x8(Dsrc, Ddst, Dsrc_alloc_dim, Ddst_alloc_dim, ker_dim, S);
    }
    else {
        int2 src_diff;
        src_diff.x = bounded_kernel_R16 - ker_dim.y / 2;                src_diff.y = bounded_kernel_R8 - ker_dim.x / 2;
        
        decx::sconv2_kernel_within16x8(Dsrc, Ddst, src_diff, Dsrc_alloc_dim, Ddst_alloc_dim, ker_dim, S);
    }
    
    checkCudaErrors(cudaMemcpy2DAsync(dst->Mat.ptr,        // copy the datas of src from device to host
        dst->pitch * sizeof(float),
        Ddst,
        Ddst_alloc_dim.x * sizeof(float4),
        dst->width * sizeof(float),
        dst->height,
        cudaMemcpyDeviceToHost,
        S->get_raw_stream_ref()));

    checkCudaErrors(cudaDeviceSynchronize());

    decx::alloc::_device_dealloc(&dev_tmp1);
    decx::alloc::_device_dealloc(&dev_tmp2);
    S->detach();
}




static
void decx::_Conv2_BC_R16x16(decx::_Matrix* src, decx::_Matrix* kernel, decx::_Matrix* dst, de::DH *handle)
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

    Dsrc_alloc_dim.x = Ddst_alloc_dim.x + bounded_kernel_R16 / 2;        // bounded_kernel_R16 * 2 / 4
    Dsrc_alloc_dim.y = Ddst_alloc_dim.y + bounded_kernel_R16 * 2;

    size_t dev_src_size = static_cast<size_t>(Dsrc_alloc_dim.x) * static_cast<size_t>(Dsrc_alloc_dim.y);
    size_t dev_dst_size = static_cast<size_t>(Ddst_alloc_dim.x) * static_cast<size_t>(Ddst_alloc_dim.y);

    decx::PtrInfo<float4> dev_tmp1, dev_tmp2;
    decx::cuda_stream* S = NULL;

    S = decx::CStream.stream_accessor_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        Print_Error_Message(4, CUDA_STREAM_ACCESS_FAIL);
        decx::err::CUDA_Stream_access_fail(handle);
        return;
    }

    if (decx::alloc::_device_malloc(&dev_tmp1, dev_src_size * sizeof(float4), S)) {
        Print_Error_Message(4, ALLOC_FAIL);
        decx::err::AllocateFailure(handle);
        return;
    }
    if (decx::alloc::_device_malloc(&dev_tmp2, dev_dst_size * sizeof(float4), S)) {
        Print_Error_Message(4, ALLOC_FAIL);
        decx::err::AllocateFailure(handle);
        return;
    }

    Dsrc = dev_tmp1.ptr;
    Ddst = dev_tmp2.ptr;

    uint offset_lin = 0, offset_ker = 0;

    for (uint i = 0; i < kernel->height; ++i) {
        cudaMemcpyToSymbolAsync(Const_Mem,
            (float*)kernel->Mat.ptr + offset_ker,
            kernel->width * sizeof(float),
            offset_lin * sizeof(float), cudaMemcpyHostToDevice, S->get_raw_stream_ref());

        offset_lin += kernel->width;
        offset_ker += kernel->pitch;
    }

    checkCudaErrors(cudaMemcpy2DAsync(
        reinterpret_cast<float*>(Dsrc) + Dsrc_alloc_dim.x * bounded_kernel_R16 * 4 + bounded_kernel_R16,
        Dsrc_alloc_dim.x * sizeof(float4),
        src->Mat.ptr,
        src->pitch * sizeof(float),
        src->width * sizeof(float),
        src->height,
        cudaMemcpyHostToDevice,
        S->get_raw_stream_ref()));                            // copy the datas of src from host to device

    if (ker_dim.x == (bounded_kernel_R16 * 2 + 1) && ker_dim.y == (bounded_kernel_R16 * 2 + 1)) {    
        decx::sconv2_kernel_exact16x16(Dsrc, Ddst, Dsrc_alloc_dim, Ddst_alloc_dim, ker_dim, S);
    }
    else {
        int2 src_diff;
        src_diff.x = bounded_kernel_R16 - ker_dim.y / 2;                src_diff.y = bounded_kernel_R16 - ker_dim.x / 2;
        
        decx::sconv2_kernel_within16x16(Dsrc, Ddst, src_diff, Dsrc_alloc_dim, Ddst_alloc_dim, ker_dim, S);
    }

    checkCudaErrors(cudaMemcpy2DAsync(dst->Mat.ptr,        // copy the datas of src from device to host
        dst->pitch * sizeof(float),
        Ddst,
        Ddst_alloc_dim.x * sizeof(float4),
        dst->width * sizeof(float),
        dst->height,
        cudaMemcpyDeviceToHost,
        S->get_raw_stream_ref()));

    checkCudaErrors(cudaDeviceSynchronize());

    decx::alloc::_device_dealloc(&dev_tmp1);
    decx::alloc::_device_dealloc(&dev_tmp2);
    S->detach();
}




static 
void decx::sConv2_border_zero(decx::_Matrix& src, decx::_Matrix& kernel, decx::_Matrix& dst, de::DH* handle)
{
    int2 half_ker_dim;
    half_ker_dim.x = kernel.width / 2;                half_ker_dim.y = kernel.height / 2;
    
    dst.re_construct(src.type, src.width,
        src.height,
        decx::DATA_STORE_TYPE::Page_Locked);
    
    if (half_ker_dim.x < bounded_kernel_R8 + 1) {
        if (half_ker_dim.y < bounded_kernel_R8 + 1) {
            decx::_Conv2_BC_R8x8(&src, &kernel, &dst, handle);
        }
        else {
            decx::_Conv2_BC_R16x8(&src, &kernel, &dst, handle);
        }
    }
    else {
        if (half_ker_dim.y < bounded_kernel_R8 + 1) {
            decx::_Conv2_BC_R8x16(&src, &kernel, &dst, handle);
        }
        else {
            decx::_Conv2_BC_R16x16(&src, &kernel, &dst, handle);
        }
    }
}


#endif