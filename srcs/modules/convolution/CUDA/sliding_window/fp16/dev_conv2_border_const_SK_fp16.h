/**
*    ---------------------------------------------------------------------
*    Author : Wayne Anderson
*    Date   : 2021.04.16
*    ---------------------------------------------------------------------
*    This is a part of the open source program named "DECX", copyright c Wayne,
*    2021.04.16
*/

#ifndef _DEV_CONV2_MC_BORDER_CONST_SK_FP16_H_
#define _DEV_CONV2_MC_BORDER_CONST_SK_FP16_H_


#include "../../../../core/basic.h"
#include "../../../../classes/GPU_Matrix.h"
#include "../../../../classes/GPU_MatrixArray.h"
#include "../Conv2_MC_macros.h"
#include "hconv2_kernel_callers.h"


using decx::_GPU_Matrix;
using decx::alloc::MIF;
using decx::_GPU_MatrixArray;


namespace decx
{
    static void dev_main_loop_hconv2_sk_within8x8_BC(
        int2* Dsrc_alloc_dim,                int2* Ddst_alloc_dim,                int2* ker_dim,
        decx::_GPU_MatrixArray* src,     decx::_GPU_Matrix* kernel,       decx::_GPU_MatrixArray* dst,
        float4* Dsrc,                        float4* Ddst,                        decx::cuda_stream* S, const int flag);



    static void dev_main_loop_hconv2_sk_exact8x8_BC(
        int2* Dsrc_alloc_dim,                int2* Ddst_alloc_dim,                int2* ker_dim,
        decx::_GPU_MatrixArray* src,     decx::_GPU_Matrix* kernel,       decx::_GPU_MatrixArray* dst,
        float4* Dsrc,                        float4* Ddst,                        decx::cuda_stream* S, const int flag);



    static void dev_main_loop_hconv2_sk_within8x16_BC(
        int2* Dsrc_alloc_dim,                int2* Ddst_alloc_dim,                int2* ker_dim,
        decx::_GPU_MatrixArray* src,     decx::_GPU_Matrix* kernel,       decx::_GPU_MatrixArray* dst,
        float4* Dsrc,                        float4* Ddst,                        decx::cuda_stream* S, const int flag);


    static void dev_main_loop_hconv2_sk_exact8x16_BC(
        int2* Dsrc_alloc_dim,                int2* Ddst_alloc_dim,                int2* ker_dim,
        decx::_GPU_MatrixArray* src,     decx::_GPU_Matrix* kernel,       decx::_GPU_MatrixArray* dst,
        float4* Dsrc,                        float4* Ddst,                        decx::cuda_stream* S, const int flag);


    static void dev_main_loop_hconv2_sk_within16x8_BC(
        int2* Dsrc_alloc_dim,                int2* Ddst_alloc_dim,                int2* ker_dim,
        decx::_GPU_MatrixArray* src,     decx::_GPU_Matrix* kernel,       decx::_GPU_MatrixArray* dst,
        float4* Dsrc,                        float4* Ddst,                        decx::cuda_stream* S, const int flag);


    static void dev_main_loop_hconv2_sk_exact16x8_BC(
        int2* Dsrc_alloc_dim,                int2* Ddst_alloc_dim,                int2* ker_dim,
        decx::_GPU_MatrixArray* src,     decx::_GPU_Matrix* kernel,       decx::_GPU_MatrixArray* dst,
        float4* Dsrc,                        float4* Ddst,                        decx::cuda_stream* S, const int flag);


    static void dev_main_loop_hconv2_sk_within16x16_BC(
        int2* Dsrc_alloc_dim,                int2* Ddst_alloc_dim,                int2* ker_dim,
        decx::_GPU_MatrixArray* src,     decx::_GPU_Matrix* kernel,       decx::_GPU_MatrixArray* dst,
        float4* Dsrc,                        float4* Ddst,                        decx::cuda_stream* S, const int flag);


    static void dev_main_loop_hconv2_sk_exact16x16_BC(
        int2* Dsrc_alloc_dim,                int2* Ddst_alloc_dim,                int2* ker_dim,
        decx::_GPU_MatrixArray* src,     decx::_GPU_Matrix* kernel,       decx::_GPU_MatrixArray* dst,
        float4* Dsrc,                        float4* Ddst,                        decx::cuda_stream* S, const int flag);
}




static void decx::dev_main_loop_hconv2_sk_within8x8_BC(
        int2* Dsrc_alloc_dim,                int2* Ddst_alloc_dim,                int2* ker_dim,
        decx::_GPU_MatrixArray* src,     decx::_GPU_Matrix* kernel,       decx::_GPU_MatrixArray* dst,
        float4* Dsrc,                        float4* Ddst,                        decx::cuda_stream* S, const int flag)
{
    int2 src_diff;
    // copy the first part to device memory
    src_diff.x = bounded_kernel_R8 - ker_dim->y / 2;                src_diff.y = bounded_kernel_R8 - ker_dim->x / 2;

    // strat the main loop
    for (int i = 0; i < src->ArrayNumber; ++i) {
        checkCudaErrors(cudaMemcpy2DAsync(
            reinterpret_cast<de::Half*>(Dsrc) + Dsrc_alloc_dim->x * bounded_kernel_R8 * 8 + bounded_kernel_R8,
            Dsrc_alloc_dim->x * sizeof(float4),
            src->MatptrArr.ptr[i],
            src->pitch * sizeof(de::Half),
            src->width * sizeof(de::Half),
            src->height,
            cudaMemcpyDeviceToDevice,
            S->get_raw_stream_ref()));                            // copy the datas of src from host to device

        hconv2_kernel_within8x8(Dsrc, Ddst, src_diff, *Dsrc_alloc_dim, *Ddst_alloc_dim, *ker_dim, S, flag);

        checkCudaErrors(cudaMemcpy2DAsync(dst->MatptrArr.ptr[i],
            dst->pitch * sizeof(de::Half), Ddst, Ddst_alloc_dim->x * sizeof(float4), dst->width * sizeof(de::Half),
            dst->height, cudaMemcpyDeviceToDevice, S->get_raw_stream_ref()));
    }
}




static void decx::dev_main_loop_hconv2_sk_exact8x8_BC(
        int2* Dsrc_alloc_dim,                int2* Ddst_alloc_dim,                int2* ker_dim,
        decx::_GPU_MatrixArray* src,     decx::_GPU_Matrix* kernel,       decx::_GPU_MatrixArray* dst,
        float4* Dsrc,                        float4* Ddst,                        decx::cuda_stream* S, const int flag)
{
    // strat the main loop
    for (int i = 0; i < src->ArrayNumber; ++i) {
        checkCudaErrors(cudaMemcpy2DAsync(
            reinterpret_cast<de::Half*>(Dsrc) + Dsrc_alloc_dim->x * bounded_kernel_R8 * 8 + bounded_kernel_R8,
            Dsrc_alloc_dim->x * sizeof(float4),
            src->MatptrArr.ptr[0],
            src->pitch * sizeof(de::Half),
            src->width * sizeof(de::Half),
            src->height,
            cudaMemcpyDeviceToDevice,
            S->get_raw_stream_ref()));                            // copy the datas of src from host to device

        hconv2_kernel_exact8x8(Dsrc, Ddst, *Dsrc_alloc_dim, *Ddst_alloc_dim, *ker_dim, S, flag);

        checkCudaErrors(cudaMemcpy2DAsync(dst->MatptrArr.ptr[i],
            dst->pitch * sizeof(de::Half), Ddst, Ddst_alloc_dim->x * sizeof(float4), dst->width * sizeof(de::Half),
            dst->height, cudaMemcpyDeviceToDevice, S->get_raw_stream_ref()));
    }
}





static void decx::dev_main_loop_hconv2_sk_within8x16_BC(
        int2* Dsrc_alloc_dim,                int2* Ddst_alloc_dim,                int2* ker_dim,
        decx::_GPU_MatrixArray* src,     decx::_GPU_Matrix* kernel,       decx::_GPU_MatrixArray* dst,
        float4* Dsrc,                        float4* Ddst,                        decx::cuda_stream* S, const int flag)
{
    int2 src_diff;
    // copy the first part to device memory
    src_diff.x = bounded_kernel_R8 - ker_dim->y / 2;                src_diff.y = bounded_kernel_R16 - ker_dim->x / 2;

    // strat the main loop
    for (int i = 0; i < src->ArrayNumber; ++i) {
        checkCudaErrors(cudaMemcpy2DAsync(
            reinterpret_cast<de::Half*>(Dsrc) + Dsrc_alloc_dim->x * bounded_kernel_R8 * 4 + bounded_kernel_R16,
            Dsrc_alloc_dim->x * sizeof(float4),
            src->MatptrArr.ptr[i],
            src->pitch * sizeof(de::Half),
            src->width * sizeof(de::Half),
            src->height,
            cudaMemcpyDeviceToDevice,
            S->get_raw_stream_ref()));                            // copy the datas of src from host to device

        hconv2_kernel_within8x16(Dsrc, Ddst, src_diff, *Dsrc_alloc_dim, *Ddst_alloc_dim, *ker_dim, S, flag);

        checkCudaErrors(cudaMemcpy2DAsync(dst->MatptrArr.ptr[i],
            dst->pitch * sizeof(de::Half), Ddst, Ddst_alloc_dim->x * sizeof(float4), dst->width * sizeof(de::Half),
            dst->height, cudaMemcpyDeviceToDevice, S->get_raw_stream_ref()));
    }
}




static void decx::dev_main_loop_hconv2_sk_exact8x16_BC(
        int2* Dsrc_alloc_dim,                int2* Ddst_alloc_dim,                int2* ker_dim,
        decx::_GPU_MatrixArray* src,     decx::_GPU_Matrix* kernel,       decx::_GPU_MatrixArray* dst,
        float4* Dsrc,                        float4* Ddst,                        decx::cuda_stream* S, const int flag)
{
    // strat the main loop
    for (int i = 0; i < src->ArrayNumber; ++i) {
        checkCudaErrors(cudaMemcpy2DAsync(
            reinterpret_cast<de::Half*>(Dsrc) + Dsrc_alloc_dim->x * bounded_kernel_R8 * 8 + bounded_kernel_R16,
            Dsrc_alloc_dim->x * sizeof(float4),
            src->MatptrArr.ptr[i],
            src->pitch * sizeof(de::Half),
            src->width * sizeof(de::Half),
            src->height,
            cudaMemcpyDeviceToDevice,
            S->get_raw_stream_ref()));                            // copy the datas of src from host to device

        hconv2_kernel_exact8x16(Dsrc, Ddst, *Dsrc_alloc_dim, *Ddst_alloc_dim, *ker_dim, S, flag);

        checkCudaErrors(cudaMemcpy2DAsync(dst->MatptrArr.ptr[i],
            dst->pitch * sizeof(de::Half), Ddst, Ddst_alloc_dim->x * sizeof(float4), dst->width * sizeof(de::Half),
            dst->height, cudaMemcpyDeviceToDevice, S->get_raw_stream_ref()));
    }
}




static void decx::dev_main_loop_hconv2_sk_within16x8_BC(
        int2* Dsrc_alloc_dim,                int2* Ddst_alloc_dim,                int2* ker_dim,
        decx::_GPU_MatrixArray* src,     decx::_GPU_Matrix* kernel,       decx::_GPU_MatrixArray* dst,
        float4* Dsrc,                        float4* Ddst,                        decx::cuda_stream* S, const int flag)
{
    int2 src_diff;
    // copy the first part to device memory
    src_diff.x = bounded_kernel_R16 - ker_dim->y / 2;                src_diff.y = bounded_kernel_R8 - ker_dim->x / 2;

    // strat the main loop
    for (int i = 0; i < src->ArrayNumber; ++i) {
        checkCudaErrors(cudaMemcpy2DAsync(
            reinterpret_cast<de::Half*>(Dsrc) + Dsrc_alloc_dim->x * bounded_kernel_R16 * 8 + bounded_kernel_R8,
            Dsrc_alloc_dim->x * sizeof(float4),
            src->MatptrArr.ptr[i],
            src->pitch * sizeof(de::Half),
            src->width * sizeof(de::Half),
            src->height,
            cudaMemcpyDeviceToDevice,
            S->get_raw_stream_ref()));                            // copy the datas of src from host to device

        hconv2_kernel_within16x8(Dsrc, Ddst, src_diff, *Dsrc_alloc_dim, *Ddst_alloc_dim, *ker_dim, S, flag);

        checkCudaErrors(cudaMemcpy2DAsync(dst->MatptrArr.ptr[i],
            dst->pitch * sizeof(de::Half), Ddst, Ddst_alloc_dim->x * sizeof(float4), dst->width * sizeof(de::Half),
            dst->height, cudaMemcpyDeviceToDevice, S->get_raw_stream_ref()));
    }
}




static void decx::dev_main_loop_hconv2_sk_exact16x8_BC(
        int2* Dsrc_alloc_dim,                int2* Ddst_alloc_dim,                int2* ker_dim,
        decx::_GPU_MatrixArray* src,     decx::_GPU_Matrix* kernel,       decx::_GPU_MatrixArray* dst,
        float4* Dsrc,                        float4* Ddst,                        decx::cuda_stream* S, const int flag)
{
    // strat the main loop
    for (int i = 0; i < src->ArrayNumber; ++i) {
        checkCudaErrors(cudaMemcpy2DAsync(
            reinterpret_cast<de::Half*>(Dsrc) + Dsrc_alloc_dim->x * bounded_kernel_R16 * 8 + bounded_kernel_R8,
            Dsrc_alloc_dim->x * sizeof(float4),
            src->MatptrArr.ptr[i],
            src->pitch * sizeof(de::Half),
            src->width * sizeof(de::Half),
            src->height,
            cudaMemcpyDeviceToDevice,
            S->get_raw_stream_ref()));                            // copy the datas of src from host to device

        hconv2_kernel_exact8x16(Dsrc, Ddst, *Dsrc_alloc_dim, *Ddst_alloc_dim, *ker_dim, S, flag);

        checkCudaErrors(cudaMemcpy2DAsync(dst->MatptrArr.ptr[i],
            dst->pitch * sizeof(de::Half), Ddst, Ddst_alloc_dim->x * sizeof(float4), dst->width * sizeof(de::Half),
            dst->height, cudaMemcpyDeviceToDevice, S->get_raw_stream_ref()));
    }
}





static void decx::dev_main_loop_hconv2_sk_within16x16_BC(
        int2* Dsrc_alloc_dim,                int2* Ddst_alloc_dim,                int2* ker_dim,
        decx::_GPU_MatrixArray* src,     decx::_GPU_Matrix* kernel,       decx::_GPU_MatrixArray* dst,
        float4* Dsrc,                        float4* Ddst,                        decx::cuda_stream* S, const int flag)
{
    int2 src_diff;
    // copy the first part to device memory
    src_diff.x = bounded_kernel_R16 - ker_dim->y / 2;                src_diff.y = bounded_kernel_R16 - ker_dim->x / 2;

    // strat the main loop
    for (int i = 0; i < src->ArrayNumber; ++i) {
        checkCudaErrors(cudaMemcpy2DAsync(
            reinterpret_cast<de::Half*>(Dsrc) + Dsrc_alloc_dim->x * bounded_kernel_R16 * 8 + bounded_kernel_R16,
            Dsrc_alloc_dim->x * sizeof(float4),
            src->MatptrArr.ptr[i],
            src->pitch * sizeof(de::Half),
            src->width * sizeof(de::Half),
            src->height,
            cudaMemcpyDeviceToDevice,
            S->get_raw_stream_ref()));                            // copy the datas of src from host to device

        hconv2_kernel_within16x8(Dsrc, Ddst, src_diff, *Dsrc_alloc_dim, *Ddst_alloc_dim, *ker_dim, S, flag);

        checkCudaErrors(cudaMemcpy2DAsync(dst->MatptrArr.ptr[i],
            dst->pitch * sizeof(de::Half), Ddst, Ddst_alloc_dim->x * sizeof(float4), dst->width * sizeof(de::Half),
            dst->height, cudaMemcpyDeviceToDevice, S->get_raw_stream_ref()));
    }
}




static void decx::dev_main_loop_hconv2_sk_exact16x16_BC(
        int2* Dsrc_alloc_dim,                int2* Ddst_alloc_dim,                int2* ker_dim,
        decx::_GPU_MatrixArray* src,     decx::_GPU_Matrix* kernel,       decx::_GPU_MatrixArray* dst,
        float4* Dsrc,                        float4* Ddst,                        decx::cuda_stream* S, const int flag)
{
    // strat the main loop
    for (int i = 0; i < src->ArrayNumber; ++i) {
        checkCudaErrors(cudaMemcpy2DAsync(
            reinterpret_cast<de::Half*>(Dsrc) + Dsrc_alloc_dim->x * bounded_kernel_R16 * 8 + bounded_kernel_R16,
            Dsrc_alloc_dim->x * sizeof(float4),
            src->MatptrArr.ptr[i],
            src->pitch * sizeof(de::Half),
            src->width * sizeof(de::Half),
            src->height,
            cudaMemcpyDeviceToDevice,
            S->get_raw_stream_ref()));                            // copy the datas of src from host to device

        hconv2_kernel_exact16x16(Dsrc, Ddst, *Dsrc_alloc_dim, *Ddst_alloc_dim, *ker_dim, S, flag);

        checkCudaErrors(cudaMemcpy2DAsync(dst->MatptrArr.ptr[i],
            dst->pitch * sizeof(de::Half), Ddst, Ddst_alloc_dim->x * sizeof(float4), dst->width * sizeof(de::Half),
            dst->height, cudaMemcpyDeviceToDevice, S->get_raw_stream_ref()));
    }
}



// **************************************************************************************************************************



namespace decx
{
    static void dev_Conv2_BC_R8x8_SK(
            decx::_GPU_MatrixArray* src, decx::_GPU_Matrix* kernel, decx::_GPU_MatrixArray* dst, de::DH *handle, const int flag);


    static void dev_Conv2_BC_R8x16_SK(
            decx::_GPU_MatrixArray* src, decx::_GPU_Matrix* kernel, decx::_GPU_MatrixArray* dst, de::DH *handle, const int flag);


    static void dev_Conv2_BC_R16x8_SK(
            decx::_GPU_MatrixArray* src, decx::_GPU_Matrix* kernel, decx::_GPU_MatrixArray* dst, de::DH *handle, const int flag);


    static void dev_Conv2_BC_R16x16_SK(
            decx::_GPU_MatrixArray* src, decx::_GPU_Matrix* kernel, decx::_GPU_MatrixArray* dst, de::DH *handle, const int flag);


    static void dev_hConv2_border_zero_sk(
            decx::_GPU_MatrixArray* src, decx::_GPU_Matrix* kernel, decx::_GPU_MatrixArray* dst, de::DH* handle, const int flag);
}



// single kernel
static void decx::dev_Conv2_BC_R8x8_SK(
    decx::_GPU_MatrixArray* src, decx::_GPU_Matrix* kernel, decx::_GPU_MatrixArray* dst, de::DH* handle, const int flag)
{
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

    decx::cuda_stream* S = NULL;
    S = decx::CStream.stream_accessor_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        Print_Error_Message(4, CUDA_STREAM_ACCESS_FAIL);
        decx::err::CUDA_Stream_access_fail(handle);
        return;
    }

    decx::PtrInfo<float4> dev_tmp_1, dev_tmp_2;
    if (decx::alloc::_device_malloc(&dev_tmp_1, dev_src_size * sizeof(float4), true, S)) {
        Print_Error_Message(4, DEV_ALLOC_FAIL);
        decx::err::device_AllocateFailure(handle);
        return;
    }
    if (decx::alloc::_device_malloc(&dev_tmp_2, dev_dst_size * sizeof(float4), true, S)) {
        Print_Error_Message(4, DEV_ALLOC_FAIL);
        decx::err::device_AllocateFailure(handle);
        return;
    }

    float4 *Dsrc = dev_tmp_1.ptr;
    float4 *Ddst = dev_tmp_2.ptr;

    uint offset_lin = 0, offset_ker = 0;

    for (uint k = 0; k < kernel->height; ++k) {
        cudaMemcpyToSymbolAsync(decx::Const_Mem, (de::Half*)kernel->Mat.ptr + offset_ker,
            kernel->width * sizeof(de::Half), offset_lin * sizeof(de::Half), cudaMemcpyDeviceToDevice, S->get_raw_stream_ref());
        offset_lin += kernel->width;
        offset_ker += kernel->pitch;
    }
    // strat the main loop
    if (ker_dim.x == (bounded_kernel_R8 * 2 + 1) && ker_dim.y == (bounded_kernel_R8 * 2 + 1)) {
        decx::dev_main_loop_hconv2_sk_exact8x8_BC(
            &Dsrc_alloc_dim, &Ddst_alloc_dim, &ker_dim, src, kernel, dst, Dsrc, Ddst, S, flag);
    }
    else {
        decx::dev_main_loop_hconv2_sk_within8x8_BC(
            &Dsrc_alloc_dim, &Ddst_alloc_dim, &ker_dim, src, kernel, dst, Dsrc, Ddst, S, flag);
    }

    checkCudaErrors(cudaDeviceSynchronize());

    decx::alloc::_device_dealloc(&dev_tmp_1);
    decx::alloc::_device_dealloc(&dev_tmp_2);

    S->detach();
}




// single kernel
static void decx::dev_Conv2_BC_R8x16_SK(
    decx::_GPU_MatrixArray* src, decx::_GPU_Matrix* kernel, decx::_GPU_MatrixArray* dst, de::DH* handle, const int flag)
{
    int2 Dsrc_alloc_dim,
        Ddst_alloc_dim,
        ker_dim;

    ker_dim.x = kernel->width;            ker_dim.y = kernel->height;
    // first, allocate the memory according to R8 alignments
    Ddst_alloc_dim.x = decx::utils::ceil<int>(dst->width, 64) * bounded_kernel_R16 * 2;
    Ddst_alloc_dim.y = decx::utils::ceil<int>(dst->height, 16) * bounded_kernel_R8 * 2;

    Dsrc_alloc_dim.x = Ddst_alloc_dim.x + bounded_kernel_R16 / 4;        // bounded_kernel_R8 * 2 / 4
    Dsrc_alloc_dim.y = Ddst_alloc_dim.y + bounded_kernel_R8 * 2;

    size_t dev_src_size = static_cast<size_t>(Dsrc_alloc_dim.x) * static_cast<size_t>(Dsrc_alloc_dim.y);
    size_t dev_dst_size = static_cast<size_t>(Ddst_alloc_dim.x) * static_cast<size_t>(Ddst_alloc_dim.y);

    decx::cuda_stream* S = NULL;
    S = decx::CStream.stream_accessor_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        Print_Error_Message(4, CUDA_STREAM_ACCESS_FAIL);
        decx::err::CUDA_Stream_access_fail(handle);
        return;
    }

    decx::PtrInfo<float4> dev_tmp_1, dev_tmp_2;
    if (decx::alloc::_device_malloc(&dev_tmp_1, dev_src_size * sizeof(float4), true, S)) {
        Print_Error_Message(4, DEV_ALLOC_FAIL);
        decx::err::device_AllocateFailure(handle);
        return;
    }
    if (decx::alloc::_device_malloc(&dev_tmp_2, dev_dst_size * sizeof(float4), true, S)) {
        Print_Error_Message(4, DEV_ALLOC_FAIL);
        decx::err::device_AllocateFailure(handle);
        return;
    }

    float4* Dsrc = dev_tmp_1.ptr;
    float4* Ddst = dev_tmp_2.ptr;

    uint offset_lin = 0, offset_ker = 0;

    for (uint k = 0; k < kernel->height; ++k) {
        cudaMemcpyToSymbolAsync(decx::Const_Mem, (de::Half*)kernel->Mat.ptr + offset_ker,
            kernel->width * sizeof(de::Half), offset_lin * sizeof(de::Half), cudaMemcpyDeviceToDevice, 
            S->get_raw_stream_ref());
        offset_lin += kernel->width;
        offset_ker += kernel->pitch;
    }
    // strat the main loop
    if (ker_dim.x == (bounded_kernel_R16 * 2 + 1) && ker_dim.y == (bounded_kernel_R8 * 2 + 1)) {
        decx::dev_main_loop_hconv2_sk_exact8x16_BC(
            &Dsrc_alloc_dim, &Ddst_alloc_dim, &ker_dim, src, kernel, dst, Dsrc, Ddst, S, flag);
    }
    else {
        decx::dev_main_loop_hconv2_sk_within8x16_BC(
            &Dsrc_alloc_dim, &Ddst_alloc_dim, &ker_dim, src, kernel, dst, Dsrc, Ddst, S, flag);
    }

    checkCudaErrors(cudaDeviceSynchronize());

    decx::alloc::_device_dealloc(&dev_tmp_1);
    decx::alloc::_device_dealloc(&dev_tmp_2);

    S->detach();
}



// single kernel
static void decx::dev_Conv2_BC_R16x8_SK(
    decx::_GPU_MatrixArray* src, decx::_GPU_Matrix* kernel, decx::_GPU_MatrixArray* dst, de::DH* handle, const int flag)
{
    MIF<float4> Dmem1, Dmem2,    // for src
        Dmem3, Dmem4;            // for dst

    int2 Dsrc_alloc_dim,
        Ddst_alloc_dim,
        ker_dim;

    ker_dim.x = kernel->width;            ker_dim.y = kernel->height;
    // first, allocate the memory according to R8 alignments
    Ddst_alloc_dim.x = decx::utils::ceil<int>(dst->width, 64) * bounded_kernel_R8 * 2;
    Ddst_alloc_dim.y = decx::utils::ceil<int>(dst->height, 16) * bounded_kernel_R16 * 2;

    Dsrc_alloc_dim.x = Ddst_alloc_dim.x + bounded_kernel_R8 / 4;        // bounded_kernel_R8 * 2 / 4
    Dsrc_alloc_dim.y = Ddst_alloc_dim.y + bounded_kernel_R16 * 2;

    size_t dev_src_size = static_cast<size_t>(Dsrc_alloc_dim.x) * static_cast<size_t>(Dsrc_alloc_dim.y);
    size_t dev_dst_size = static_cast<size_t>(Ddst_alloc_dim.x) * static_cast<size_t>(Ddst_alloc_dim.y);

    decx::cuda_stream* S = NULL;
    S = decx::CStream.stream_accessor_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        Print_Error_Message(4, CUDA_STREAM_ACCESS_FAIL);
        decx::err::CUDA_Stream_access_fail(handle);
        return;
    }

    decx::PtrInfo<float4> dev_tmp_1, dev_tmp_2;
    if (decx::alloc::_device_malloc(&dev_tmp_1, dev_src_size * sizeof(float4), true, S)) {
        Print_Error_Message(4, DEV_ALLOC_FAIL);
        decx::err::device_AllocateFailure(handle);
        return;
    }
    if (decx::alloc::_device_malloc(&dev_tmp_2, dev_dst_size * sizeof(float4), true, S)) {
        Print_Error_Message(4, DEV_ALLOC_FAIL);
        decx::err::device_AllocateFailure(handle);
        return;
    }

    float4* Dsrc = dev_tmp_1.ptr;
    float4* Ddst = dev_tmp_2.ptr;

    uint offset_lin = 0, offset_ker = 0;

    for (int k = 0; k < kernel->height; ++k) {
        cudaMemcpyToSymbolAsync(decx::Const_Mem, (de::Half*)kernel->Mat.ptr + offset_ker,
            kernel->width * sizeof(de::Half), offset_lin * sizeof(de::Half), cudaMemcpyDeviceToDevice, 
            S->get_raw_stream_ref());
        offset_lin += kernel->width;
        offset_ker += kernel->pitch;
    }
    // strat the main loop
    if (ker_dim.x == (bounded_kernel_R8 * 2 + 1) && ker_dim.y == (bounded_kernel_R16 * 2 + 1)) {
        decx::dev_main_loop_hconv2_sk_exact16x8_BC(
            &Dsrc_alloc_dim, &Ddst_alloc_dim, &ker_dim, src, kernel, dst, Dsrc, Ddst, S, flag);
    }
    else {
        decx::dev_main_loop_hconv2_sk_within16x8_BC(
            &Dsrc_alloc_dim, &Ddst_alloc_dim, &ker_dim, src, kernel, dst, Dsrc, Ddst, S, flag);
    }

    checkCudaErrors(cudaDeviceSynchronize());

    decx::alloc::_device_dealloc(&dev_tmp_1);
    decx::alloc::_device_dealloc(&dev_tmp_2);

    S->detach();
}





// single kernel
static void decx::dev_Conv2_BC_R16x16_SK(
    decx::_GPU_MatrixArray* src, decx::_GPU_Matrix* kernel, decx::_GPU_MatrixArray* dst, de::DH* handle, const int flag)
{
    MIF<float4> Dmem1, Dmem2,    // for src
        Dmem3, Dmem4;            // for dst

    int2 Dsrc_alloc_dim,
        Ddst_alloc_dim,
        ker_dim;

    ker_dim.x = kernel->width;            ker_dim.y = kernel->height;
    // first, allocate the memory according to R8 alignments
    Ddst_alloc_dim.x = decx::utils::ceil<int>(dst->width, 64) * bounded_kernel_R16 * 2;
    Ddst_alloc_dim.y = decx::utils::ceil<int>(dst->height, 16) * bounded_kernel_R16 * 2;

    Dsrc_alloc_dim.x = Ddst_alloc_dim.x + bounded_kernel_R16 / 4;        // bounded_kernel_R8 * 2 / 4
    Dsrc_alloc_dim.y = Ddst_alloc_dim.y + bounded_kernel_R16 * 2;

    size_t dev_src_size = static_cast<size_t>(Dsrc_alloc_dim.x) * static_cast<size_t>(Dsrc_alloc_dim.y);
    size_t dev_dst_size = static_cast<size_t>(Ddst_alloc_dim.x) * static_cast<size_t>(Ddst_alloc_dim.y);

    decx::cuda_stream* S = NULL;
    S = decx::CStream.stream_accessor_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        Print_Error_Message(4, CUDA_STREAM_ACCESS_FAIL);
        decx::err::CUDA_Stream_access_fail(handle);
        return;
    }

    decx::PtrInfo<float4> dev_tmp_1, dev_tmp_2;
    if (decx::alloc::_device_malloc(&dev_tmp_1, dev_src_size * sizeof(float4), true, S)) {
        Print_Error_Message(4, DEV_ALLOC_FAIL);
        decx::err::device_AllocateFailure(handle);
        return;
    }
    if (decx::alloc::_device_malloc(&dev_tmp_2, dev_dst_size * sizeof(float4), true, S)) {
        Print_Error_Message(4, DEV_ALLOC_FAIL);
        decx::err::device_AllocateFailure(handle);
        return;
    }

    float4* Dsrc = dev_tmp_1.ptr;
    float4* Ddst = dev_tmp_2.ptr;

    uint offset_lin = 0, offset_ker = 0;

    for (int k = 0; k < kernel->height; ++k) {
        cudaMemcpyToSymbolAsync(decx::Const_Mem, (de::Half*)kernel->Mat.ptr + offset_ker,
            kernel->width * sizeof(de::Half), offset_lin * sizeof(de::Half), cudaMemcpyDeviceToDevice, 
            S->get_raw_stream_ref());
        offset_lin += kernel->width;
        offset_ker += kernel->pitch;
    }
    // strat the main loop
    if (ker_dim.x == (bounded_kernel_R16 * 2 + 1) && ker_dim.y == (bounded_kernel_R16 * 2 + 1)) {
        decx::dev_main_loop_hconv2_sk_exact16x16_BC(
            &Dsrc_alloc_dim, &Ddst_alloc_dim, &ker_dim, src, kernel, dst, Dsrc, Ddst, S, flag);
    }
    else {
        decx::dev_main_loop_hconv2_sk_within16x16_BC(
            &Dsrc_alloc_dim, &Ddst_alloc_dim, &ker_dim, src, kernel, dst, Dsrc, Ddst, S, flag);
    }

    checkCudaErrors(cudaDeviceSynchronize());

    decx::alloc::_device_dealloc(&dev_tmp_1);
    decx::alloc::_device_dealloc(&dev_tmp_2);

    S->detach();
}



// ******************************************************************************************************************************

static void decx::dev_hConv2_border_zero_sk(
    decx::_GPU_MatrixArray* src, decx::_GPU_Matrix* kernel, decx::_GPU_MatrixArray* dst, de::DH* handle, const int flag)
{
    int2 half_ker_dim;
    half_ker_dim.x = kernel->width / 2;                half_ker_dim.y = kernel->height / 2;

    dst->re_construct(src->type, src->width, src->height, decx::DATA_STORE_TYPE::Page_Locked);

    if (half_ker_dim.x < bounded_kernel_R8 + 1) {
        if (half_ker_dim.y < bounded_kernel_R8 + 1) {
            decx::dev_Conv2_BC_R8x8_SK(src, kernel, dst, handle, flag);
        }
        else {
            decx::dev_Conv2_BC_R16x8_SK(src, kernel, dst, handle, flag);
        }
    }
    else {
        if (half_ker_dim.y < bounded_kernel_R8 + 1) {
            decx::dev_Conv2_BC_R8x16_SK(src, kernel, dst, handle, flag);
        }
        else {
            decx::dev_Conv2_BC_R16x16_SK(src, kernel, dst, handle, flag);
        }
    }
}


#endif