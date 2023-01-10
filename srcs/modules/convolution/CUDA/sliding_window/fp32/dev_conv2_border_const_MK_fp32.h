/**
*    ---------------------------------------------------------------------
*    Author : Wayne Anderson
*    Date   : 2021.04.16
*    ---------------------------------------------------------------------
*    This is a part of the open source program named "DECX", copyright c Wayne,
*    2021.04.16
*/

#ifndef _DEV_CONV2_MC_BORDER_CONST_MK_FP32_H_
#define _DEV_CONV2_MC_BORDER_CONST_MK_FP32_H_


#include "../../../../core/basic.h"
#include "../../../../classes/GPU_Matrix.h"
#include "../../../../classes/GPU_MatrixArray.h"
#include "../Conv2_MC_macros.h"
#include "sconv2_kernel_callers.h"
#include "../../../conv_utils.h"


using decx::_Matrix;
using decx::alloc::MIF;
using decx::_MatrixArray;


namespace decx
{
    static void dev_main_loop_sconv2_mk_within8x8_BC(int2* Dsrc_alloc_dim, int2* Ddst_alloc_dim, int2* ker_dim,
        decx::_GPU_MatrixArray* src, decx::_GPU_MatrixArray* kernel, decx::_GPU_MatrixArray* dst,
        float4* Dsrc, float4 *Ddst,
        decx::cuda_stream* S);



    static void dev_main_loop_sconv2_mk_exact8x8_BC(int2* Dsrc_alloc_dim, int2* Ddst_alloc_dim, int2* ker_dim,
        decx::_GPU_MatrixArray* src, decx::_GPU_MatrixArray* kernel, decx::_GPU_MatrixArray* dst,
        float4* Dsrc, float4* Ddst,
        decx::cuda_stream* S);



    static void dev_main_loop_sconv2_mk_within8x16_BC(int2* Dsrc_alloc_dim, int2* Ddst_alloc_dim, int2* ker_dim,
        decx::_GPU_MatrixArray* src, decx::_GPU_MatrixArray* kernel, decx::_GPU_MatrixArray* dst,
        float4* Dsrc, float4* Ddst,
        decx::cuda_stream* S);


    static void dev_main_loop_sconv2_mk_exact8x16_BC(int2* Dsrc_alloc_dim, int2* Ddst_alloc_dim, int2* ker_dim,
        decx::_GPU_MatrixArray* src, decx::_GPU_MatrixArray* kernel, decx::_GPU_MatrixArray* dst,
        float4* Dsrc, float4* Ddst,
        decx::cuda_stream* S);


    static void dev_main_loop_sconv2_mk_within16x8_BC(int2* Dsrc_alloc_dim, int2* Ddst_alloc_dim, int2* ker_dim,
        decx::_GPU_MatrixArray* src, decx::_GPU_MatrixArray* kernel, decx::_GPU_MatrixArray* dst,
        float4* Dsrc, float4* Ddst,
        decx::cuda_stream* S);


    static void dev_main_loop_sconv2_mk_exact16x8_BC(int2* Dsrc_alloc_dim, int2* Ddst_alloc_dim, int2* ker_dim,
        decx::_GPU_MatrixArray* src, decx::_GPU_MatrixArray* kernel, decx::_GPU_MatrixArray* dst,
        float4* Dsrc, float4* Ddst,
        decx::cuda_stream* S);


    static void dev_main_loop_sconv2_mk_within16x16_BC(int2* Dsrc_alloc_dim, int2* Ddst_alloc_dim, int2* ker_dim,
        decx::_GPU_MatrixArray* src, decx::_GPU_MatrixArray* kernel, decx::_GPU_MatrixArray* dst,
        float4* Dsrc, float4* Ddst,
        decx::cuda_stream* S);


    static void dev_main_loop_sconv2_mk_exact16x16_BC(int2* Dsrc_alloc_dim, int2* Ddst_alloc_dim, int2* ker_dim,
        decx::_GPU_MatrixArray* src, decx::_GPU_MatrixArray* kernel, decx::_GPU_MatrixArray* dst,
        float4* Dsrc, float4* Ddst,
        decx::cuda_stream* S);


    // single kernel
    static void dev_Conv2_BC_R8x8_MK(decx::_GPU_MatrixArray* src, decx::_GPU_MatrixArray* kernel, decx::_GPU_MatrixArray* dst);


    // single kernel
    static void dev_Conv2_BC_R8x16_MK(decx::_GPU_MatrixArray* src, decx::_GPU_MatrixArray* kernel, decx::_GPU_MatrixArray* dst);


    // single kernel
    static void dev_Conv2_BC_R16x8_MK(decx::_GPU_MatrixArray* src, decx::_GPU_MatrixArray* kernel, decx::_GPU_MatrixArray* dst);


    // single kernel
    static void dev_Conv2_BC_R16x16_MK(decx::_GPU_MatrixArray* src, decx::_GPU_MatrixArray* kernel, decx::_GPU_MatrixArray* dst);


    static void dev_sConv2_border_zero_mk(decx::_GPU_MatrixArray* src, decx::_GPU_MatrixArray* kernel, decx::_GPU_MatrixArray* dst, de::DH* handle);
}






static void decx::dev_main_loop_sconv2_mk_within8x8_BC(int2* Dsrc_alloc_dim, int2* Ddst_alloc_dim, int2* ker_dim,
    decx::_GPU_MatrixArray* src, decx::_GPU_MatrixArray* kernel, decx::_GPU_MatrixArray* dst,
    float4* Dsrc, float4* Ddst,
    decx::cuda_stream* S)
{
    int2 src_diff;
    // copy the first part to device memory
    src_diff.x = bounded_kernel_R8 - ker_dim->y / 2;                src_diff.y = bounded_kernel_R8 - ker_dim->x / 2;

    size_t offset_lin, offset_ker;
    for (int i = 0; i < src->ArrayNumber; ++i) 
    {
        checkCudaErrors(cudaMemcpy2DAsync(
            reinterpret_cast<float*>(Dsrc) + Dsrc_alloc_dim->x * bounded_kernel_R8 * 4 + bounded_kernel_R8,
            Dsrc_alloc_dim->x * sizeof(float4),
            src->MatptrArr.ptr[i],
            src->pitch * sizeof(float),
            src->width * sizeof(float),
            src->height,
            cudaMemcpyDeviceToDevice,
            S->get_raw_stream_ref()));                            // copy the datas of src from host to device

        // copy kernel data (one plane) to the constant memory
        offset_lin = 0;
        offset_ker = 0;
        for (uint k = 0; k < kernel->height; ++k) {
            cudaMemcpyToSymbolAsync(decx::Const_Mem, (float*)kernel->MatptrArr.ptr[i] + offset_ker,
                kernel->width * sizeof(float), offset_lin * sizeof(float), cudaMemcpyDeviceToDevice, S->get_raw_stream_ref());
            offset_lin += kernel->width;
            offset_ker += kernel->pitch;
        }

        sconv2_kernel_within8x8(Dsrc, Ddst, src_diff, *Dsrc_alloc_dim, *Ddst_alloc_dim, *ker_dim, S);

        checkCudaErrors(cudaMemcpy2DAsync(dst->MatptrArr.ptr[i],
            dst->pitch * sizeof(float), Ddst, Ddst_alloc_dim->x * sizeof(float4), dst->width * sizeof(float),
            dst->height, cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));
    }
}




static void decx::dev_main_loop_sconv2_mk_exact8x8_BC(int2* Dsrc_alloc_dim, int2* Ddst_alloc_dim, int2* ker_dim,
    decx::_GPU_MatrixArray* src, decx::_GPU_MatrixArray* kernel, decx::_GPU_MatrixArray* dst,
    float4* Dsrc, float4* Ddst,
    decx::cuda_stream* S)
{
    size_t offset_lin, offset_ker;
    for (int i = 0; i < src->ArrayNumber; ++i)
    {
        checkCudaErrors(cudaMemcpy2DAsync(
            reinterpret_cast<float*>(Dsrc) + Dsrc_alloc_dim->x * bounded_kernel_R8 * 4 + bounded_kernel_R8,
            Dsrc_alloc_dim->x * sizeof(float4),
            src->MatptrArr.ptr[i],
            src->pitch * sizeof(float),
            src->width * sizeof(float),
            src->height,
            cudaMemcpyDeviceToDevice,
            S->get_raw_stream_ref()));                            // copy the datas of src from host to device

        // copy kernel data (one plane) to the constant memory
        offset_lin = 0;
        offset_ker = 0;
        for (uint k = 0; k < kernel->height; ++k) {
            cudaMemcpyToSymbolAsync(decx::Const_Mem, (float*)kernel->MatptrArr.ptr[i] + offset_ker,
                kernel->width * sizeof(float), offset_lin * sizeof(float), cudaMemcpyDeviceToDevice, S->get_raw_stream_ref());
            offset_lin += kernel->width;
            offset_ker += kernel->pitch;
        }

        sconv2_kernel_exact8x8(Dsrc, Ddst, *Dsrc_alloc_dim, *Ddst_alloc_dim, *ker_dim, S);

        checkCudaErrors(cudaMemcpy2DAsync(dst->MatptrArr.ptr[i],
            dst->pitch * sizeof(float), Ddst, Ddst_alloc_dim->x * sizeof(float4), dst->width * sizeof(float),
            dst->height, cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));
    }
}





static void decx::dev_main_loop_sconv2_mk_within8x16_BC(int2* Dsrc_alloc_dim, int2* Ddst_alloc_dim, int2* ker_dim,
    decx::_GPU_MatrixArray* src, decx::_GPU_MatrixArray* kernel, decx::_GPU_MatrixArray* dst,
    float4* Dsrc, float4* Ddst,
    decx::cuda_stream* S)
{
    int2 src_diff;
    // copy the first part to device memory
    src_diff.x = bounded_kernel_R8 - ker_dim->y / 2;                src_diff.y = bounded_kernel_R16 - ker_dim->x / 2;

    size_t offset_lin, offset_ker;
    for (int i = 0; i < src->ArrayNumber; ++i)
    {
        checkCudaErrors(cudaMemcpy2DAsync(
            reinterpret_cast<float*>(Dsrc) + Dsrc_alloc_dim->x * bounded_kernel_R8 * 4 + bounded_kernel_R16,
            Dsrc_alloc_dim->x * sizeof(float4),
            src->MatptrArr.ptr[i],
            src->pitch * sizeof(float),
            src->width * sizeof(float),
            src->height,
            cudaMemcpyDeviceToDevice,
            S->get_raw_stream_ref()));                            // copy the datas of src from host to device

        // copy kernel data (one plane) to the constant memory
        offset_lin = 0;
        offset_ker = 0;
        for (uint k = 0; k < kernel->height; ++k) {
            cudaMemcpyToSymbolAsync(decx::Const_Mem, (float*)kernel->MatptrArr.ptr[i] + offset_ker,
                kernel->width * sizeof(float), offset_lin * sizeof(float), cudaMemcpyDeviceToDevice, S->get_raw_stream_ref());
            offset_lin += kernel->width;
            offset_ker += kernel->pitch;
        }

        sconv2_kernel_within8x16(Dsrc, Ddst, src_diff, *Dsrc_alloc_dim, *Ddst_alloc_dim, *ker_dim, S);

        checkCudaErrors(cudaMemcpy2DAsync(dst->MatptrArr.ptr[i],
            dst->pitch * sizeof(float), Ddst, Ddst_alloc_dim->x * sizeof(float4), dst->width * sizeof(float),
            dst->height, cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));
    }
}




static void decx::dev_main_loop_sconv2_mk_exact8x16_BC(int2* Dsrc_alloc_dim, int2* Ddst_alloc_dim, int2* ker_dim,
    decx::_GPU_MatrixArray* src, decx::_GPU_MatrixArray* kernel, decx::_GPU_MatrixArray* dst,
    float4* Dsrc, float4* Ddst,
    decx::cuda_stream* S)
{
    size_t offset_lin, offset_ker;
    for (int i = 0; i < src->ArrayNumber; ++i)
    {
        checkCudaErrors(cudaMemcpy2DAsync(
            reinterpret_cast<float*>(Dsrc) + Dsrc_alloc_dim->x * bounded_kernel_R8 * 4 + bounded_kernel_R16,
            Dsrc_alloc_dim->x * sizeof(float4),
            src->MatptrArr.ptr[i],
            src->pitch * sizeof(float),
            src->width * sizeof(float),
            src->height,
            cudaMemcpyDeviceToDevice,
            S->get_raw_stream_ref()));                            // copy the datas of src from host to device

        // copy kernel data (one plane) to the constant memory
        offset_lin = 0;
        offset_ker = 0;
        for (uint k = 0; k < kernel->height; ++k) {
            cudaMemcpyToSymbolAsync(decx::Const_Mem, (float*)kernel->MatptrArr.ptr[i] + offset_ker,
                kernel->width * sizeof(float), offset_lin * sizeof(float), cudaMemcpyDeviceToDevice, S->get_raw_stream_ref());
            offset_lin += kernel->width;
            offset_ker += kernel->pitch;
        }

        sconv2_kernel_exact8x16(Dsrc, Ddst, *Dsrc_alloc_dim, *Ddst_alloc_dim, *ker_dim, S);

        checkCudaErrors(cudaMemcpy2DAsync(dst->MatptrArr.ptr[i],
            dst->pitch * sizeof(float), Ddst, Ddst_alloc_dim->x * sizeof(float4), dst->width * sizeof(float),
            dst->height, cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));
    }
}




static void decx::dev_main_loop_sconv2_mk_within16x8_BC(int2* Dsrc_alloc_dim, int2* Ddst_alloc_dim, int2* ker_dim,
    decx::_GPU_MatrixArray* src, decx::_GPU_MatrixArray* kernel, decx::_GPU_MatrixArray* dst,
    float4* Dsrc, float4* Ddst,
    decx::cuda_stream* S)
{
    int2 src_diff;
    // copy the first part to device memory
    src_diff.x = bounded_kernel_R16 - ker_dim->y / 2;                src_diff.y = bounded_kernel_R8 - ker_dim->x / 2;

    size_t offset_lin, offset_ker;
    for (int i = 0; i < src->ArrayNumber; ++i)
    {
        checkCudaErrors(cudaMemcpy2DAsync(
            reinterpret_cast<float*>(Dsrc) + Dsrc_alloc_dim->x * bounded_kernel_R16 * 4 + bounded_kernel_R8,
            Dsrc_alloc_dim->x * sizeof(float4),
            src->MatptrArr.ptr[i],
            src->pitch * sizeof(float),
            src->width * sizeof(float),
            src->height,
            cudaMemcpyDeviceToDevice,
            S->get_raw_stream_ref()));                            // copy the datas of src from host to device

        // copy kernel data (one plane) to the constant memory
        offset_lin = 0;
        offset_ker = 0;
        for (uint k = 0; k < kernel->height; ++k) {
            cudaMemcpyToSymbolAsync(decx::Const_Mem, (float*)kernel->MatptrArr.ptr[i] + offset_ker,
                kernel->width * sizeof(float), offset_lin * sizeof(float), cudaMemcpyDeviceToDevice, S->get_raw_stream_ref());
            offset_lin += kernel->width;
            offset_ker += kernel->pitch;
        }

        sconv2_kernel_within16x8(Dsrc, Ddst, src_diff, *Dsrc_alloc_dim, *Ddst_alloc_dim, *ker_dim, S);

        checkCudaErrors(cudaMemcpy2DAsync(dst->MatptrArr.ptr[i],
            dst->pitch * sizeof(float), Ddst, Ddst_alloc_dim->x * sizeof(float4), dst->width * sizeof(float),
            dst->height, cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));
    }
}




static void decx::dev_main_loop_sconv2_mk_exact16x8_BC(int2* Dsrc_alloc_dim, int2* Ddst_alloc_dim, int2* ker_dim,
    decx::_GPU_MatrixArray* src, decx::_GPU_MatrixArray* kernel, decx::_GPU_MatrixArray* dst,
    float4* Dsrc, float4* Ddst,
    decx::cuda_stream* S)
{
    size_t offset_lin, offset_ker;
    for (int i = 0; i < src->ArrayNumber; ++i)
    {
        checkCudaErrors(cudaMemcpy2DAsync(
            reinterpret_cast<float*>(Dsrc) + Dsrc_alloc_dim->x * bounded_kernel_R16 * 4 + bounded_kernel_R8,
            Dsrc_alloc_dim->x * sizeof(float4),
            src->MatptrArr.ptr[i],
            src->pitch * sizeof(float),
            src->width * sizeof(float),
            src->height,
            cudaMemcpyDeviceToDevice,
            S->get_raw_stream_ref()));                            // copy the datas of src from host to device

        // copy kernel data (one plane) to the constant memory
        offset_lin = 0;
        offset_ker = 0;
        for (uint k = 0; k < kernel->height; ++k) {
            cudaMemcpyToSymbolAsync(decx::Const_Mem, (float*)kernel->MatptrArr.ptr[i] + offset_ker,
                kernel->width * sizeof(float), offset_lin * sizeof(float), cudaMemcpyDeviceToDevice, S->get_raw_stream_ref());
            offset_lin += kernel->width;
            offset_ker += kernel->pitch;
        }

        sconv2_kernel_exact16x8(Dsrc, Ddst, *Dsrc_alloc_dim, *Ddst_alloc_dim, *ker_dim, S);

        checkCudaErrors(cudaMemcpy2DAsync(dst->MatptrArr.ptr[i],
            dst->pitch * sizeof(float), Ddst, Ddst_alloc_dim->x * sizeof(float4), dst->width * sizeof(float),
            dst->height, cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));
    }
}





static void decx::dev_main_loop_sconv2_mk_within16x16_BC(int2* Dsrc_alloc_dim, int2* Ddst_alloc_dim, int2* ker_dim,
    decx::_GPU_MatrixArray* src, decx::_GPU_MatrixArray* kernel, decx::_GPU_MatrixArray* dst,
    float4* Dsrc, float4* Ddst,
    decx::cuda_stream* S)
{
    int2 src_diff;
    // copy the first part to device memory
    src_diff.x = bounded_kernel_R16 - ker_dim->y / 2;                src_diff.y = bounded_kernel_R16 - ker_dim->x / 2;

    size_t offset_lin, offset_ker;
    for (int i = 0; i < src->ArrayNumber; ++i)
    {
        checkCudaErrors(cudaMemcpy2DAsync(
            reinterpret_cast<float*>(Dsrc) + Dsrc_alloc_dim->x * bounded_kernel_R16 * 4 + bounded_kernel_R16,
            Dsrc_alloc_dim->x * sizeof(float4),
            src->MatptrArr.ptr[i],
            src->pitch * sizeof(float),
            src->width * sizeof(float),
            src->height,
            cudaMemcpyDeviceToDevice,
            S->get_raw_stream_ref()));                            // copy the datas of src from host to device

        // copy kernel data (one plane) to the constant memory
        offset_lin = 0;
        offset_ker = 0;
        for (uint k = 0; k < kernel->height; ++k) {
            cudaMemcpyToSymbolAsync(decx::Const_Mem, (float*)kernel->MatptrArr.ptr[i] + offset_ker,
                kernel->width * sizeof(float), offset_lin * sizeof(float), cudaMemcpyDeviceToDevice, S->get_raw_stream_ref());
            offset_lin += kernel->width;
            offset_ker += kernel->pitch;
        }

        sconv2_kernel_within16x16(Dsrc, Ddst, src_diff, *Dsrc_alloc_dim, *Ddst_alloc_dim, *ker_dim, S);

        checkCudaErrors(cudaMemcpy2DAsync(dst->MatptrArr.ptr[i],
            dst->pitch * sizeof(float), Ddst, Ddst_alloc_dim->x * sizeof(float4), dst->width * sizeof(float),
            dst->height, cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));
    }
}




static void decx::dev_main_loop_sconv2_mk_exact16x16_BC(int2* Dsrc_alloc_dim, int2* Ddst_alloc_dim, int2* ker_dim,
    decx::_GPU_MatrixArray* src, decx::_GPU_MatrixArray* kernel, decx::_GPU_MatrixArray* dst,
    float4* Dsrc, float4* Ddst,
    decx::cuda_stream* S)
{
    size_t offset_lin, offset_ker;
    for (int i = 0; i < src->ArrayNumber; ++i)
    {
        checkCudaErrors(cudaMemcpy2DAsync(
            reinterpret_cast<float*>(Dsrc) + Dsrc_alloc_dim->x * bounded_kernel_R16 * 4 + bounded_kernel_R16,
            Dsrc_alloc_dim->x * sizeof(float4),
            src->MatptrArr.ptr[i],
            src->pitch * sizeof(float),
            src->width * sizeof(float),
            src->height,
            cudaMemcpyDeviceToDevice,
            S->get_raw_stream_ref()));                            // copy the datas of src from host to device

        // copy kernel data (one plane) to the constant memory
        offset_lin = 0;
        offset_ker = 0;
        for (uint k = 0; k < kernel->height; ++k) {
            cudaMemcpyToSymbolAsync(decx::Const_Mem, (float*)kernel->MatptrArr.ptr[i] + offset_ker,
                kernel->width * sizeof(float), offset_lin * sizeof(float), cudaMemcpyDeviceToDevice, S->get_raw_stream_ref());
            offset_lin += kernel->width;
            offset_ker += kernel->pitch;
        }

        sconv2_kernel_exact16x16(Dsrc, Ddst, *Dsrc_alloc_dim, *Ddst_alloc_dim, *ker_dim, S);

        checkCudaErrors(cudaMemcpy2DAsync(dst->MatptrArr.ptr[i],
            dst->pitch * sizeof(float), Ddst, Ddst_alloc_dim->x * sizeof(float4), dst->width * sizeof(float),
            dst->height, cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));
    }
}



 // **************************************************************************************************************************


// single kernel
static void decx::dev_Conv2_BC_R8x8_MK(decx::_GPU_MatrixArray* src, decx::_GPU_MatrixArray* kernel, decx::_GPU_MatrixArray* dst)
{
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

    decx::cuda_stream* S = NULL;
    S = decx::CStream.stream_accessor_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        Print_Error_Message(4, CUDA_STREAM_ACCESS_FAIL);
        return;
    }
    decx::PtrInfo<float4> dev_buffer1, dev_buffer2;
    if (decx::alloc::_device_malloc(&dev_buffer1, dev_src_size * sizeof(float4), true, S)) {
        Print_Error_Message(4, ALLOC_FAIL);
        return;
    }
    if (decx::alloc::_device_malloc(&dev_buffer2, dev_dst_size * sizeof(float4), true, S)) {
        Print_Error_Message(4, ALLOC_FAIL);
        return;
    }

    float4* Dsrc = dev_buffer1.ptr;
    float4* Ddst = dev_buffer2.ptr;

    // strat the main loop
    if (ker_dim.x == (bounded_kernel_R8 * 2 + 1) && ker_dim.y == (bounded_kernel_R8 * 2 + 1)) {
        dev_main_loop_sconv2_mk_exact8x8_BC(
            &Dsrc_alloc_dim, &Ddst_alloc_dim, &ker_dim, src, kernel, dst, dev_buffer1.ptr, dev_buffer2.ptr, S);
    }
    else {
        dev_main_loop_sconv2_mk_within8x8_BC(
            &Dsrc_alloc_dim, &Ddst_alloc_dim, &ker_dim, src, kernel, dst, dev_buffer1.ptr, dev_buffer2.ptr, S);
    }
    checkCudaErrors(cudaDeviceSynchronize());
    decx::alloc::_device_dealloc(&dev_buffer1);
    decx::alloc::_device_dealloc(&dev_buffer2);
    
    S->detach();
}




// single kernel
static void decx::dev_Conv2_BC_R8x16_MK(decx::_GPU_MatrixArray* src, decx::_GPU_MatrixArray* kernel, decx::_GPU_MatrixArray* dst)
{
    int2 Dsrc_alloc_dim,
        Ddst_alloc_dim,
        ker_dim;

    ker_dim.x = kernel->width;            ker_dim.y = kernel->height;
    // first, allocate the memory according to R8 alignments
    Ddst_alloc_dim.x = decx::utils::ceil<int>(dst->width, 64) * bounded_kernel_R16 * 2;
    Ddst_alloc_dim.y = decx::utils::ceil<int>(dst->height, 16) * bounded_kernel_R8 * 2;

    Dsrc_alloc_dim.x = Ddst_alloc_dim.x + bounded_kernel_R16 / 2;        // bounded_kernel_R8 * 2 / 4
    Dsrc_alloc_dim.y = Ddst_alloc_dim.y + bounded_kernel_R8 * 2;

    size_t dev_src_size = static_cast<size_t>(Dsrc_alloc_dim.x) * static_cast<size_t>(Dsrc_alloc_dim.y);
    size_t dev_dst_size = static_cast<size_t>(Ddst_alloc_dim.x) * static_cast<size_t>(Ddst_alloc_dim.y);


    decx::cuda_stream* S = NULL;
    S = decx::CStream.stream_accessor_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        Print_Error_Message(4, CUDA_STREAM_ACCESS_FAIL);
        return;
    }
    decx::PtrInfo<float4> dev_buffer1, dev_buffer2;
    if (decx::alloc::_device_malloc(&dev_buffer1, dev_src_size * sizeof(float4), true, S)) {
        Print_Error_Message(4, ALLOC_FAIL);
        return;
    }
    if (decx::alloc::_device_malloc(&dev_buffer2, dev_dst_size * sizeof(float4), true, S)) {
        Print_Error_Message(4, ALLOC_FAIL);
        return;
    }

    // strat the main loop
    if (ker_dim.x == (bounded_kernel_R16 * 2 + 1) && ker_dim.y == (bounded_kernel_R8 * 2 + 1)) {
        decx::dev_main_loop_sconv2_mk_exact8x16_BC(
            &Dsrc_alloc_dim, &Ddst_alloc_dim, &ker_dim, src, kernel, dst, dev_buffer1.ptr, dev_buffer2.ptr, S);
    }
    else {
        decx::dev_main_loop_sconv2_mk_within8x16_BC(
            &Dsrc_alloc_dim, &Ddst_alloc_dim, &ker_dim, src, kernel, dst, dev_buffer1.ptr, dev_buffer2.ptr, S);
    }
    checkCudaErrors(cudaDeviceSynchronize());
    decx::alloc::_device_dealloc(&dev_buffer1);
    decx::alloc::_device_dealloc(&dev_buffer2);

    S->detach();
}



// single kernel
static void decx::dev_Conv2_BC_R16x8_MK(decx::_GPU_MatrixArray* src, decx::_GPU_MatrixArray* kernel, decx::_GPU_MatrixArray* dst)
{
    int2 Dsrc_alloc_dim,
        Ddst_alloc_dim,
        ker_dim;

    ker_dim.x = kernel->width;            ker_dim.y = kernel->height;
    // first, allocate the memory according to R8 alignments
    Ddst_alloc_dim.x = decx::utils::ceil<int>(dst->width, 64) * bounded_kernel_R8 * 2;
    Ddst_alloc_dim.y = decx::utils::ceil<int>(dst->height, 16) * bounded_kernel_R16 * 2;

    Dsrc_alloc_dim.x = Ddst_alloc_dim.x + bounded_kernel_R8 / 2;        // bounded_kernel_R8 * 2 / 4
    Dsrc_alloc_dim.y = Ddst_alloc_dim.y + bounded_kernel_R16 * 2;

    size_t dev_src_size = static_cast<size_t>(Dsrc_alloc_dim.x) * static_cast<size_t>(Dsrc_alloc_dim.y);
    size_t dev_dst_size = static_cast<size_t>(Ddst_alloc_dim.x) * static_cast<size_t>(Ddst_alloc_dim.y);


    decx::cuda_stream* S = NULL;
    S = decx::CStream.stream_accessor_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        Print_Error_Message(4, CUDA_STREAM_ACCESS_FAIL);
        return;
    }
    decx::PtrInfo<float4> dev_buffer1, dev_buffer2;
    if (decx::alloc::_device_malloc(&dev_buffer1, dev_src_size * sizeof(float4), true, S)) {
        Print_Error_Message(4, ALLOC_FAIL);
        return;
    }
    if (decx::alloc::_device_malloc(&dev_buffer2, dev_dst_size * sizeof(float4), true, S)) {
        Print_Error_Message(4, ALLOC_FAIL);
        return;
    }

    // strat the main loop
    if (ker_dim.x == (bounded_kernel_R8 * 2 + 1) && ker_dim.y == (bounded_kernel_R16 * 2 + 1)) {
        decx::dev_main_loop_sconv2_mk_exact16x8_BC(
            &Dsrc_alloc_dim, &Ddst_alloc_dim, &ker_dim, src, kernel, dst, dev_buffer1.ptr, dev_buffer2.ptr, S);
    }
    else {
        decx::dev_main_loop_sconv2_mk_within16x8_BC(
            &Dsrc_alloc_dim, &Ddst_alloc_dim, &ker_dim, src, kernel, dst, dev_buffer1.ptr, dev_buffer2.ptr, S);
    }
    checkCudaErrors(cudaDeviceSynchronize());
    decx::alloc::_device_dealloc(&dev_buffer1);
    decx::alloc::_device_dealloc(&dev_buffer2);

    S->detach();
}





// single kernel
static void decx::dev_Conv2_BC_R16x16_MK(decx::_GPU_MatrixArray* src, decx::_GPU_MatrixArray* kernel, decx::_GPU_MatrixArray* dst)
{
    int2 Dsrc_alloc_dim,
        Ddst_alloc_dim,
        ker_dim;

    ker_dim.x = kernel->width;            ker_dim.y = kernel->height;
    // first, allocate the memory according to R8 alignments
    Ddst_alloc_dim.x = decx::utils::ceil<int>(dst->width, 64) * bounded_kernel_R16 * 2;
    Ddst_alloc_dim.y = decx::utils::ceil<int>(dst->height, 16) * bounded_kernel_R16 * 2;

    Dsrc_alloc_dim.x = Ddst_alloc_dim.x + bounded_kernel_R16 / 2;        // bounded_kernel_R8 * 2 / 4
    Dsrc_alloc_dim.y = Ddst_alloc_dim.y + bounded_kernel_R16 * 2;

    size_t dev_src_size = static_cast<size_t>(Dsrc_alloc_dim.x) * static_cast<size_t>(Dsrc_alloc_dim.y);
    size_t dev_dst_size = static_cast<size_t>(Ddst_alloc_dim.x) * static_cast<size_t>(Ddst_alloc_dim.y);


    decx::cuda_stream* S = NULL;
    S = decx::CStream.stream_accessor_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        Print_Error_Message(4, CUDA_STREAM_ACCESS_FAIL);
        return;
    }
    decx::PtrInfo<float4> dev_buffer1, dev_buffer2;
    if (decx::alloc::_device_malloc(&dev_buffer1, dev_src_size * sizeof(float4), true, S)) {
        Print_Error_Message(4, ALLOC_FAIL);
        return;
    }
    if (decx::alloc::_device_malloc(&dev_buffer2, dev_dst_size * sizeof(float4), true, S)) {
        Print_Error_Message(4, ALLOC_FAIL);
        return;
    }

    // strat the main loop
    if (ker_dim.x == (bounded_kernel_R16 * 2 + 1) && ker_dim.y == (bounded_kernel_R16 * 2 + 1)) {
        decx::dev_main_loop_sconv2_mk_exact16x16_BC(
            &Dsrc_alloc_dim, &Ddst_alloc_dim, &ker_dim, src, kernel, dst, dev_buffer1.ptr, dev_buffer2.ptr, S);
    }
    else {
        decx::dev_main_loop_sconv2_mk_within16x16_BC(
            &Dsrc_alloc_dim, &Ddst_alloc_dim, &ker_dim, src, kernel, dst, dev_buffer1.ptr, dev_buffer2.ptr, S);
    }
    checkCudaErrors(cudaDeviceSynchronize());
    decx::alloc::_device_dealloc(&dev_buffer1);
    decx::alloc::_device_dealloc(&dev_buffer2);

    S->detach();
}



// ******************************************************************************************************************************

static void decx::dev_sConv2_border_zero_mk(decx::_GPU_MatrixArray* src, decx::_GPU_MatrixArray* kernel, decx::_GPU_MatrixArray* dst, de::DH* handle)
{
    int2 half_ker_dim;
    half_ker_dim.x = kernel->width / 2;                half_ker_dim.y = kernel->height / 2;

    dst->re_construct(src->type, src->width, src->height, src->ArrayNumber);

    if (half_ker_dim.x < bounded_kernel_R8 + 1) {
        if (half_ker_dim.y < bounded_kernel_R8 + 1) {
            decx::dev_Conv2_BC_R8x8_MK(src, kernel, dst);
        }
        else {
            decx::dev_Conv2_BC_R16x8_MK(src, kernel, dst);
        }
    }
    else {
        if (half_ker_dim.y < bounded_kernel_R8 + 1) {
            decx::dev_Conv2_BC_R8x16_MK(src, kernel, dst);
        }
        else {
            decx::dev_Conv2_BC_R16x16_MK(src, kernel, dst);
        }
    }
    decx::Success(handle);
}


#endif