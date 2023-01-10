/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/

#ifndef _GPU_IFFT2D_SUB_FUNCTION_H_
#define _GPU_IFFT2D_SUB_FUNCTION_H_

#include "../CUDA_FFT_configs.h"
#include "../../../../classes/Matrix.h"
#include "../../../../core/configs/config.h"
#include "FFT2D_Radix_2_kernel.cuh"
#include "FFT2D_Radix_3_kernel.cuh"
#include "FFT2D_Radix_4_kernel.cuh"
#include "FFT2D_Radix_5_kernel.cuh"
#include "FFT2D_utils_kernel.cuh"


namespace decx
{
    namespace signal
    {
        static void GPU_IFFT2D_C2C_fp32_organizer(decx::_Matrix* src, decx::_Matrix* dst,
            decx::signal::CUDA_FFT_Configs _conf[2], de::DH* handle, decx::cuda_stream* S);


        static void GPU_IFFT2D_C2R_fp32_organizer(decx::_Matrix* src, decx::_Matrix* dst,
            decx::signal::CUDA_FFT_Configs _conf[2], de::DH* handle, decx::cuda_stream* S);
    }
}




static void decx::signal::GPU_IFFT2D_C2C_fp32_organizer(decx::_Matrix* src, decx::_Matrix* dst,
    decx::signal::CUDA_FFT_Configs _conf[2], de::DH* handle, decx::cuda_stream* S)
{
    const uint2 tmp_dims = make_uint2(dst->pitch,       // aligned to 4 already (4 de::CPf) -> (2 double2)
                                      decx::utils::ceil<uint>(src->height, 4) * 4);

    decx::PtrInfo<de::CPf> dev_tmp0, dev_tmp1;
    if (decx::alloc::_device_malloc(&dev_tmp0, tmp_dims.x * tmp_dims.y * sizeof(de::CPf), true, S)) {
        Print_Error_Message(4, ALLOC_FAIL);
        decx::err::AllocateFailure(handle);
        return;
    }
    if (decx::alloc::_device_malloc(&dev_tmp1, tmp_dims.x * tmp_dims.y * sizeof(de::CPf), true, S)) {
        Print_Error_Message(4, ALLOC_FAIL);
        decx::err::AllocateFailure(handle);
        return;
    }

    // copy the data from host to device
    checkCudaErrors(cudaMemcpy2DAsync(dev_tmp0.ptr, tmp_dims.x * sizeof(de::CPf), src->Mat.ptr, src->pitch * sizeof(de::CPf),
        src->width * sizeof(de::CPf), src->height, cudaMemcpyHostToDevice, S->get_raw_stream_ref()));

    decx::alloc::MIF<de::CPf> MIF_0(dev_tmp0.ptr, true), 
                              MIF_1(dev_tmp1.ptr, false);

    int current_base;
    size_t warp_proc_len = 1;

    // row FFT1D
    for (int i = 0; i < _conf[0]._base_num; ++i) {
        current_base = _conf[0]._base[i];
        warp_proc_len *= current_base;

        switch (current_base)
        {
        case 2:
            if (i == 0) {
                decx::signal::gpu::CUDA_FFT2D_R2_C2C_first_caller<true>(&MIF_0, &MIF_1, src->width,
                    tmp_dims.x, src->height, S);
                decx::utils::set_mutex_memory_state<de::CPf, de::CPf>(&MIF_1, &MIF_0); }
            else {
                decx::signal::gpu::CUDA_FFT2D_R2_C2C_caller(&MIF_0, &MIF_1, src->width, tmp_dims.x,
                    src->height, warp_proc_len, S); }
            break;
        case 3:
            if (i == 0) {
                decx::signal::gpu::CUDA_FFT2D_R3_C2C_first_caller<true>(&MIF_0, &MIF_1, src->width,
                    tmp_dims.x, src->height, S);
                decx::utils::set_mutex_memory_state<de::CPf, de::CPf>(&MIF_1, &MIF_0); }
            else {
                decx::signal::gpu::CUDA_FFT2D_R3_C2C_caller(&MIF_0, &MIF_1, src->width, tmp_dims.x,
                    src->height, warp_proc_len, S); }
            break;
        case 4:
            if (i == 0) {
                decx::signal::gpu::CUDA_FFT2D_R4_C2C_first_caller<true>(&MIF_0, &MIF_1, src->width, 
                    tmp_dims.x, src->height, S);
                decx::utils::set_mutex_memory_state<de::CPf, de::CPf>(&MIF_1, &MIF_0); }
            else {
                decx::signal::gpu::CUDA_FFT2D_R4_C2C_caller(&MIF_0, &MIF_1, src->width, tmp_dims.x,
                    src->height, warp_proc_len, S); }
            break;
        case 5:
            if (i == 0) {
                decx::signal::gpu::CUDA_FFT2D_R5_C2C_first_caller<true>(&MIF_0, &MIF_1, src->width,
                    tmp_dims.x, src->height, S);
                decx::utils::set_mutex_memory_state<de::CPf, de::CPf>(&MIF_1, &MIF_0);
            }
            else {
                decx::signal::gpu::CUDA_FFT2D_R5_C2C_caller(&MIF_0, &MIF_1, src->width, tmp_dims.x,
                    src->height, warp_proc_len, S);
            }
            break;
        default:
            break;
        }
    }

    decx::signal::utils::gpu::_FFT2D_transpose_C2C_Async(&MIF_0, &MIF_1,
        tmp_dims.x / 2, tmp_dims.y / 2, make_uint2(tmp_dims.y, tmp_dims.x), S);

    warp_proc_len = 1;

    // col FFT1D
    for (int i = 0; i < _conf[1]._base_num; ++i) {
        current_base = _conf[1]._base[i];
        warp_proc_len *= current_base;

        switch (current_base)
        {
        case 2:
            if (i == 0) {
                decx::signal::gpu::CUDA_FFT2D_R2_C2C_first_caller<false>(&MIF_0, &MIF_1, src->height,
                    tmp_dims.y, src->width, S); }
            else {
                decx::signal::gpu::CUDA_FFT2D_R2_C2C_caller(&MIF_0, &MIF_1, src->height, tmp_dims.y,
                    src->width, warp_proc_len, S); }
            break;
        case 3:
            if (i == 0) {
                decx::signal::gpu::CUDA_FFT2D_R3_C2C_first_caller<false>(&MIF_0, &MIF_1, src->height,
                    tmp_dims.y, src->width, S); }
            else {
                decx::signal::gpu::CUDA_FFT2D_R3_C2C_caller(&MIF_0, &MIF_1, src->height, tmp_dims.y,
                    src->width, warp_proc_len, S); }
            break;
        case 4:
            if (i == 0) {
                decx::signal::gpu::CUDA_FFT2D_R4_C2C_first_caller<false>(&MIF_0, &MIF_1, src->height,
                    tmp_dims.y, src->width, S); }
            else {
                decx::signal::gpu::CUDA_FFT2D_R4_C2C_caller(&MIF_0, &MIF_1, src->height, tmp_dims.y,
                    src->width, warp_proc_len, S); }
            break;
        case 5:
            if (i == 0) {
                decx::signal::gpu::CUDA_FFT2D_R5_C2C_first_caller<false>(&MIF_0, &MIF_1, src->height,
                    tmp_dims.y, src->width, S);
            }
            else {
                decx::signal::gpu::CUDA_FFT2D_R5_C2C_caller(&MIF_0, &MIF_1, src->height, tmp_dims.y,
                    src->width, warp_proc_len, S);
            }
            break;
        default:
            break;
        }
    }

    decx::signal::utils::gpu::_FFT2D_transpose_C2C_div_Async(&MIF_0, &MIF_1,
        tmp_dims.y / 2, tmp_dims.x / 2, make_uint2(tmp_dims.x, tmp_dims.y), src->height, S);

    if (MIF_0.leading) {
        // copy back the data from device to host
        checkCudaErrors(cudaMemcpy2DAsync(dst->Mat.ptr, dst->pitch * sizeof(de::CPf), dev_tmp0.ptr, 
            tmp_dims.x * sizeof(de::CPf), dst->width * sizeof(de::CPf), dst->height, cudaMemcpyDeviceToHost, 
            S->get_raw_stream_ref()));
    }
    else {
        // copy back the data from device to host
        checkCudaErrors(cudaMemcpy2DAsync(dst->Mat.ptr, dst->pitch * sizeof(de::CPf), dev_tmp1.ptr,
            tmp_dims.x * sizeof(de::CPf), dst->width * sizeof(de::CPf), dst->height, cudaMemcpyDeviceToHost,
            S->get_raw_stream_ref()));
    }

    checkCudaErrors(cudaDeviceSynchronize());
    decx::alloc::_device_dealloc(&dev_tmp0);
    decx::alloc::_device_dealloc(&dev_tmp1);
}



static void decx::signal::GPU_IFFT2D_C2R_fp32_organizer(decx::_Matrix* src, decx::_Matrix* dst,
    decx::signal::CUDA_FFT_Configs _conf[2], de::DH* handle, decx::cuda_stream* S)
{
    const uint2 tmp_dims = make_uint2(dst->pitch,       // aligned to 4 already (4 de::CPf) -> (2 double2)
                                      decx::utils::ceil<uint>(src->height, 4) * 4);

    decx::PtrInfo<de::CPf> dev_tmp0, dev_tmp1;
    if (decx::alloc::_device_malloc(&dev_tmp0, tmp_dims.x * tmp_dims.y * sizeof(de::CPf), true, S)) {
        Print_Error_Message(4, ALLOC_FAIL);
        decx::err::AllocateFailure(handle);
        return;
    }
    if (decx::alloc::_device_malloc(&dev_tmp1, tmp_dims.x * tmp_dims.y * sizeof(de::CPf), true, S)) {
        Print_Error_Message(4, ALLOC_FAIL);
        decx::err::AllocateFailure(handle);
        return;
    }

    // copy the data from host to device
    checkCudaErrors(cudaMemcpy2DAsync(dev_tmp0.ptr, tmp_dims.x * sizeof(de::CPf), src->Mat.ptr, src->pitch * sizeof(de::CPf),
        src->width * sizeof(de::CPf), src->height, cudaMemcpyHostToDevice, S->get_raw_stream_ref()));

    decx::alloc::MIF<de::CPf> MIF_0(dev_tmp0.ptr, true), 
                              MIF_1(dev_tmp1.ptr, false);

    int current_base;
    size_t warp_proc_len = 1;

    // row FFT1D
    for (int i = 0; i < _conf[0]._base_num;) {
        current_base = _conf[0]._base[i];
        warp_proc_len *= current_base;

        switch (current_base)
        {
        case 2:
            if (i == 0) {
                decx::signal::gpu::CUDA_FFT2D_R2_C2C_first_caller<true>(&MIF_0, &MIF_1, src->width,
                    tmp_dims.x, src->height, S); }
            else {
                decx::signal::gpu::CUDA_FFT2D_R2_C2C_caller(&MIF_0, &MIF_1, src->width, tmp_dims.x,
                    src->height, warp_proc_len, S); }
            break;
        case 3:
            if (i == 0) {
                decx::signal::gpu::CUDA_FFT2D_R3_C2C_first_caller<true>(&MIF_0, &MIF_1, src->width,
                    tmp_dims.x, src->height, S); }
            else {
                decx::signal::gpu::CUDA_FFT2D_R3_C2C_caller(&MIF_0, &MIF_1, src->width, tmp_dims.x,
                    src->height, warp_proc_len, S); }
            break;
        case 4:
            if (i == 0) {
                decx::signal::gpu::CUDA_FFT2D_R4_C2C_first_caller<true>(&MIF_0, &MIF_1, src->width, 
                    tmp_dims.x, src->height, S); }
            else {
                decx::signal::gpu::CUDA_FFT2D_R4_C2C_caller(&MIF_0, &MIF_1, src->width, tmp_dims.x,
                    src->height, warp_proc_len, S); }
            break;
        case 5:
            if (i == 0) {
                decx::signal::gpu::CUDA_FFT2D_R5_C2C_first_caller<true>(&MIF_0, &MIF_1, src->width,
                    tmp_dims.x, src->height, S); }
            else {
                decx::signal::gpu::CUDA_FFT2D_R5_C2C_caller(&MIF_0, &MIF_1, src->width, tmp_dims.x,
                    src->height, warp_proc_len, S); }
            break;
        default:
            break;
        }
        ++i;
    }

    decx::signal::utils::gpu::_FFT2D_transpose_C2C_Async(&MIF_0, &MIF_1,
        tmp_dims.x / 2, tmp_dims.y / 2, make_uint2(tmp_dims.y, tmp_dims.x), S);

    warp_proc_len = 1;

    // col FFT1D
    for (int i = 0; i < _conf[1]._base_num;) {
        current_base = _conf[1]._base[i];
        warp_proc_len *= current_base;

        switch (current_base)
        {
        case 2:
            if (i != _conf[1]._base_num - 1 && i == 0) {
                decx::signal::gpu::CUDA_FFT2D_R2_C2C_first_caller<false>(&MIF_0, &MIF_1, src->height,
                    tmp_dims.y, src->width, S); }
            else if (i == _conf[1]._base_num - 1 && i != 0) {
                decx::signal::gpu::CUDA_IFFT2D_R2_C2R_last_caller(&MIF_0, &MIF_1, src->height, tmp_dims.y,
                    src->width, warp_proc_len, S); }
            else if (i == _conf[1]._base_num - 1 && i == 0) {
                decx::signal::gpu::CUDA_IFFT2D_R2_C2R_once_caller(&MIF_0, &MIF_1, src->height, tmp_dims.y,
                    src->width, S); }
            else {
                decx::signal::gpu::CUDA_FFT2D_R2_C2C_caller(&MIF_0, &MIF_1, src->height, tmp_dims.y,
                    src->width, warp_proc_len, S); }
            break;
        case 3:
            if (i != _conf[1]._base_num - 1 && i == 0) {
                decx::signal::gpu::CUDA_FFT2D_R3_C2C_first_caller<false>(&MIF_0, &MIF_1, src->height,
                    tmp_dims.y, src->width, S); }
            else if (i == _conf[1]._base_num - 1 && i != 0) {
                decx::signal::gpu::CUDA_IFFT2D_R3_C2R_last_caller(&MIF_0, &MIF_1, src->height, tmp_dims.y,
                    src->width, warp_proc_len, S); }
            else if (i == _conf[1]._base_num - 1 && i == 0) {
                decx::signal::gpu::CUDA_IFFT2D_R3_C2R_once_caller(&MIF_0, &MIF_1, src->height, tmp_dims.y,
                    src->width, S); }
            else {
                decx::signal::gpu::CUDA_FFT2D_R3_C2C_caller(&MIF_0, &MIF_1, src->height, tmp_dims.y,
                    src->width, warp_proc_len, S); }
            break;
        case 4:
            if (i != _conf[1]._base_num - 1 && i == 0) {
                decx::signal::gpu::CUDA_FFT2D_R4_C2C_first_caller<false>(&MIF_0, &MIF_1, src->height,
                    tmp_dims.y, src->width, S); }
            else if (i == _conf[1]._base_num - 1 && i != 0) {
                decx::signal::gpu::CUDA_IFFT2D_R4_C2R_last_caller(&MIF_0, &MIF_1, src->height, tmp_dims.y,
                    src->width, warp_proc_len, S); }
            else if (i == _conf[1]._base_num - 1 && i == 0) {
                decx::signal::gpu::CUDA_IFFT2D_R4_C2R_once_caller(&MIF_0, &MIF_1, src->height, tmp_dims.y,
                    src->width, S); }
            else {
                decx::signal::gpu::CUDA_FFT2D_R4_C2C_caller(&MIF_0, &MIF_1, src->height, tmp_dims.y,
                    src->width, warp_proc_len, S); }
            break;
        case 5:
            if (i != _conf[1]._base_num - 1 && i == 0) {
                decx::signal::gpu::CUDA_FFT2D_R5_C2C_first_caller<false>(&MIF_0, &MIF_1, src->height,
                    tmp_dims.y, src->width, S); }
            else if (i == _conf[1]._base_num - 1 && i != 0) {
                decx::signal::gpu::CUDA_IFFT2D_R5_C2R_last_caller(&MIF_0, &MIF_1, src->height, tmp_dims.y,
                    src->width, warp_proc_len, S); }
            else if (i == _conf[1]._base_num - 1 && i == 0) {
                decx::signal::gpu::CUDA_IFFT2D_R5_C2R_once_caller(&MIF_0, &MIF_1, src->height, tmp_dims.y,
                    src->width, S); }
            else {
                decx::signal::gpu::CUDA_FFT2D_R5_C2C_caller(&MIF_0, &MIF_1, src->height, tmp_dims.y,
                    src->width, warp_proc_len, S);
            }
            break;
        default:
            break;
        }
        ++i;
    }

    decx::signal::utils::gpu::_IFFT2D_transpose_R2R_Async(&MIF_0, &MIF_1,
        tmp_dims.y / 2, tmp_dims.x / 2, make_uint2(tmp_dims.x, tmp_dims.y), S);

    if (MIF_0.leading) {
        // copy back the data from device to host
        checkCudaErrors(cudaMemcpy2DAsync(dst->Mat.ptr, dst->pitch * sizeof(float), dev_tmp0.ptr, 
            tmp_dims.x * sizeof(de::CPf), dst->width * sizeof(float), dst->height, cudaMemcpyDeviceToHost, 
            S->get_raw_stream_ref()));
    }
    else {
        // copy back the data from device to host
        checkCudaErrors(cudaMemcpy2DAsync(dst->Mat.ptr, dst->pitch * sizeof(float), dev_tmp1.ptr,
            tmp_dims.x * sizeof(de::CPf), dst->width * sizeof(float), dst->height, cudaMemcpyDeviceToHost,
            S->get_raw_stream_ref()));
    }

    checkCudaErrors(cudaDeviceSynchronize());
    decx::alloc::_device_dealloc(&dev_tmp0);
    decx::alloc::_device_dealloc(&dev_tmp1);
}


#endif