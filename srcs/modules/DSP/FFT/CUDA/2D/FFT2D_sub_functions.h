/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _GPU_FFT2D_SUB_FUNCTION_H_
#define _GPU_FFT2D_SUB_FUNCTION_H_

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
        static void GPU_FFT2D_R2C_fp32_organizer(decx::_Matrix* src, decx::_Matrix* dst,
            decx::signal::CUDA_FFT_Configs _conf[2], de::DH* handle, decx::cuda_stream* S, decx::cuda_event* E);


        static void GPU_FFT2D_C2C_fp32_organizer(decx::_Matrix* src, decx::_Matrix* dst,
            decx::signal::CUDA_FFT_Configs _conf[2], de::DH* handle, decx::cuda_stream* S, decx::cuda_event* E);
    }
}




static void decx::signal::GPU_FFT2D_R2C_fp32_organizer(decx::_Matrix* src, decx::_Matrix* dst,
    decx::signal::CUDA_FFT_Configs _conf[2], de::DH* handle, decx::cuda_stream* S, decx::cuda_event* E)
{
    const uint2 tmp_dims = make_uint2(dst->Pitch(),       // aligned to 4 already (4 de::CPf) -> (2 double2)
                                      decx::utils::ceil<uint>(src->Height(), 4) * 4);

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
    checkCudaErrors(cudaMemcpy2DAsync(dev_tmp0.ptr, tmp_dims.x * sizeof(de::CPf), src->Mat.ptr, src->Pitch() * sizeof(float),
        src->Width() * sizeof(float), src->Height(), cudaMemcpyHostToDevice, S->get_raw_stream_ref()));

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
                decx::signal::gpu::CUDA_FFT2D_R2_R2C_first_caller(&MIF_0, &MIF_1, src->Width(),
                    tmp_dims.x * 2, tmp_dims.x, src->Height(), S); }
            else {
                decx::signal::gpu::CUDA_FFT2D_R2_C2C_caller(&MIF_0, &MIF_1, src->Width(), tmp_dims.x,
                    src->Height(), warp_proc_len, S); }
            break;
        case 3:
            if (i == 0) {
                decx::signal::gpu::CUDA_FFT2D_R3_R2C_first_caller(&MIF_0, &MIF_1, src->Width(),
                    tmp_dims.x * 2, tmp_dims.x, src->Height(), S); }
            else {
                decx::signal::gpu::CUDA_FFT2D_R3_C2C_caller(&MIF_0, &MIF_1, src->Width(), tmp_dims.x,
                    src->Height(), warp_proc_len, S); }
            break;
        case 4:
            if (i == 0) {
                decx::signal::gpu::CUDA_FFT2D_R4_R2C_first_caller(&MIF_0, &MIF_1, src->Width(), 
                    tmp_dims.x * 2, tmp_dims.x, src->Height(), S); }
            else {
                decx::signal::gpu::CUDA_FFT2D_R4_C2C_caller(&MIF_0, &MIF_1, src->Width(), tmp_dims.x,
                    src->Height(), warp_proc_len, S);
            }
            break;
        case 5:
            if (i == 0) {
                decx::signal::gpu::CUDA_FFT2D_R5_R2C_first_caller(&MIF_0, &MIF_1, src->Width(),
                    tmp_dims.x * 2, tmp_dims.x, src->Height(), S); }
            else {
                decx::signal::gpu::CUDA_FFT2D_R5_C2C_caller(&MIF_0, &MIF_1, src->Width(), tmp_dims.x,
                    src->Height(), warp_proc_len, S);
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
                decx::signal::gpu::CUDA_FFT2D_R2_C2C_first_caller<false>(&MIF_0, &MIF_1, src->Height(),
                    tmp_dims.y, src->Width(), S); }
            else {
                decx::signal::gpu::CUDA_FFT2D_R2_C2C_caller(&MIF_0, &MIF_1, src->Height(), tmp_dims.y,
                    src->Width(), warp_proc_len, S); }
            break;
        case 3:
            if (i == 0) {
                decx::signal::gpu::CUDA_FFT2D_R3_C2C_first_caller<false>(&MIF_0, &MIF_1, src->Height(),
                    tmp_dims.y, src->Width(), S); }
            else {
                decx::signal::gpu::CUDA_FFT2D_R3_C2C_caller(&MIF_0, &MIF_1, src->Height(), tmp_dims.y,
                    src->Width(), warp_proc_len, S); }
            break;
        case 4:
            if (i == 0) {
                decx::signal::gpu::CUDA_FFT2D_R4_C2C_first_caller<false>(&MIF_0, &MIF_1, src->Height(),
                    tmp_dims.y, src->Width(), S); }
            else {
                decx::signal::gpu::CUDA_FFT2D_R4_C2C_caller(&MIF_0, &MIF_1, src->Height(), tmp_dims.y,
                    src->Width(), warp_proc_len, S); }
            break;
        case 5:
            if (i == 0) {
                decx::signal::gpu::CUDA_FFT2D_R5_C2C_first_caller<false>(&MIF_0, &MIF_1, src->Height(),
                    tmp_dims.y, src->Width(), S);
            }
            else {
                decx::signal::gpu::CUDA_FFT2D_R5_C2C_caller(&MIF_0, &MIF_1, src->Height(), tmp_dims.y,
                    src->Width(), warp_proc_len, S);
            }
            break;
        default:
            break;
        }
    }

    decx::signal::utils::gpu::_FFT2D_transpose_C2C_Async(&MIF_0, &MIF_1,
        tmp_dims.y / 2, tmp_dims.x / 2, make_uint2(tmp_dims.x, tmp_dims.y), S);

    if (MIF_0.leading) {
        // copy back the data from device to host
        checkCudaErrors(cudaMemcpy2DAsync(dst->Mat.ptr, dst->Pitch() * sizeof(de::CPf), dev_tmp0.ptr, 
            tmp_dims.x * sizeof(de::CPf), dst->Width() * sizeof(de::CPf), dst->Height(), cudaMemcpyDeviceToHost, 
            S->get_raw_stream_ref()));
    }
    else {
        // copy back the data from device to host
        checkCudaErrors(cudaMemcpy2DAsync(dst->Mat.ptr, dst->Pitch() * sizeof(de::CPf), dev_tmp1.ptr,
            tmp_dims.x * sizeof(de::CPf), dst->Width() * sizeof(de::CPf), dst->Height(), cudaMemcpyDeviceToHost,
            S->get_raw_stream_ref()));
    }

    E->event_record(S);
    E->synchronize();
    decx::alloc::_device_dealloc(&dev_tmp0);
    decx::alloc::_device_dealloc(&dev_tmp1);
}



static void decx::signal::GPU_FFT2D_C2C_fp32_organizer(decx::_Matrix* src, decx::_Matrix* dst,
    decx::signal::CUDA_FFT_Configs _conf[2], de::DH* handle, decx::cuda_stream* S, decx::cuda_event* E)
{
    const uint2 tmp_dims = make_uint2(dst->Pitch(),       // aligned to 4 already (4 de::CPf) -> (2 double2)
        decx::utils::ceil<uint>(src->Height(), 4) * 4);

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
    checkCudaErrors(cudaMemcpy2DAsync(dev_tmp0.ptr, tmp_dims.x * sizeof(de::CPf), src->Mat.ptr, src->Pitch() * sizeof(de::CPf),
        src->Width() * sizeof(de::CPf), src->Height(), cudaMemcpyHostToDevice, S->get_raw_stream_ref()));

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
                decx::signal::gpu::CUDA_FFT2D_R2_C2C_first_caller<false>(&MIF_0, &MIF_1, src->Width(),
                    tmp_dims.x, src->Height(), S);
                decx::utils::set_mutex_memory_state<de::CPf, de::CPf>(&MIF_1, &MIF_0);
            }
            else {
                decx::signal::gpu::CUDA_FFT2D_R2_C2C_caller(&MIF_0, &MIF_1, src->Width(), tmp_dims.x,
                    src->Height(), warp_proc_len, S);
            }
            break;
        case 3:
            if (i == 0) {
                decx::signal::gpu::CUDA_FFT2D_R3_C2C_first_caller<false>(&MIF_0, &MIF_1, src->Width(),
                    tmp_dims.x, src->Height(), S);
                decx::utils::set_mutex_memory_state<de::CPf, de::CPf>(&MIF_1, &MIF_0);
            }
            else {
                decx::signal::gpu::CUDA_FFT2D_R3_C2C_caller(&MIF_0, &MIF_1, src->Width(), tmp_dims.x,
                    src->Height(), warp_proc_len, S);
            }
            break;
        case 4:
            if (i == 0) {
                decx::signal::gpu::CUDA_FFT2D_R4_C2C_first_caller<false>(&MIF_0, &MIF_1, src->Width(),
                    tmp_dims.x, src->Height(), S);
                decx::utils::set_mutex_memory_state<de::CPf, de::CPf>(&MIF_1, &MIF_0);
            }
            else {
                decx::signal::gpu::CUDA_FFT2D_R4_C2C_caller(&MIF_0, &MIF_1, src->Width(), tmp_dims.x,
                    src->Height(), warp_proc_len, S);
            }
            break;
        case 5:
            if (i == 0) {
                decx::signal::gpu::CUDA_FFT2D_R5_C2C_first_caller<false>(&MIF_0, &MIF_1, src->Width(),
                    tmp_dims.x, src->Height(), S);
                decx::utils::set_mutex_memory_state<de::CPf, de::CPf>(&MIF_1, &MIF_0);
            }
            else {
                decx::signal::gpu::CUDA_FFT2D_R5_C2C_caller(&MIF_0, &MIF_1, src->Width(), tmp_dims.x,
                    src->Height(), warp_proc_len, S);
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
                decx::signal::gpu::CUDA_FFT2D_R2_C2C_first_caller<false>(&MIF_0, &MIF_1, src->Height(),
                    tmp_dims.y, src->Width(), S);
            }
            else {
                decx::signal::gpu::CUDA_FFT2D_R2_C2C_caller(&MIF_0, &MIF_1, src->Height(), tmp_dims.y,
                    src->Width(), warp_proc_len, S);
            }
            break;
        case 3:
            if (i == 0) {
                decx::signal::gpu::CUDA_FFT2D_R3_C2C_first_caller<false>(&MIF_0, &MIF_1, src->Height(),
                    tmp_dims.y, src->Width(), S);
            }
            else {
                decx::signal::gpu::CUDA_FFT2D_R3_C2C_caller(&MIF_0, &MIF_1, src->Height(), tmp_dims.y,
                    src->Width(), warp_proc_len, S);
            }
            break;
        case 4:
            if (i == 0) {
                decx::signal::gpu::CUDA_FFT2D_R4_C2C_first_caller<false>(&MIF_0, &MIF_1, src->Height(),
                    tmp_dims.y, src->Width(), S);
            }
            else {
                decx::signal::gpu::CUDA_FFT2D_R4_C2C_caller(&MIF_0, &MIF_1, src->Height(), tmp_dims.y,
                    src->Width(), warp_proc_len, S);
            }
            break;
        case 5:
            if (i == 0) {
                decx::signal::gpu::CUDA_FFT2D_R5_C2C_first_caller<false>(&MIF_0, &MIF_1, src->Height(),
                    tmp_dims.y, src->Width(), S);
            }
            else {
                decx::signal::gpu::CUDA_FFT2D_R5_C2C_caller(&MIF_0, &MIF_1, src->Height(), tmp_dims.y,
                    src->Width(), warp_proc_len, S);
            }
            break;
        default:
            break;
        }
    }

    decx::signal::utils::gpu::_FFT2D_transpose_C2C_Async(&MIF_0, &MIF_1,
        tmp_dims.y / 2, tmp_dims.x / 2, make_uint2(tmp_dims.x, tmp_dims.y), S);

    if (MIF_0.leading) {
        // copy back the data from device to host
        checkCudaErrors(cudaMemcpy2DAsync(dst->Mat.ptr, dst->Pitch() * sizeof(de::CPf), dev_tmp0.ptr,
            tmp_dims.x * sizeof(de::CPf), dst->Width() * sizeof(de::CPf), dst->Height(), cudaMemcpyDeviceToHost,
            S->get_raw_stream_ref()));
    }
    else {
        // copy back the data from device to host
        checkCudaErrors(cudaMemcpy2DAsync(dst->Mat.ptr, dst->Pitch() * sizeof(de::CPf), dev_tmp1.ptr,
            tmp_dims.x * sizeof(de::CPf), dst->Width() * sizeof(de::CPf), dst->Height(), cudaMemcpyDeviceToHost,
            S->get_raw_stream_ref()));
    }

    E->event_record(S);
    E->synchronize();
    decx::alloc::_device_dealloc(&dev_tmp0);
    decx::alloc::_device_dealloc(&dev_tmp1);
}


#endif