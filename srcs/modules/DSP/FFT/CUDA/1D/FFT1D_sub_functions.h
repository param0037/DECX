/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _GPU_FFT1D_SUB_FUNCTION_H_
#define _GPU_FFT1D_SUB_FUNCTION_H_

#include "../CUDA_FFT_configs.h"
#include "../../../../classes/Vector.h"
#include "../../../../core/configs/config.h"
#include "FFT1D_Radix_2_kernel.cuh"
#include "FFT1D_Radix_3_kernel.cuh"
#include "FFT1D_Radix_4_kernel.cuh"
#include "FFT1D_Radix_5_kernel.cuh"


namespace decx
{
    namespace signal
    {
        static void GPU_FFT1D_R2C_fp32_organizer(decx::_Vector* src, decx::_Vector* dst, 
            decx::signal::CUDA_FFT_Configs* _conf, de::DH* handle, decx::cuda_stream* S, decx::cuda_event* E);


        static void GPU_FFT1D_C2C_fp32_organizer(decx::_Vector* src, decx::_Vector* dst,
            decx::signal::CUDA_FFT_Configs* _conf, de::DH* handle, decx::cuda_stream* S, decx::cuda_event* E);
    }
}


static void decx::signal::GPU_FFT1D_R2C_fp32_organizer(decx::_Vector* src, decx::_Vector* dst,
    decx::signal::CUDA_FFT_Configs* _conf, de::DH* handle, decx::cuda_stream* S, decx::cuda_event* E)
{
    decx::PtrInfo<de::CPf> dev_tmp0, dev_tmp1;
    if (decx::alloc::_device_malloc(&dev_tmp0, dst->_length * sizeof(de::CPf), true, S)) {
        Print_Error_Message(4, ALLOC_FAIL);
        decx::err::AllocateFailure(handle);
        return;
    }
    if (decx::alloc::_device_malloc(&dev_tmp1, dst->_length * sizeof(de::CPf), true, S)) {
        Print_Error_Message(4, ALLOC_FAIL);
        decx::err::AllocateFailure(handle);
        return;
    }

    checkCudaErrors(cudaMemcpyAsync(
        dev_tmp0.ptr, src->Vec.ptr, src->length * sizeof(float), cudaMemcpyHostToDevice, S->get_raw_stream_ref()));

    decx::alloc::MIF<de::CPf> MIF_0, MIF_1;
    MIF_0.mem = dev_tmp0.ptr;
    MIF_1.mem = dev_tmp1.ptr;

    int current_base, count = 0;
    size_t warp_proc_len = 1;

    for (int i = 0; i < _conf->_base_num; ++i) {
        current_base = _conf->_base[i];
        warp_proc_len *= current_base;
        switch (current_base)
        {
        case 2:
            if (count == 0) {
                decx::signal::CUDA_FFT1D_R2_R2C_first_caller(MIF_0.mem, MIF_1.mem, src->length, S);
                decx::utils::set_mutex_memory_state<de::CPf, de::CPf>(&MIF_1, &MIF_0);
            }
            else {
                decx::signal::CUDA_FFT1D_R2_C2C_caller(&MIF_0, &MIF_1, src->length, warp_proc_len, S);
            }
            break;

        case 3:
            if (count == 0) {
                decx::signal::CUDA_FFT1D_R3_R2C_first_caller(MIF_0.mem, MIF_1.mem, src->length, S);
                decx::utils::set_mutex_memory_state<de::CPf, de::CPf>(&MIF_1, &MIF_0);
            }
            else {
                decx::signal::CUDA_FFT1D_R3_C2C_caller(&MIF_0, &MIF_1, src->length, warp_proc_len, S);
            }
            break;

        case 4:
            if (count == 0) {
                decx::signal::CUDA_FFT1D_R4_R2C_first_caller((float*)MIF_0.mem, MIF_1.mem, src->length, S);
                decx::utils::set_mutex_memory_state<de::CPf, de::CPf>(&MIF_1, &MIF_0);
            }
            else {
                decx::signal::CUDA_FFT1D_R4_C2C_caller(&MIF_0, &MIF_1, src->length, warp_proc_len, S);
            }
            break;

        case 5:
            if (count == 0) {
                decx::signal::CUDA_FFT1D_R5_R2C_first_caller((float*)MIF_0.mem, MIF_1.mem, src->length, S);
                decx::utils::set_mutex_memory_state<de::CPf, de::CPf>(&MIF_1, &MIF_0);
            }
            else {
                decx::signal::CUDA_FFT1D_R5_C2C_caller(&MIF_0, &MIF_1, src->length, warp_proc_len, S);
            }
            break;

        default:
            break;
        }
        ++count;
    }

    if (MIF_0.leading) {
        checkCudaErrors(cudaMemcpyAsync(
            dst->Vec.ptr, MIF_0.mem, dst->length * sizeof(de::CPf), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));
    }
    else {
        checkCudaErrors(cudaMemcpyAsync(
            dst->Vec.ptr, MIF_1.mem, dst->length * sizeof(de::CPf), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));
    }
    
    E->event_record(S);
    E->synchronize();

    decx::alloc::_device_dealloc(&dev_tmp0);
    decx::alloc::_device_dealloc(&dev_tmp1);
}



static void decx::signal::GPU_FFT1D_C2C_fp32_organizer(decx::_Vector* src, decx::_Vector* dst,
    decx::signal::CUDA_FFT_Configs* _conf, de::DH* handle, decx::cuda_stream* S, decx::cuda_event* E)
{
    decx::PtrInfo<de::CPf> dev_tmp0, dev_tmp1;
    if (decx::alloc::_device_malloc(&dev_tmp0, dst->_length * sizeof(de::CPf), true, S)) {
        Print_Error_Message(4, ALLOC_FAIL);
        decx::err::AllocateFailure(handle);
        return;
    }
    if (decx::alloc::_device_malloc(&dev_tmp1, dst->_length * sizeof(de::CPf), true, S)) {
        Print_Error_Message(4, ALLOC_FAIL);
        decx::err::AllocateFailure(handle);
        return;
    }

    checkCudaErrors(cudaMemcpyAsync(
        dev_tmp0.ptr, src->Vec.ptr, src->length * sizeof(de::CPf), cudaMemcpyHostToDevice, S->get_raw_stream_ref()));

    decx::alloc::MIF<de::CPf> MIF_0, MIF_1;
    MIF_0.mem = dev_tmp0.ptr;
    MIF_1.mem = dev_tmp1.ptr;

    int current_base, count = 0;
    size_t warp_proc_len = 1;

    for (int i = 0; i < _conf->_base_num; ++i) {
        current_base = _conf->_base[i];
        warp_proc_len *= current_base;
        switch (current_base)
        {
        case 2:
            if (count == 0) {
                decx::signal::CUDA_FFT1D_R2_C2C_first_caller<false>(MIF_0.mem, MIF_1.mem, src->length, S);
                decx::utils::set_mutex_memory_state<de::CPf, de::CPf>(&MIF_1, &MIF_0);
            }
            else {
                decx::signal::CUDA_FFT1D_R2_C2C_caller(&MIF_0, &MIF_1, src->length, warp_proc_len, S);
            }
            break;

        case 3:
            if (count == 0) {
                decx::signal::CUDA_FFT1D_R3_C2C_first_caller<false>(MIF_0.mem, MIF_1.mem, src->length, S);
                decx::utils::set_mutex_memory_state<de::CPf, de::CPf>(&MIF_1, &MIF_0);
            }
            else {
                decx::signal::CUDA_FFT1D_R3_C2C_caller(&MIF_0, &MIF_1, src->length, warp_proc_len, S);
            }
            break;

        case 4:
            if (count == 0) {
                decx::signal::CUDA_FFT1D_R4_C2C_first_caller<false>(MIF_0.mem, MIF_1.mem, src->length, S);
                decx::utils::set_mutex_memory_state<de::CPf, de::CPf>(&MIF_1, &MIF_0);
            }
            else {
                decx::signal::CUDA_FFT1D_R4_C2C_caller(&MIF_0, &MIF_1, src->length, warp_proc_len, S);
            }
            break;

        case 5:
            if (count == 0) {
                decx::signal::CUDA_FFT1D_R5_C2C_first_caller<false>(MIF_0.mem, MIF_1.mem, src->length, S);
                decx::utils::set_mutex_memory_state<de::CPf, de::CPf>(&MIF_1, &MIF_0);
            }
            else {
                decx::signal::CUDA_FFT1D_R5_C2C_caller(&MIF_0, &MIF_1, src->length, warp_proc_len, S);
            }
            break;

        default:
            break;
        }
        ++count;
    }

    if (MIF_0.leading) {
        checkCudaErrors(cudaMemcpyAsync(
            dst->Vec.ptr, MIF_0.mem, dst->length * sizeof(de::CPf), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));
    }
    else {
        checkCudaErrors(cudaMemcpyAsync(
            dst->Vec.ptr, MIF_1.mem, dst->length * sizeof(de::CPf), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));
    }
    
    E->event_record(S);
    E->synchronize();

    decx::alloc::_device_dealloc(&dev_tmp0);
    decx::alloc::_device_dealloc(&dev_tmp1);
}




#endif