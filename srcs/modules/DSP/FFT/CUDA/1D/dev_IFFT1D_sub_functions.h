/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _dev_GPU_IFFT1D_SUB_FUNCTION_H_
#define _dev_GPU_IFFT1D_SUB_FUNCTION_H_

#include "../CUDA_FFT_configs.h"
#include "../../../../classes/GPU_Vector.h"
#include "../../../../core/configs/config.h"
#include "FFT1D_Radix_2_kernel.cuh"
#include "FFT1D_Radix_3_kernel.cuh"
#include "FFT1D_Radix_4_kernel.cuh"
#include "FFT1D_Radix_5_kernel.cuh"


namespace decx
{
    namespace signal
    {
        static void dev_GPU_IFFT1D_C2C_fp32_organizer(decx::_GPU_Vector* src, decx::_GPU_Vector* dst, 
            decx::signal::CUDA_FFT_Configs* _conf, de::DH* handle, decx::cuda_stream* S, decx::cuda_event* E);


        static void dev_GPU_IFFT1D_C2R_fp32_organizer(decx::_GPU_Vector* src, decx::_GPU_Vector* dst,
            decx::signal::CUDA_FFT_Configs* _conf, de::DH* handle, decx::cuda_stream* S, decx::cuda_event* E);
    }
}


static void decx::signal::dev_GPU_IFFT1D_C2C_fp32_organizer(decx::_GPU_Vector* src, decx::_GPU_Vector* dst,
    decx::signal::CUDA_FFT_Configs* _conf, de::DH* handle, decx::cuda_stream* S, decx::cuda_event* E)
{
    decx::PtrInfo<de::CPf> dev_tmp0, dev_tmp1;
    if (decx::alloc::_device_malloc(&dev_tmp0, dst->_length * sizeof(de::CPf), true, S)) {
        decx::err::AllocateFailure(handle);
        return;
    }
    if (decx::alloc::_device_malloc(&dev_tmp1, dst->_length * sizeof(de::CPf), true, S)) {
        decx::err::AllocateFailure(handle);
        return;
    }

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
                decx::signal::CUDA_FFT1D_R2_C2C_first_caller<true>((de::CPf*)src->Vec.ptr, MIF_1.mem, src->length, S);
                decx::utils::set_mutex_memory_state<de::CPf, de::CPf>(&MIF_1, &MIF_0);
            }
            else if (count == _conf->_base_num - 1) {
                decx::signal::dev_CUDA_FFT1D_R2_C2C_last_caller(&MIF_0, &MIF_1, (de::CPf*)dst->Vec.ptr, src->length, warp_proc_len, S);
            }
            else {
                decx::signal::CUDA_FFT1D_R2_C2C_caller(&MIF_0, &MIF_1, src->length, warp_proc_len, S);
            }
            break;

        case 3:
            if (count == 0) {
                decx::signal::CUDA_FFT1D_R3_C2C_first_caller<true>((de::CPf*)src->Vec.ptr, MIF_1.mem, src->length, S);
                decx::utils::set_mutex_memory_state<de::CPf, de::CPf>(&MIF_1, &MIF_0);
            }
            else if (count == _conf->_base_num - 1) {
                decx::signal::dev_CUDA_FFT1D_R3_C2C_last_caller(&MIF_0, &MIF_1, (de::CPf*)dst->Vec.ptr, src->length, warp_proc_len, S);
            }
            else {
                decx::signal::CUDA_FFT1D_R3_C2C_caller(&MIF_0, &MIF_1, src->length, warp_proc_len, S);
            }
            break;

        case 4:
            if (count == 0) {
                decx::signal::CUDA_FFT1D_R4_C2C_first_caller<true>((de::CPf*)src->Vec.ptr, MIF_1.mem, src->length, S);
                decx::utils::set_mutex_memory_state<de::CPf, de::CPf>(&MIF_1, &MIF_0);
            }
            else if (count == _conf->_base_num - 1) {
                decx::signal::dev_CUDA_FFT1D_R4_C2C_last_caller(&MIF_0, &MIF_1, (de::CPf*)dst->Vec.ptr, src->length, warp_proc_len, S);
            }
            else {
                decx::signal::CUDA_FFT1D_R4_C2C_caller(&MIF_0, &MIF_1, src->length, warp_proc_len, S);
            }
            break;

        case 5:
            if (count == 0) {
                decx::signal::CUDA_FFT1D_R5_C2C_first_caller<true>((de::CPf*)src->Vec.ptr, MIF_1.mem, src->length, S);
                decx::utils::set_mutex_memory_state<de::CPf, de::CPf>(&MIF_1, &MIF_0);
            }
            else if (count == _conf->_base_num - 1) {
                decx::signal::dev_CUDA_FFT1D_R5_C2C_last_caller(&MIF_0, &MIF_1, (de::CPf*)dst->Vec.ptr, src->length, warp_proc_len, S);
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

    E->event_record(S);
    E->synchronize();

    decx::alloc::_device_dealloc(&dev_tmp0);
    decx::alloc::_device_dealloc(&dev_tmp1);
}




static void decx::signal::dev_GPU_IFFT1D_C2R_fp32_organizer(decx::_GPU_Vector* src, decx::_GPU_Vector* dst,
    decx::signal::CUDA_FFT_Configs* _conf, de::DH* handle, decx::cuda_stream* S, decx::cuda_event* E)
{
    decx::PtrInfo<de::CPf> dev_tmp0, dev_tmp1;

    if (decx::alloc::_device_malloc(&dev_tmp0, dst->_length * sizeof(de::CPf), true, S)) {
        decx::err::AllocateFailure(handle);
        return;
    }
    if (decx::alloc::_device_malloc(&dev_tmp1, dst->_length * sizeof(de::CPf), true, S)) {
        decx::err::AllocateFailure(handle);
        return;
    }
    
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
            if (i == 0 && i == _conf->_base_num - 1) {
                decx::signal::CUDA_IFFT1D_R2_C2R_once_caller((de::CPf*)src->Vec.ptr, MIF_1.mem, src->length, S);
                decx::utils::set_mutex_memory_state<de::CPf, de::CPf>(&MIF_1, &MIF_0);
            }
            else if (i == 0 && i != _conf->_base_num - 1) {
                decx::signal::CUDA_FFT1D_R2_C2C_first_caller<true>((de::CPf*)src->Vec.ptr, MIF_1.mem, src->length, S);
                decx::utils::set_mutex_memory_state<de::CPf, de::CPf>(&MIF_1, &MIF_0);
            }
            else if (i != 0 && i == _conf->_base_num - 1) {
                decx::signal::dev_CUDA_IFFT1D_R2_C2R_last_caller(&MIF_0, &MIF_1, (de::CPf*)dst->Vec.ptr, src->length, warp_proc_len, S);
            }
            else {
                decx::signal::CUDA_FFT1D_R2_C2C_caller(&MIF_0, &MIF_1, src->length, warp_proc_len, S);
            }
            break;

        case 3:
            if (i == 0 && i == _conf->_base_num - 1) {
                decx::signal::CUDA_IFFT1D_R3_C2R_once_caller((de::CPf*)src->Vec.ptr, MIF_1.mem, src->length, S);
                decx::utils::set_mutex_memory_state<de::CPf, de::CPf>(&MIF_1, &MIF_0);
            }
            else if (i == 0 && i != _conf->_base_num - 1) {
                decx::signal::CUDA_FFT1D_R3_C2C_first_caller<true>((de::CPf*)src->Vec.ptr, MIF_1.mem, src->length, S);
                decx::utils::set_mutex_memory_state<de::CPf, de::CPf>(&MIF_1, &MIF_0);
            }
            else if (i != 0 && i == _conf->_base_num - 1) {
                decx::signal::dev_CUDA_IFFT1D_R3_C2R_last_caller(&MIF_0, &MIF_1, (de::CPf*)dst->Vec.ptr, src->length, warp_proc_len, S);
            }
            else {
                decx::signal::CUDA_FFT1D_R3_C2C_caller(&MIF_0, &MIF_1, src->length, warp_proc_len, S);
            }
            break;

        case 4:
            if (i == 0 && i == _conf->_base_num - 1) {
                decx::signal::CUDA_IFFT1D_R4_R2C_once_caller((de::CPf*)src->Vec.ptr, MIF_1.mem, src->length, S);
                decx::utils::set_mutex_memory_state<de::CPf, de::CPf>(&MIF_1, &MIF_0);
            }
            else if (i == 0 && i != _conf->_base_num - 1) {
                decx::signal::CUDA_FFT1D_R4_C2C_first_caller<true>((de::CPf*)src->Vec.ptr, MIF_1.mem, src->length, S);
                decx::utils::set_mutex_memory_state<de::CPf, de::CPf>(&MIF_1, &MIF_0);
            }
            else if (i != 0 && i == _conf->_base_num - 1) {
                decx::signal::dev_CUDA_IFFT1D_R4_C2R_last_caller(&MIF_0, &MIF_1, (de::CPf*)dst->Vec.ptr, src->length, warp_proc_len, S);
            }
            else {
                decx::signal::CUDA_FFT1D_R4_C2C_caller(&MIF_0, &MIF_1, src->length, warp_proc_len, S);
            }
            break;

        case 5:
            if (i == 0 && i == _conf->_base_num - 1) {
                decx::signal::CUDA_IFFT1D_R5_C2R_once_caller((de::CPf*)src->Vec.ptr, MIF_1.mem, src->length, S);
                decx::utils::set_mutex_memory_state<de::CPf, de::CPf>(&MIF_1, &MIF_0);
            }
            else if (i == 0 && i != _conf->_base_num - 1) {
                decx::signal::CUDA_FFT1D_R5_C2C_first_caller<false>((de::CPf*)src->Vec.ptr, MIF_1.mem, src->length, S);
                decx::utils::set_mutex_memory_state<de::CPf, de::CPf>(&MIF_1, &MIF_0);
            }
            else if (i != 0 && i == _conf->_base_num - 1) {
                decx::signal::dev_CUDA_IFFT1D_R5_C2R_last_caller(&MIF_0, &MIF_1, (de::CPf*)dst->Vec.ptr, src->length, warp_proc_len, S);
            }
            else {
                decx::signal::CUDA_FFT1D_R5_C2C_caller(&MIF_0, &MIF_1, src->length, warp_proc_len, S);
            }
            break;

        default:
            break;
        }
    }

    E->event_record(S);
    E->synchronize();

    decx::alloc::_device_dealloc(&dev_tmp0);
    decx::alloc::_device_dealloc(&dev_tmp1);
}


#endif