/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _CPU_IFFT1D_SUB_FUNCTIONS_H_
#define _CPU_IFFT1D_SUB_FUNCTIONS_H_


#include "../../fft_utils.h"
#include "../CPU_FFT_configs.h"
#include "../../../../classes/Vector.h"
#include "../../../../classes/classes_util.h"
#include "FFT1D_radix_2_kernel.h"
#include "FFT1D_radix_3_kernel.h"
#include "FFT1D_radix_4_kernel.h"
#include "FFT1D_radix_5_kernel.h"
#include "../CPU_FFT_task_allocators.h"
#include "../FFT_utils_kernel.h"


namespace decx
{
    namespace signal
    {
        static void IFFT1D_C2C_fp32_organizer(decx::_Vector* src, decx::_Vector* dst, decx::signal::CPU_FFT_Configs* _conf, de::DH* handle);


        static void IFFT1D_C2R_fp32_organizer(decx::_Vector* src, decx::_Vector* dst, decx::signal::CPU_FFT_Configs* _conf, de::DH* handle);
    }
}



static void 
decx::signal::IFFT1D_C2C_fp32_organizer(decx::_Vector* src, decx::_Vector* dst, decx::signal::CPU_FFT_Configs* _conf, de::DH* handle)
{
    const size_t act_length = src->_length;

    decx::PtrInfo<double> dev_tmp0, dev_tmp1;
    if (decx::alloc::_host_virtual_page_malloc(&dev_tmp0, act_length * sizeof(de::CPf), true)) {
        Print_Error_Message(4, ALLOC_FAIL);
        decx::err::AllocateFailure(handle);
        return;
    }
    if (decx::alloc::_host_virtual_page_malloc(&dev_tmp1, act_length * sizeof(de::CPf), true)) {
        Print_Error_Message(4, ALLOC_FAIL);
        decx::err::AllocateFailure(handle);
        return;
    }

    decx::alloc::MIF<double> MIF_0, MIF_1;
    MIF_0.mem = dev_tmp0.ptr;
    MIF_1.mem = dev_tmp1.ptr;

    int current_base, count = 0;
    size_t warp_proc_len = 1;

    uint2 start_end;

    decx::utils::_thread_arrange_1D t1D(decx::cpu::_get_permitted_concurrency());
    //decx::utils::_thread_arrange_1D t1D(1);
    decx::utils::frag_manager f_mgr;
    
    for (int i = 0; i < _conf->_base_num; ++i) {
        current_base = _conf->_base[i];
        warp_proc_len *= current_base;
        switch (current_base)
        {
        case 2:
            if (count == 0) {
                decx::signal::cpu::IFFT_R2_C2C_assign_task_1D_first(&t1D, &f_mgr, (double*)src->Vec.ptr, dev_tmp0.ptr, src->length);
                decx::utils::set_mutex_memory_state<double, double>(&MIF_0, &MIF_1);
            }
            else {
                decx::signal::cpu::FFT_R2_C2C_assign_task_1D(&t1D, &f_mgr, &MIF_0, &MIF_1, src->length, warp_proc_len);
            }
            break;

        case 3:
            if (count == 0) {
                decx::signal::IFFT_R3_C2C_assign_task_1D_first(&t1D, &f_mgr, (double*)src->Vec.ptr, dev_tmp0.ptr, src->length);
                decx::utils::set_mutex_memory_state<double, double>(&MIF_0, &MIF_1);
            }
            else {
                decx::signal::FFT_R3_C2C_assign_task_1D(&t1D, &f_mgr, &MIF_0, &MIF_1, src->length, warp_proc_len);
            }
            break;

        case 4:
            if (count == 0) {
                decx::signal::cpu::IFFT_R4_C2C_assign_task_1D_first(&t1D, &f_mgr, (double*)src->Vec.ptr, dev_tmp0.ptr, src->length);
                decx::utils::set_mutex_memory_state<double, double>(&MIF_0, &MIF_1);
            }
            else {
                decx::signal::cpu::FFT_R4_C2C_assign_task_1D(&t1D, &f_mgr, &MIF_0, &MIF_1, src->length, warp_proc_len);
            }
            break;

        case 5:
            if (count == 0) {
                decx::utils::set_mutex_memory_state<double, double>(&MIF_0, &MIF_1);
            }
            else {
                decx::signal::cpu::FFT_R5_C2C_assign_task_1D(&t1D, &f_mgr, &MIF_0, &MIF_1, src->length, warp_proc_len);
            }
            break;
        default:
            break;
        }
        ++count;
    }

    if (MIF_0.leading) {
        memcpy(dst->Vec.ptr, MIF_0.mem, dst->length * sizeof(de::CPf));
    }
    else {
        memcpy(dst->Vec.ptr, MIF_1.mem, dst->length * sizeof(de::CPf));
    }

    decx::alloc::_host_virtual_page_dealloc(&dev_tmp0);
    decx::alloc::_host_virtual_page_dealloc(&dev_tmp1);
}




static void
decx::signal::IFFT1D_C2R_fp32_organizer(decx::_Vector* src, decx::_Vector* dst, decx::signal::CPU_FFT_Configs* _conf, de::DH* handle)
{
    const size_t act_length = src->_length;

    decx::PtrInfo<double> dev_tmp0, dev_tmp1;
    if (decx::alloc::_host_virtual_page_malloc(&dev_tmp0, act_length * sizeof(de::CPf), true)) {
        Print_Error_Message(4, ALLOC_FAIL);
        decx::err::AllocateFailure(handle);
        return;
    }
    if (decx::alloc::_host_virtual_page_malloc(&dev_tmp1, act_length * sizeof(de::CPf), true)) {
        Print_Error_Message(4, ALLOC_FAIL);
        decx::err::AllocateFailure(handle);
        return;
    }

    decx::alloc::MIF<double> MIF_0, MIF_1;
    MIF_0.mem = dev_tmp0.ptr;
    MIF_1.mem = dev_tmp1.ptr;

    int current_base, count = 0;
    size_t warp_proc_len = 1;

    uint2 start_end;

    decx::utils::_thread_arrange_1D t1D(decx::cpu::_get_permitted_concurrency());

    decx::utils::frag_manager f_mgr;

    for (int i = 0; i < _conf->_base_num; ++i) {
        current_base = _conf->_base[i];
        warp_proc_len *= current_base;
        switch (current_base)
        {
        case 2:
            if (count == 0) {
                decx::signal::cpu::IFFT_R2_C2C_assign_task_1D_first(&t1D, &f_mgr, (double*)src->Vec.ptr, dev_tmp0.ptr, src->length);
                decx::utils::set_mutex_memory_state<double, double>(&MIF_0, &MIF_1);
            }
            else {
                decx::signal::cpu::FFT_R2_C2C_assign_task_1D(&t1D, &f_mgr, &MIF_0, &MIF_1, src->length, warp_proc_len);
            }
            break;

        case 3:
            if (count == 0) {
                decx::signal::IFFT_R3_C2C_assign_task_1D_first(&t1D, &f_mgr, (double*)src->Vec.ptr, dev_tmp0.ptr, src->length);
                decx::utils::set_mutex_memory_state<double, double>(&MIF_0, &MIF_1);
            }
            else {
                decx::signal::FFT_R3_C2C_assign_task_1D(&t1D, &f_mgr, &MIF_0, &MIF_1, src->length, warp_proc_len);
            }
            break;

        case 4:
            if (count == 0) {
                decx::signal::cpu::IFFT_R4_C2C_assign_task_1D_first(&t1D, &f_mgr, (double*)src->Vec.ptr, dev_tmp0.ptr, src->length);
                decx::utils::set_mutex_memory_state<double, double>(&MIF_0, &MIF_1);
            }
            else {
                decx::signal::cpu::FFT_R4_C2C_assign_task_1D(&t1D, &f_mgr, &MIF_0, &MIF_1, src->length, warp_proc_len);
            }
            break;

        case 5:
            if (count == 0) {
                decx::signal::cpu::IFFT_R5_C2C_assign_task_1D_first(&t1D, &f_mgr, (double*)src->Vec.ptr, dev_tmp0.ptr, src->length);
                decx::utils::set_mutex_memory_state<double, double>(&MIF_0, &MIF_1);
            }
            else {
                decx::signal::cpu::FFT_R5_C2C_assign_task_1D(&t1D, &f_mgr, &MIF_0, &MIF_1, src->length, warp_proc_len);
            }
            break;
        default:
            break;
        }
        ++count;
    }

    if (MIF_0.leading) {
        decx::signal::CPUK::_FFT1D_cpy_cvtcp_f32(MIF_0.mem, (float*)dst->Vec.ptr, dst->_length / 4);
    }
    else {
        decx::signal::CPUK::_FFT1D_cpy_cvtcp_f32(MIF_1.mem, (float*)dst->Vec.ptr, dst->_length / 4);
    }

    decx::alloc::_host_virtual_page_dealloc(&dev_tmp0);
    decx::alloc::_host_virtual_page_dealloc(&dev_tmp1);
}

#endif