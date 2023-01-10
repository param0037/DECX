/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/


#ifndef _FFT2D_SUB_FUNCTIONS_H_
#define _FFT2D_SUB_FUNCTIONS_H_


#include "../../fft_utils.h"
#include "../CPU_FFT_configs.h"
#include "../../../../classes/Matrix.h"
#include "../../../../classes/classes_util.h"
#include "FFT2D_Radix_2_kernel.h"
#include "FFT2D_Radix_3_kernel.h"
#include "FFT2D_Radix_4_kernel.h"
#include "FFT2D_Radix_5_kernel.h"


namespace decx
{
    namespace signal
    {
        /*
        * @param signal_W : effective signal width, in element
        * @param proc_dim : ~.x -> pitch of proc_area (in element); ~.y -> height of proc_area (in vec4)
        */
        _THREAD_FUNCTION_ void
        FFT2D_R2C_fp32_1D_caller(decx::signal::_R2C_PM* _packed_params);


        template <bool is_IFFT>
        /*
        * @param signal_W : effective signal width, in element
        * @param proc_dim : ~.x -> pitch of proc_area (in element); ~.y -> height of proc_area (in vec4)
        */
        _THREAD_FUNCTION_ void
        FFT2D_C2C_fp32_1D_caller(decx::signal::_C2C_PM* _packed_params);




        template <bool is_IFFT>
        /*
        * @param signal_W : effective signal width, in element
        * @param proc_dim : ~.x -> pitch of proc_area (in element); ~.y -> height of proc_area (in vec4)
        */
        _THREAD_FUNCTION_ void
        FFT2D_C2C_fp32_1D_caller_mid(decx::signal::_C2C_PM* _packed_params);


        /*
        * @param signal_W : effective signal width, in element
        * @param proc_dim : ~.x -> pitch of proc_area (in element); ~.y -> height of proc_area (!!! in ELEMENT !!!)
        */
        _THREAD_FUNCTION_ void
        FFT2D_R2C_fp32_1D_caller_L4(decx::signal::_R2C_PM* _packed_params);


        template <bool is_IFFT>
        /**
        * @brief : can be used in 
        * 1. IFFT<C2C>
        * 2. FFT<C2C> when it is in <_L4> case
        * @param signal_W : effective signal width, in element
        * @param proc_dim : ~.x -> pitch of proc_area (in element); ~.y -> height of proc_area (!!! in ELEMENT !!!)
        */
        _THREAD_FUNCTION_ void
        FFT2D_C2C_fp32_1D_caller_L4(decx::signal::_C2C_PM* _packed_params);
    }
}


_THREAD_FUNCTION_ void
decx::signal::FFT2D_R2C_fp32_1D_caller(decx::signal::_R2C_PM* _packed_params)
{
    int current_base, count = 0;
    size_t warp_proc_len = 1;

    for (int i = 0; i < _packed_params->_conf->_base_num; ++i) {
        current_base = _packed_params->_conf->_base[i];
        warp_proc_len *= current_base;
        switch (current_base)
        {
        case 2:
            decx::signal::cpu::FFT2D_R2_R2C_fp32_1D_caller(_packed_params, count, warp_proc_len);
            break;

        case 3:
            decx::signal::cpu::FFT2D_R3_R2C_fp32_1D_caller(_packed_params, count, warp_proc_len);
            break;

        case 4:
            decx::signal::cpu::FFT2D_R4_R2C_fp32_1D_caller(_packed_params, count, warp_proc_len);
            break;

        case 5:
            decx::signal::cpu::FFT2D_R5_R2C_fp32_1D_caller(_packed_params, count, warp_proc_len);
            break;

        default:
            break;
        }
        ++count;
    }
}


template <bool is_IFFT>
_THREAD_FUNCTION_ void
decx::signal::FFT2D_C2C_fp32_1D_caller(decx::signal::_C2C_PM* _packed_params)
{
    int current_base, count = 0;
    size_t warp_proc_len = 1;
    
    for (int i = 0; i < _packed_params->_conf->_base_num; ++i) {
        current_base = _packed_params->_conf->_base[i];
        warp_proc_len *= current_base;
        switch (current_base)
        {
        case 2:
            decx::signal::cpu::FFT2D_R2_C2C_fp32_1D_caller<is_IFFT>(_packed_params, count, warp_proc_len);
            break;

        case 3:
            decx::signal::cpu::FFT2D_R3_C2C_fp32_1D_caller<is_IFFT>(_packed_params, count, warp_proc_len);
            break;

        case 4:
            decx::signal::cpu::FFT2D_R4_C2C_fp32_1D_caller<is_IFFT>(_packed_params, count, warp_proc_len);
            break;

        case 5:
            decx::signal::cpu::FFT2D_R5_C2C_fp32_1D_caller<is_IFFT>(_packed_params, count, warp_proc_len);
            break;

        default:
            break;
        }
        ++count;
    }
}



template <bool is_IFFT>
_THREAD_FUNCTION_ void
decx::signal::FFT2D_C2C_fp32_1D_caller_mid(decx::signal::_C2C_PM* _packed_params)
{
    int current_base, count = 0;
    size_t warp_proc_len = 1;

    for (int i = 0; i < _packed_params->_conf->_base_num; ++i) {
        current_base = _packed_params->_conf->_base[i];
        warp_proc_len *= current_base;
        switch (current_base)
        {
        case 2:
            decx::signal::cpu::FFT2D_R2_C2C_fp32_1D_caller_mid<is_IFFT>(_packed_params, count, warp_proc_len);
            break;

        case 3:
            decx::signal::cpu::FFT2D_R3_C2C_fp32_1D_caller_mid<is_IFFT>(_packed_params, count, warp_proc_len);
            break;

        case 4:
            decx::signal::cpu::FFT2D_R4_C2C_fp32_1D_caller_mid<is_IFFT>(_packed_params, count, warp_proc_len);
            break;

        case 5:
            decx::signal::cpu::FFT2D_R5_C2C_fp32_1D_caller_mid<is_IFFT>(_packed_params, count, warp_proc_len);
            break;

        default:
            break;
        }
        ++count;
    }
}




_THREAD_FUNCTION_ void
decx::signal::FFT2D_R2C_fp32_1D_caller_L4(decx::signal::_R2C_PM* _packed_params)
{
    int current_base, count = 0;
    size_t warp_proc_len = 1;

    for (int i = 0; i < _packed_params->_conf->_base_num; ++i) {
        current_base = _packed_params->_conf->_base[i];
        warp_proc_len *= current_base;
        switch (current_base)
        {
        case 2:
            decx::signal::cpu::FFT2D_R2_R2C_fp32_1D_caller_L4(_packed_params, count, warp_proc_len);
            break;

        case 3:
            decx::signal::cpu::FFT2D_R3_R2C_fp32_1D_caller_L4(_packed_params, count, warp_proc_len);
            break;

        case 4:
            decx::signal::cpu::FFT2D_R4_R2C_fp32_1D_caller_L4(_packed_params, count, warp_proc_len);
            break;

        case 5:
            decx::signal::cpu::FFT2D_R5_R2C_fp32_1D_caller_L4(_packed_params, count, warp_proc_len);
            break;
        default:
            break;
        }
        ++count;
    }
}


template <bool is_IFFT>
_THREAD_FUNCTION_ void
decx::signal::FFT2D_C2C_fp32_1D_caller_L4(decx::signal::_C2C_PM* _packed_params)
{
    int current_base, count = 0;
    size_t warp_proc_len = 1;
    for (int i = 0; i < _packed_params->_conf->_base_num; ++i) {
        current_base = _packed_params->_conf->_base[i];
        warp_proc_len *= current_base;
        switch (current_base)
        {
        case 2:
            decx::signal::cpu::FFT2D_R2_C2C_fp32_1D_caller_L4<is_IFFT>(_packed_params, count, warp_proc_len);
            break;

        case 3:
            decx::signal::cpu::FFT2D_R3_C2C_fp32_1D_caller_L4<is_IFFT>(_packed_params, count, warp_proc_len);
            break;

        case 4:
            decx::signal::cpu::FFT2D_R4_C2C_fp32_1D_caller_L4<is_IFFT>(_packed_params, count, warp_proc_len);
            break;

        case 5:
            decx::signal::cpu::FFT2D_R5_C2C_fp32_1D_caller_L4<is_IFFT>(_packed_params, count, warp_proc_len);
            break;
        default:
            break;
        }
        ++count;
    }
}



namespace decx
{
    namespace signal
    {
        static void FFT2D_R2C_fp32_organizer(decx::_Matrix* src, decx::_Matrix* dst, 
            decx::signal::CPU_FFT_Configs _conf[2], de::DH* handle);


        static void FFT2D_C2C_fp32_organizer(decx::_Matrix* src, decx::_Matrix* dst,
            decx::signal::CPU_FFT_Configs _conf[2], de::DH* handle);
    }
}




static void decx::signal::FFT2D_C2C_fp32_organizer(decx::_Matrix* src, 
                                                   decx::_Matrix* dst, 
                                                   decx::signal::CPU_FFT_Configs _conf[2], 
                                                   de::DH* handle)
{
    decx::utils::_thread_arrange_1D exec_t1D(decx::cpI.cpu_concurrency);

    decx::PtrInfo<double> cache_for_last;

    decx::PtrInfo<double> buffer0, buffer1;
    const uint2 tmp_dim = make_uint2(dst->pitch, decx::utils::ceil<uint>(src->height, 4) * 4);

    if (decx::alloc::_host_virtual_page_malloc(&buffer0, tmp_dim.x * tmp_dim.y * sizeof(de::CPf))) {
        decx::err::AllocateFailure(handle);
        Print_Error_Message(4, ALLOC_FAIL);
        return;
    }
    if (decx::alloc::_host_virtual_page_malloc(&buffer1, tmp_dim.x * tmp_dim.y * sizeof(de::CPf))) {
        decx::err::AllocateFailure(handle);
        Print_Error_Message(4, ALLOC_FAIL);
        return;
    }

    double* t_ptr_init = reinterpret_cast<double*>(src->Mat.ptr);
    double* t_ptr_tmp0 = buffer0.ptr;
    double* t_ptr_tmp1 = buffer1.ptr;

    decx::signal::_C2C_PM* packed_param_array = new decx::signal::_C2C_PM[exec_t1D.total_thread];

    decx::utils::frag_manager _f_mgrW, _f_mgrH;
    bool critH = decx::utils::frag_manager_gen_Nx(&_f_mgrH, src->height, exec_t1D.total_thread, 4);
    bool critW = decx::utils::frag_manager_gen_Nx(&_f_mgrW, tmp_dim.x, exec_t1D.total_thread, 4);

    size_t frag_size = _f_mgrH.frag_len * tmp_dim.x;

    for (int i = 0; i < exec_t1D.total_thread - 1; ++i) 
    {
        new (packed_param_array + i) _NEW_FFT_PM_C2C_(t_ptr_init,
                         decx::alloc::MIF<double>(t_ptr_tmp0), 
                         decx::alloc::MIF<double>(t_ptr_tmp1),
                         &_conf[0], src->width,
                         make_uint2(tmp_dim.x, _f_mgrH.frag_len / 4));

        exec_t1D._async_thread[i] = decx::cpu::register_task(&decx::thread_pool, decx::signal::FFT2D_C2C_fp32_1D_caller<false>, packed_param_array + i);
        t_ptr_init += frag_size;
        t_ptr_tmp0 += frag_size;
        t_ptr_tmp1 += frag_size;
    }

    new (packed_param_array + exec_t1D.total_thread - 1) _NEW_FFT_PM_C2C_(t_ptr_init,
                                        decx::alloc::MIF<double>(t_ptr_tmp0), 
                                        decx::alloc::MIF<double>(t_ptr_tmp1),
                                        &_conf[0], src->width,
                                        make_uint2(tmp_dim.x, _f_mgrH.frag_left_over));

    if (src->height % 4)
    {
        exec_t1D._async_thread[exec_t1D.total_thread - 1] = decx::cpu::register_task(&decx::thread_pool, decx::signal::FFT2D_C2C_fp32_1D_caller_L4<false>,
            packed_param_array + exec_t1D.total_thread - 1);
    }
    else {
        if (critH) {
            packed_param_array[exec_t1D.total_thread - 1].proc_dim.y = _f_mgrH.frag_len / 4;
        }
        else {
            packed_param_array[exec_t1D.total_thread - 1].proc_dim.y /= 4;
        }
        exec_t1D._async_thread[exec_t1D.total_thread - 1] = decx::cpu::register_task(&decx::thread_pool, decx::signal::FFT2D_C2C_fp32_1D_caller<false>,
            packed_param_array + exec_t1D.total_thread - 1);
    }

    exec_t1D.__sync_all_threads();

    t_ptr_tmp0 = buffer0.ptr;
    t_ptr_tmp1 = buffer1.ptr;

    bool tmp0_leading = false, tmp1_leading = false;


    if (packed_param_array[0].tmp0.leading) {
        decx::signal::CPUK::_FFT2D_transpose_C(buffer0.ptr, buffer1.ptr, tmp_dim.x, tmp_dim.y, make_uint2(tmp_dim.y / 4, tmp_dim.x / 4), 0);
        tmp1_leading = true;
        tmp0_leading = false;
    }
    else {
        decx::signal::CPUK::_FFT2D_transpose_C(buffer1.ptr, buffer0.ptr, tmp_dim.x, tmp_dim.y, make_uint2(tmp_dim.y / 4, tmp_dim.x / 4), 0);
        tmp0_leading = true;
        tmp1_leading = false;
    }

    frag_size = _f_mgrW.frag_len * tmp_dim.y;

    for (int i = 0; i < exec_t1D.total_thread - 1; ++i) 
    {
        new (packed_param_array + i)
            _NEW_FFT_PM_C2C_mid(decx::alloc::MIF<double>(t_ptr_tmp0, tmp0_leading),
                                decx::alloc::MIF<double>(t_ptr_tmp1, tmp1_leading),
                                &_conf[1], src->height,
                                make_uint2(tmp_dim.y, _f_mgrW.frag_len / 4));

        exec_t1D._async_thread[i] = decx::cpu::register_task(&decx::thread_pool, decx::signal::FFT2D_C2C_fp32_1D_caller_mid<false>, packed_param_array + i);
        t_ptr_tmp0 += frag_size;
        t_ptr_tmp1 += frag_size;
    }

    uint _LH_W = critW ? _f_mgrW.frag_len : _f_mgrW.frag_left_over;
    new (packed_param_array + exec_t1D.total_thread - 1)
        _NEW_FFT_PM_C2C_mid(decx::alloc::MIF<double>(t_ptr_tmp0, tmp0_leading), 
                            decx::alloc::MIF<double>(t_ptr_tmp1, tmp1_leading),
                            &_conf[1], src->height,
                            make_uint2(tmp_dim.y, _LH_W / 4));

    exec_t1D._async_thread[exec_t1D.total_thread - 1] = decx::cpu::register_task(&decx::thread_pool, decx::signal::FFT2D_C2C_fp32_1D_caller_mid<false>,
        packed_param_array + exec_t1D.total_thread - 1);

    exec_t1D.__sync_all_threads();

    if (packed_param_array[0].tmp0.leading) {
        decx::signal::CPUK::_FFT2D_transpose_C(buffer0.ptr, (double*)dst->Mat.ptr, tmp_dim.y, tmp_dim.x,
            make_uint2(tmp_dim.x / 4, dst->height / 4), dst->height % 4);
    }
    else {
        decx::signal::CPUK::_FFT2D_transpose_C(buffer1.ptr, (double*)dst->Mat.ptr, tmp_dim.y, tmp_dim.x,
            make_uint2(tmp_dim.x / 4, dst->height / 4), dst->height % 4);
    }

    delete[] packed_param_array;
    decx::alloc::_host_virtual_page_dealloc(&buffer0);
    decx::alloc::_host_virtual_page_dealloc(&buffer1);
}





static void decx::signal::FFT2D_R2C_fp32_organizer(decx::_Matrix* src, 
                                                   decx::_Matrix* dst, 
                                                   decx::signal::CPU_FFT_Configs _conf[2], 
                                                   de::DH* handle)
{
    decx::utils::_thread_arrange_1D exec_t1D(decx::cpI.cpu_concurrency);

    decx::PtrInfo<double> cache_for_last;

    decx::PtrInfo<double> buffer0, buffer1;
    const uint2 tmp_dim = make_uint2(dst->pitch, decx::utils::ceil<uint>(src->height, 4) * 4);

    if (decx::alloc::_host_virtual_page_malloc(&buffer0, tmp_dim.x * tmp_dim.y * sizeof(de::CPf))) {
        decx::err::AllocateFailure(handle);
        Print_Error_Message(4, ALLOC_FAIL);
        return;
    }
    if (decx::alloc::_host_virtual_page_malloc(&buffer1, tmp_dim.x * tmp_dim.y * sizeof(de::CPf))) {
        decx::err::AllocateFailure(handle);
        Print_Error_Message(4, ALLOC_FAIL);
        return;
    }

    float* t_ptr_init = reinterpret_cast<float*>(src->Mat.ptr);
    double* t_ptr_tmp0 = buffer0.ptr;
    double* t_ptr_tmp1 = buffer1.ptr;

    decx::signal::_R2C_PM* packed_param_array = new decx::signal::_R2C_PM[exec_t1D.total_thread];

    decx::utils::frag_manager _f_mgrW, _f_mgrH;
    bool critH = decx::utils::frag_manager_gen_Nx(&_f_mgrH, src->height, exec_t1D.total_thread, 4);
    bool critW = decx::utils::frag_manager_gen_Nx(&_f_mgrW, tmp_dim.x, exec_t1D.total_thread, 4);
    
    size_t frag_size_src = _f_mgrH.frag_len * src->pitch;
    size_t frag_size = _f_mgrH.frag_len * tmp_dim.x;

    for (int i = 0; i < exec_t1D.total_thread - 1; ++i) 
    {
        new (packed_param_array + i) 
            _NEW_FFT_PM_R2C_(t_ptr_init,
                             decx::alloc::MIF<double>(t_ptr_tmp0), 
                             decx::alloc::MIF<double>(t_ptr_tmp1),
                             &_conf[0], src->width, src->pitch, 
                             make_uint2(tmp_dim.x, _f_mgrH.frag_len / 4));

        exec_t1D._async_thread[i] = decx::cpu::register_task(&decx::thread_pool, decx::signal::FFT2D_R2C_fp32_1D_caller, packed_param_array + i);
        t_ptr_init += frag_size_src;
        t_ptr_tmp0 += frag_size;
        t_ptr_tmp1 += frag_size;
    }

    new (packed_param_array + exec_t1D.total_thread - 1)
        _NEW_FFT_PM_R2C_(t_ptr_init,
                         decx::alloc::MIF<double>(t_ptr_tmp0), 
                         decx::alloc::MIF<double>(t_ptr_tmp1),
                         &_conf[0], src->width, src->pitch, 
                         make_uint2(tmp_dim.x, _f_mgrH.frag_left_over));
    
    if (src->height % 4)
    {
        exec_t1D._async_thread[exec_t1D.total_thread - 1] = decx::cpu::register_task(&decx::thread_pool, decx::signal::FFT2D_R2C_fp32_1D_caller_L4,
            packed_param_array + exec_t1D.total_thread - 1);
    }
    else {
        if (critH) {
            packed_param_array[exec_t1D.total_thread - 1].proc_dim.y = _f_mgrH.frag_len / 4;
        }
        else {
            packed_param_array[exec_t1D.total_thread - 1].proc_dim.y /= 4;
        }
        exec_t1D._async_thread[exec_t1D.total_thread - 1] = decx::cpu::register_task(&decx::thread_pool, decx::signal::FFT2D_R2C_fp32_1D_caller,
            packed_param_array + exec_t1D.total_thread - 1);
    }

    exec_t1D.__sync_all_threads();

    t_ptr_tmp0 = buffer0.ptr;
    t_ptr_tmp1 = buffer1.ptr;

    bool tmp0_leading = false, tmp1_leading = false;

    if (packed_param_array[0].tmp0.leading) {
        decx::signal::CPUK::_FFT2D_transpose_C(buffer0.ptr, buffer1.ptr, tmp_dim.x, tmp_dim.y, make_uint2(tmp_dim.y / 4, tmp_dim.x / 4), 0);
        tmp1_leading = true;
        tmp0_leading = false;
    }
    else {
        decx::signal::CPUK::_FFT2D_transpose_C(buffer1.ptr, buffer0.ptr, tmp_dim.x, tmp_dim.y, make_uint2(tmp_dim.y / 4, tmp_dim.x / 4), 0);
        tmp0_leading = true;
        tmp1_leading = false;
    }

    frag_size = _f_mgrW.frag_len * tmp_dim.y;
    decx::signal::_C2C_PM* PM_C2C_arr = (decx::signal::_C2C_PM*)packed_param_array;

    for (int i = 0; i < exec_t1D.total_thread - 1; ++i) 
    {
        new (PM_C2C_arr + i)
            _NEW_FFT_PM_C2C_mid(decx::alloc::MIF<double>(t_ptr_tmp0, tmp0_leading), 
                                decx::alloc::MIF<double>(t_ptr_tmp1, tmp1_leading),
                                &_conf[1], src->height, 
                                make_uint2(tmp_dim.y, _f_mgrW.frag_len / 4));

        exec_t1D._async_thread[i] = decx::cpu::register_task(&decx::thread_pool, decx::signal::FFT2D_C2C_fp32_1D_caller_mid<false>, PM_C2C_arr + i);
        t_ptr_tmp0 += frag_size;
        t_ptr_tmp1 += frag_size;
    }

    uint _LH_W = critW ? _f_mgrW.frag_len : _f_mgrW.frag_left_over;
    new (PM_C2C_arr + exec_t1D.total_thread - 1)
        _NEW_FFT_PM_C2C_mid(decx::alloc::MIF<double>(t_ptr_tmp0, tmp0_leading), 
                            decx::alloc::MIF<double>(t_ptr_tmp1, tmp1_leading),
                            &_conf[1], src->height, 
                            make_uint2(tmp_dim.y, _LH_W / 4));

    exec_t1D._async_thread[exec_t1D.total_thread - 1] = decx::cpu::register_task(&decx::thread_pool, decx::signal::FFT2D_C2C_fp32_1D_caller_mid<false>,
        PM_C2C_arr + exec_t1D.total_thread - 1);

    exec_t1D.__sync_all_threads();

    if (PM_C2C_arr[0].tmp0.leading) {
        decx::signal::CPUK::_FFT2D_transpose_C(buffer0.ptr, (double*)dst->Mat.ptr, tmp_dim.y, tmp_dim.x,
            make_uint2(tmp_dim.x / 4, dst->height / 4), dst->height % 4);
    }
    else {
        decx::signal::CPUK::_FFT2D_transpose_C(buffer1.ptr, (double*)dst->Mat.ptr, tmp_dim.y, tmp_dim.x,
            make_uint2(tmp_dim.x / 4, dst->height / 4), dst->height % 4);
    }

    delete[] packed_param_array;
    decx::alloc::_host_virtual_page_dealloc(&buffer0);
    decx::alloc::_host_virtual_page_dealloc(&buffer1);
}


#endif