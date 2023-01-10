/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/


// 不存在height不能被vec4整除的现象，因为我会开一个小缓存区间，which height是4的倍数，在
// 处理之前先将src->Mat.ptr的内容拷贝上去，而dev_tmp0 and dev_tmp1 的height都是4的倍数


#ifndef _FFT3D_RADIX_5_KERNEL_H_
#define _FFT3D_RADIX_5_KERNEL_H_

#include "../../../../core/basic.h"
#include "../../../../core/thread_management/thread_pool.h"
#include "../../../../core/utils/fragment_arrangment.h"
#include "../../../../classes/classes_util.h"
#include "../../fft_utils.h"
#include "../../../CPU_cpf32_avx.h"
#include "../FFT_utils_kernel.h"
#include "../CPU_FFT_configs.h"


namespace decx
{
    namespace signal
    {
        namespace CPUK{
        /**
        * @param procH : in vec4 (real procH / 4)
        * @param procW : in element
        */
        _THREAD_CALL_ void
        _FFT2D_R5_fp32_R2C_first_ST_vec4col(const float* __restrict src, double* __restrict dst, 
            const uint signal_W, const uint Wsrc, const uint procW, const uint procH);


        /**
        * @param procH : in vec4 (real procH / 4)
        * @param procW : in element
        */
        _THREAD_CALL_ void
        _IFFT2D_R5_fp32_C2C_first_ST_vec4col(const double* __restrict src, double* __restrict dst, 
            const uint signal_W, const uint procW, const uint procH);


        /**
        * @param procH : in vec4 (real procH / 4)
        * @param procW : in element
        */
        _THREAD_CALL_ void
        _FFT2D_R5_fp32_C2C_first_ST_vec4col(const double* __restrict src, double* __restrict dst, 
            const uint signal_W, const uint procW, const uint procH);


        /**
        * @param procH : in vec4 (real procH / 4)
        * @param procW : in element
        */
        _THREAD_CALL_ void
        _IFFT2D_R5_fp32_C2C_first_ST_vec4col_L4(const double* __restrict src, double* __restrict dst, 
            const uint signal_W, const uint procW, const uint procH, const uint _Left);


        /**
        * @param procH : in vec4 (real procH / 4)
        * @param procW : in element
        */
        _THREAD_CALL_ void
        _FFT2D_R5_fp32_C2C_first_ST_vec4col_L4(const double* __restrict src, double* __restrict dst, 
            const uint signal_W, const uint procW, const uint procH, const uint _Left);


        /**
        * @param procH : in vec4 (real procH / 4)
        * @param procW : in element
        */
        _THREAD_CALL_ void
        _FFT2D_R5_fp32_R2C_first_ST_vec4col_L4(const float* __restrict src, double* __restrict dst, 
            const uint signal_W, const uint Wsrc, const uint procW, const uint procH, const uint _Left);


        /**
        * @param procH : in vec4 (real procH / 4)
        */
        _THREAD_CALL_ void
        _FFT2D_R5_fp32_C2C_ST_vec4col(const double* __restrict src, double* __restrict dst, 
            const uint warp_proc_len, const uint signal_W, const uint procW, const uint procH);



        /**
        * @param procH : in vec4 (real procH / 4)
        */
        _THREAD_CALL_ void
        _FFT2D_R5_fp32_C2C_ST_vec4col_L4(const double* __restrict src, double* __restrict dst, 
            const uint warp_proc_len, const uint signal_W, const uint procW, const uint procH, const uint _Left);
        }
    }
}




namespace decx
{
    namespace signal
    {
        namespace cpu{
        static _THREAD_CALL_
        void FFT2D_R5_R2C_fp32_1D_caller(decx::signal::_R2C_PM* _packed_params, const int count, const size_t warp_proc_len);


        template <bool is_IFFT>
        static _THREAD_CALL_
        void FFT2D_R5_C2C_fp32_1D_caller(decx::signal::_C2C_PM* _packed_params, const int count, const size_t warp_proc_len);

        
        template <bool is_IFFT>
        static _THREAD_CALL_
        void FFT2D_R5_C2C_fp32_1D_caller_mid(decx::signal::_C2C_PM* _packed_params, const int count, const size_t warp_proc_len);


        static _THREAD_CALL_ void 
        FFT2D_R5_R2C_fp32_1D_caller_L4(decx::signal::_R2C_PM* _packed_params, const int count, const size_t warp_proc_len);


        template <bool is_IFFT>
        static _THREAD_CALL_ void 
        FFT2D_R5_C2C_fp32_1D_caller_L4(decx::signal::_C2C_PM* _packed_params, const int count, const size_t warp_proc_len);
        }
    }
}






_THREAD_CALL_
void decx::signal::cpu::FFT2D_R5_R2C_fp32_1D_caller(decx::signal::_R2C_PM* _packed_params, const int count, const size_t warp_proc_len)
{
    if (count == 0) {
        decx::signal::CPUK::_FFT2D_R5_fp32_R2C_first_ST_vec4col(_packed_params->_initial, _packed_params->tmp0.mem,
            _packed_params->signal_W, _packed_params->Wsrc, _packed_params->proc_dim.x, _packed_params->proc_dim.y);
        decx::utils::set_mutex_memory_state<double, double>(&_packed_params->tmp0, &_packed_params->tmp1);
    }
    else {
        if (_packed_params->tmp0.leading) {
            decx::signal::CPUK::_FFT2D_R5_fp32_C2C_ST_vec4col(_packed_params->tmp0.mem, _packed_params->tmp1.mem,
                warp_proc_len, _packed_params->signal_W, _packed_params->proc_dim.x, _packed_params->proc_dim.y);
            decx::utils::set_mutex_memory_state<double, double>(&_packed_params->tmp1, &_packed_params->tmp0);
        }
        else {
            decx::signal::CPUK::_FFT2D_R5_fp32_C2C_ST_vec4col(_packed_params->tmp1.mem, _packed_params->tmp0.mem, warp_proc_len,
                _packed_params->signal_W, _packed_params->proc_dim.x, _packed_params->proc_dim.y);
            decx::utils::set_mutex_memory_state<double, double>(&_packed_params->tmp0, &_packed_params->tmp1);
        }
    }
}


template <bool is_IFFT>
_THREAD_CALL_
void decx::signal::cpu::FFT2D_R5_C2C_fp32_1D_caller(decx::signal::_C2C_PM* _packed_params, const int count, const size_t warp_proc_len)
{
    if (count == 0) {
        if (is_IFFT) {
            decx::signal::CPUK::_IFFT2D_R5_fp32_C2C_first_ST_vec4col(_packed_params->_initial, _packed_params->tmp0.mem,
                _packed_params->signal_W, _packed_params->proc_dim.x, _packed_params->proc_dim.y);
        }
        else {
            decx::signal::CPUK::_FFT2D_R5_fp32_C2C_first_ST_vec4col(_packed_params->_initial, _packed_params->tmp0.mem,
                _packed_params->signal_W, _packed_params->proc_dim.x, _packed_params->proc_dim.y);
        }
        decx::utils::set_mutex_memory_state<double, double>(&_packed_params->tmp0, &_packed_params->tmp1);
    }
    else {
        if (_packed_params->tmp0.leading) {
            decx::signal::CPUK::_FFT2D_R5_fp32_C2C_ST_vec4col(_packed_params->tmp0.mem, _packed_params->tmp1.mem,
                warp_proc_len, _packed_params->signal_W, _packed_params->proc_dim.x, _packed_params->proc_dim.y);
            decx::utils::set_mutex_memory_state<double, double>(&_packed_params->tmp1, &_packed_params->tmp0);
        }
        else {
            decx::signal::CPUK::_FFT2D_R5_fp32_C2C_ST_vec4col(_packed_params->tmp1.mem, _packed_params->tmp0.mem, warp_proc_len,
                _packed_params->signal_W, _packed_params->proc_dim.x, _packed_params->proc_dim.y);
            decx::utils::set_mutex_memory_state<double, double>(&_packed_params->tmp0, &_packed_params->tmp1);
        }
    }
}


template <bool is_IFFT>
_THREAD_CALL_
void decx::signal::cpu::FFT2D_R5_C2C_fp32_1D_caller_mid(decx::signal::_C2C_PM* _packed_params, const int count, const size_t warp_proc_len)
{
    if (count == 0) {
        double* read_ptr, * write_ptr;
        if (_packed_params->tmp0.leading) {
            read_ptr = _packed_params->tmp0.mem;
            write_ptr = _packed_params->tmp1.mem;

            decx::utils::set_mutex_memory_state<double, double>(&_packed_params->tmp1, &_packed_params->tmp0);
        }
        else {
            read_ptr = _packed_params->tmp1.mem;
            write_ptr = _packed_params->tmp0.mem;

            decx::utils::set_mutex_memory_state<double, double>(&_packed_params->tmp0, &_packed_params->tmp1);
        }
        if (is_IFFT) {
            decx::signal::CPUK::_IFFT2D_R5_fp32_C2C_first_ST_vec4col(read_ptr, write_ptr,
                _packed_params->signal_W, _packed_params->proc_dim.x, _packed_params->proc_dim.y);
        }
        else {
            decx::signal::CPUK::_FFT2D_R5_fp32_C2C_first_ST_vec4col(read_ptr, write_ptr,
                _packed_params->signal_W, _packed_params->proc_dim.x, _packed_params->proc_dim.y);
        }
    }
    else {
        double* read_ptr, * write_ptr;
        if (_packed_params->tmp0.leading) {
            read_ptr = _packed_params->tmp0.mem;
            write_ptr = _packed_params->tmp1.mem;

            decx::utils::set_mutex_memory_state<double, double>(&_packed_params->tmp1, &_packed_params->tmp0);
        }
        else {
            read_ptr = _packed_params->tmp1.mem;
            write_ptr = _packed_params->tmp0.mem;

            decx::utils::set_mutex_memory_state<double, double>(&_packed_params->tmp0, &_packed_params->tmp1);
        }
        decx::signal::CPUK::_FFT2D_R5_fp32_C2C_ST_vec4col(read_ptr, write_ptr,
            warp_proc_len, _packed_params->signal_W, _packed_params->proc_dim.x, _packed_params->proc_dim.y);
    }
}



_THREAD_CALL_
void decx::signal::cpu::FFT2D_R5_R2C_fp32_1D_caller_L4(decx::signal::_R2C_PM* _packed_params, const int count, const size_t warp_proc_len)
{
    const uint _int_procH = _packed_params->proc_dim.y / 4, _L4 = _packed_params->proc_dim.y % 4;
    if (count == 0) {
        decx::signal::CPUK::_FFT2D_R5_fp32_R2C_first_ST_vec4col_L4(_packed_params->_initial, _packed_params->tmp0.mem,
            _packed_params->signal_W, _packed_params->Wsrc, _packed_params->proc_dim.x, _int_procH, _L4);
        decx::utils::set_mutex_memory_state<double, double>(&_packed_params->tmp0, &_packed_params->tmp1);
    }
    else {
        if (_packed_params->tmp0.leading) {
            decx::signal::CPUK::_FFT2D_R5_fp32_C2C_ST_vec4col_L4(_packed_params->tmp0.mem, _packed_params->tmp1.mem,
                warp_proc_len, _packed_params->signal_W, _packed_params->proc_dim.x, _int_procH, _L4);
            decx::utils::set_mutex_memory_state<double, double>(&_packed_params->tmp1, &_packed_params->tmp0);
        }
        else {
            decx::signal::CPUK::_FFT2D_R5_fp32_C2C_ST_vec4col_L4(_packed_params->tmp1.mem, _packed_params->tmp0.mem, warp_proc_len,
                _packed_params->signal_W, _packed_params->proc_dim.x, _int_procH, _L4);
            decx::utils::set_mutex_memory_state<double, double>(&_packed_params->tmp0, &_packed_params->tmp1);
        }
    }
}


template <bool is_IFFT>
_THREAD_CALL_
void decx::signal::cpu::FFT2D_R5_C2C_fp32_1D_caller_L4(decx::signal::_C2C_PM* _packed_params, const int count, const size_t warp_proc_len)
{
    const uint _int_procH = _packed_params->proc_dim.y / 4, _L4 = _packed_params->proc_dim.y % 4;
    if (count == 0) {
        if (is_IFFT) {
            decx::signal::CPUK::_IFFT2D_R5_fp32_C2C_first_ST_vec4col_L4(_packed_params->_initial, _packed_params->tmp0.mem,
                _packed_params->signal_W, _packed_params->proc_dim.x, _int_procH, _L4);
        }
        else {
            decx::signal::CPUK::_FFT2D_R5_fp32_C2C_first_ST_vec4col_L4(_packed_params->_initial, _packed_params->tmp0.mem,
                _packed_params->signal_W, _packed_params->proc_dim.x, _int_procH, _L4);
        }
        decx::utils::set_mutex_memory_state<double, double>(&_packed_params->tmp0, &_packed_params->tmp1);
    }
    else {
        if (_packed_params->tmp0.leading) {
            decx::signal::CPUK::_FFT2D_R5_fp32_C2C_ST_vec4col_L4(_packed_params->tmp0.mem, _packed_params->tmp1.mem,
                warp_proc_len, _packed_params->signal_W, _packed_params->proc_dim.x, _int_procH, _L4);
            decx::utils::set_mutex_memory_state<double, double>(&_packed_params->tmp1, &_packed_params->tmp0);
        }
        else {
            decx::signal::CPUK::_FFT2D_R5_fp32_C2C_ST_vec4col_L4(_packed_params->tmp1.mem, _packed_params->tmp0.mem, warp_proc_len,
                _packed_params->signal_W, _packed_params->proc_dim.x, _int_procH, _L4);
            decx::utils::set_mutex_memory_state<double, double>(&_packed_params->tmp0, &_packed_params->tmp1);
        }
    }
}


#endif