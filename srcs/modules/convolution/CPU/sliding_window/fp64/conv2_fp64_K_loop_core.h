/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/

#ifndef _CONV2_FP64_K_LOOP_CORE_H_
#define _CONV2_FP64_K_LOOP_CORE_H_

#include "../../../../core/thread_management/thread_pool.h"
#include "../../../../core/thread_management/thread_arrange.h"
#include "../../../../regional/regional_comparision/CPU/rcp_sliding_window_avx_ops.h"


namespace decx
{
    namespace conv {
        namespace CPUK {
            // *** ATTENTION *** ! -> In this model, kernel should be stored linearly (pitch = width)
            /*
            * In this model, we only pay attention to the width of kernel, regardless its height
            * Since only the kernel width affects the behaviours during loading data from src matrix
            */

            /**
            * radius = 3
            * @param Wsrc : width of src matrix, in float
            * @param Wdst : width of dst matrix, in float
            */
            static _THREAD_CALL_
                __m256d _conv2_fp64_loop_in_kernel_2regs_rw2(const double* src, const double* kernel, const uint2 ker_dims, const size_t Wsrc);


            /**
            * radius = 4
            * @param Wsrc : width of src matrix, in float
            * @param Wdst : width of dst matrix, in float
            */
            static _THREAD_CALL_
                __m256d _conv2_fp64_loop_in_kernel_2regs_rw4(const double* src, const double* kernel, const uint2 ker_dims, const size_t Wsrc);


            /**
            * radius > 6, reg_WL = 2
            * @param Wsrc : width of src matrix, in float
            * @param Wdst : width of dst matrix, in float
            */
            static _THREAD_CALL_
                __m256d _conv2_fp64_loop_in_kernel_Nregs_rw2(const double* src, const double* kernel, const uint2 ker_dims,
                    const size_t Wsrc, const uint _loop);


            /**
            * radius > 6, reg_WL = 2
            * @param Wsrc : width of src matrix, in float
            * @param Wdst : width of dst matrix, in float
            */
            static _THREAD_CALL_
                __m256d _conv2_fp64_loop_in_kernel_Nregs_rw4(const double* src, const double* kernel, const uint2 ker_dims,
                    const size_t Wsrc, const uint _loop);
        }
    }
}



#define _CONV2_REGS_FP64_SHIFT_LOAD_3 {       \
    k_value = kernel[ker_dex];              \
    _SLIDING_WINDOW_FP64_SHIFT_LOAD_3_(_proc_reg, _store_reg);      \
    _accumulator = _mm256_fmadd_pd(_proc_reg, _mm256_set1_pd(k_value), _accumulator);                               \
    ++ker_dex;      \
}


#define _CONV2_REGS_FP64_SHIFT_LOAD_2 {       \
    k_value = kernel[ker_dex];              \
    _SLIDING_WINDOW_FP64_SHIFT_LOAD_2_(_proc_reg, _store_reg);      \
    _accumulator = _mm256_fmadd_pd(_proc_reg, _mm256_set1_pd(k_value), _accumulator);                               \
    ++ker_dex;      \
}


#define _CONV2_REGS_FP64_SHIFT_LOAD_0 {       \
    k_value = kernel[ker_dex];              \
    _SLIDING_WINDOW_FP64_SHIFT_LOAD_0_(_proc_reg, _store_reg);      \
    _accumulator = _mm256_fmadd_pd(_proc_reg, _mm256_set1_pd(k_value), _accumulator);                               \
    ++ker_dex;      \
}


#define _CONV2_REGS_FP64_SHIFT_LOAD_1 {       \
    k_value = kernel[ker_dex];              \
    _SLIDING_WINDOW_FP64_SHIFT_LOAD_1_(_proc_reg, _store_reg);      \
    _accumulator = _mm256_fmadd_pd(_proc_reg, _mm256_set1_pd(k_value), _accumulator);                               \
    ++ker_dex;      \
}




_THREAD_CALL_
__m256d decx::conv::CPUK::_conv2_fp64_loop_in_kernel_2regs_rw2(const double* src, const double* kernel, const uint2 ker_dims, const size_t Wsrc)
{
    __m256d _proc_reg, _store_reg,
        _accumulator = _mm256_set1_pd(0);

    float k_value;      // kernel value
    uint ker_dex = 0;

    for (int i = 0; i < ker_dims.y; ++i)
    {
        _proc_reg = _mm256_load_pd(src + i * (Wsrc << 2));
        _store_reg = _mm256_load_pd(src + i * (Wsrc << 2) + 4);
        k_value = kernel[ker_dex];
        // first multiply-add with the first element of kernel on every row
        _accumulator = _mm256_fmadd_pd(_proc_reg, _mm256_set1_pd(k_value), _accumulator);
        ++ker_dex;
        _CONV2_REGS_FP64_SHIFT_LOAD_0;         _CONV2_REGS_FP64_SHIFT_LOAD_1;
    }
    return _accumulator;
}



_THREAD_CALL_
__m256d decx::conv::CPUK::_conv2_fp64_loop_in_kernel_2regs_rw4(const double* src, const double* kernel, const uint2 ker_dims, const size_t Wsrc)
{
    __m256d _proc_reg, _store_reg,
        _accumulator = _mm256_set1_pd(0);

    float k_value;      // kernel value
    uint ker_dex = 0;

    for (int i = 0; i < ker_dims.y; ++i)
    {
        _proc_reg = _mm256_load_pd(src + i * (Wsrc << 2));
        _store_reg = _mm256_load_pd(src + i * (Wsrc << 2) + 4);
        k_value = kernel[ker_dex];
        // first multiply-add with the first element of kernel on every row
        _accumulator = _mm256_fmadd_pd(_proc_reg, _mm256_set1_pd(k_value), _accumulator);
        ++ker_dex;
        _CONV2_REGS_FP64_SHIFT_LOAD_0;         _CONV2_REGS_FP64_SHIFT_LOAD_1;
        _CONV2_REGS_FP64_SHIFT_LOAD_2;         _CONV2_REGS_FP64_SHIFT_LOAD_3;
    }
    return _accumulator;
}



_THREAD_CALL_
__m256d decx::conv::CPUK::_conv2_fp64_loop_in_kernel_Nregs_rw2(const double* src, const double* kernel, const uint2 ker_dims,
    const size_t Wsrc, const uint _loop)
{
    __m256d _proc_reg, _store_reg,
        _accumulator = _mm256_set1_pd(0);

    float k_value;      // kernel value
    uint ker_dex = 0;

    for (int i = 0; i < ker_dims.y; ++i)
    {
        _proc_reg = _mm256_load_pd(src + i * (Wsrc << 2));
        _store_reg = _mm256_load_pd(src + i * (Wsrc << 2) + 4);
        k_value = kernel[ker_dex];
        // first multiply-add with the first element of kernel on every row
        _accumulator = _mm256_fmadd_pd(_proc_reg, _mm256_set1_pd(k_value), _accumulator);
        ++ker_dex;
        
        _CONV2_REGS_FP64_SHIFT_LOAD_0;         _CONV2_REGS_FP64_SHIFT_LOAD_1;
        _CONV2_REGS_FP64_SHIFT_LOAD_2;         _CONV2_REGS_FP64_SHIFT_LOAD_3;

        for (int _L = 1; _L < _loop + 1; ++_L){
            _store_reg = _mm256_load_pd(src + i * (Wsrc << 2) + 4 + (_L << 2));
            _CONV2_REGS_FP64_SHIFT_LOAD_0;         _CONV2_REGS_FP64_SHIFT_LOAD_1;
            _CONV2_REGS_FP64_SHIFT_LOAD_2;         _CONV2_REGS_FP64_SHIFT_LOAD_3;
        }

        _store_reg = _mm256_load_pd(src + i * (Wsrc << 2) + ((_loop + 2) << 2));
        _CONV2_REGS_FP64_SHIFT_LOAD_0;         _CONV2_REGS_FP64_SHIFT_LOAD_1;
    }
    return _accumulator;
}




_THREAD_CALL_
__m256d decx::conv::CPUK::_conv2_fp64_loop_in_kernel_Nregs_rw4(const double* src, const double* kernel, const uint2 ker_dims, 
    const size_t Wsrc, const uint _loop)
{
    __m256d _proc_reg, _store_reg,
        _accumulator = _mm256_set1_pd(0);

    float k_value;      // kernel value
    uint ker_dex = 0;

    for (int i = 0; i < ker_dims.y; ++i)
    {
        _proc_reg = _mm256_load_pd(src + i * (Wsrc << 2));
        _store_reg = _mm256_load_pd(src + i * (Wsrc << 2) + 4);
        k_value = kernel[ker_dex];
        // first multiply-add with the first element of kernel on every row
        _accumulator = _mm256_fmadd_pd(_proc_reg, _mm256_set1_pd(k_value), _accumulator);
        ++ker_dex;
        _CONV2_REGS_FP64_SHIFT_LOAD_0;         _CONV2_REGS_FP64_SHIFT_LOAD_1;
        _CONV2_REGS_FP64_SHIFT_LOAD_2;         _CONV2_REGS_FP64_SHIFT_LOAD_3;

        for (int _L = 1; _L < _loop + 1; ++_L) {
            _store_reg = _mm256_load_pd(src + i * (Wsrc << 2) + 4 + (_L << 2));
            _CONV2_REGS_FP64_SHIFT_LOAD_0;         _CONV2_REGS_FP64_SHIFT_LOAD_1;
            _CONV2_REGS_FP64_SHIFT_LOAD_2;         _CONV2_REGS_FP64_SHIFT_LOAD_3;
        }

        _store_reg = _mm256_load_pd(src + i * (Wsrc << 2) + ((_loop + 2) << 2));
        _CONV2_REGS_FP64_SHIFT_LOAD_0;         _CONV2_REGS_FP64_SHIFT_LOAD_1;
        _CONV2_REGS_FP64_SHIFT_LOAD_2;         _CONV2_REGS_FP64_SHIFT_LOAD_3;
    }
    return _accumulator;
}



#endif