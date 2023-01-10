/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/

#ifndef _RCP_CCOEFF_T_LOOP_CORE_H_
#define _RCP_CCOEFF_T_LOOP_CORE_H_

#include "rcp_sliding_window_avx_ops.h"
#include "../../../core/thread_management/thread_pool.h"
#include "../../../core/thread_management/thread_arrange.h"


namespace decx
{
    namespace rcp {
        namespace CPUK {
            // *** ATTENTION *** ! -> In this model, kernel should be stored linearly (pitch = width)
            /*
            * In this model, we only pay attention to the width of kernel, regardless its height
            * Since only the kernel width affects the behaviours during loading data from src matrix
            */


            /*
            * @param Wsrc : width of src matrix, in float
            * @param Wdst : width of dst matrix, in float
            */
            static inline _THREAD_CALL_
            __m256 _rcp_CCOEFF_fp32_loop_in_kernel_Nregs(const float* src, const float* kernel, const float* I_src, const uint2 ker_dims,
                const ushort reg_WL, const size_t Wsrc, const size_t W_Isrc, const uint _loop);


            /*
            * @param Wsrc : width of src matrix, in float
            * @param Wdst : width of dst matrix, in float
            * @param reg_WL : ( = ker_dims.x - 1 + 8 - 8 )
            */
            static inline _THREAD_CALL_
            __m256 _rcp_CCOEFF_fp32_loop_in_kernel_2regs(const float* src, const float* kernel, const float* I_src, const uint2 ker_dims, 
                const size_t Wsrc, const size_t W_Isrc);




            /*
            * @param Wsrc : width of src matrix, in float
            * @param Wdst : width of dst matrix, in float
            */
            static inline _THREAD_CALL_
            __m256 _rcp_CCOEFF_NORM_fp32_loop_in_kernel_Nregs(const float* src, const float* kernel, const float* I_src, const float _sqrt_k_sum, const uint2 ker_dims,
                const ushort reg_WL, const size_t Wsrc, const size_t W_Isrc, const uint _loop);


            /*
            * @param Wsrc : width of src matrix, in float
            * @param Wdst : width of dst matrix, in float
            * @param reg_WL : ( = ker_dims.x - 1 + 8 - 8 )
            */
            static inline _THREAD_CALL_
            __m256 _rcp_CCOEFF_NORM_fp32_loop_in_kernel_2regs(const float* src, const float* kernel, const float* I_src, const float _sqrt_k_sum, const uint2 ker_dims, 
                const size_t Wsrc, const size_t W_Isrc);
        }
    }
}



#define _RCP_CCOEFF_FP32_SHIFT_LOAD_GEN_(dex){                                          \
    k_value = kernel[ker_dex];                                                          \
    _SLIDING_WINDOW_FP32_GENERAL_(dex, _proc_reg, _store_reg, tmp2);                    \
    tmp2 = _mm256_sub_ps(_proc_reg, _sub_src);                                          \
    _accumulator = _mm256_fmadd_ps(tmp2, _mm256_set1_ps(k_value), _accumulator);        \
    ++ker_dex;                                                                          \
}



#define _RCP_CCOEFF_FP32_SHIFT_LOAD_0 {                                                 \
    k_value = kernel[ker_dex];                                                          \
    _SLIDING_WINDOW_FP32_LOAD_0_(_proc_reg, _store_reg, tmp2);                          \
    tmp2 = _mm256_sub_ps(_proc_reg, _sub_src);                                          \
    _accumulator = _mm256_fmadd_ps(tmp2, _mm256_set1_ps(k_value), _accumulator);        \
    ++ker_dex;                                                                          \
}


#define _RCP_CCOEFF_FP32_SHIFT_LOAD_1 {                                                 \
    k_value = kernel[ker_dex];                                                          \
    _SLIDING_WINDOW_FP32_LOAD_1_(_proc_reg, _store_reg, tmp2);                          \
    tmp2 = _mm256_sub_ps(_proc_reg, _sub_src);                                          \
    _accumulator = _mm256_fmadd_ps(tmp2, _mm256_set1_ps(k_value), _accumulator);        \
    ++ker_dex;                                                                          \
}


#define _RCP_CCOEFF_FP32_SHIFT_LOAD_2 {                                                 \
    k_value = kernel[ker_dex];                                                          \
    _SLIDING_WINDOW_FP32_LOAD_2_(_proc_reg, _store_reg, tmp2);                          \
    tmp2 = _mm256_sub_ps(_proc_reg, _sub_src);                                          \
    _accumulator = _mm256_fmadd_ps(tmp2, _mm256_set1_ps(k_value), _accumulator);        \
    ++ker_dex;                                                                          \
}



#define _RCP_CCOEFF_FP32_SHIFT_LOAD_3 {                                                 \
    k_value = kernel[ker_dex];                                                          \
    _SLIDING_WINDOW_FP32_LOAD_3_(_proc_reg, _store_reg, tmp2);                          \
    tmp2 = _mm256_sub_ps(_proc_reg, _sub_src);                                          \
    _accumulator = _mm256_fmadd_ps(tmp2, _mm256_set1_ps(k_value), _accumulator);        \
    ++ker_dex;                                                                          \
}



#define _RCP_CCOEFF_FP32_SHIFT_LOAD_4 {                               \
    k_value = kernel[ker_dex];                                        \
    _SLIDING_WINDOW_FP32_LOAD_4_(_proc_reg, _store_reg, tmp2);        \
    tmp2 = _mm256_sub_ps(_proc_reg, _sub_src);                                          \
    _accumulator = _mm256_fmadd_ps(tmp2, _mm256_set1_ps(k_value), _accumulator);        \
    ++ker_dex;                                                        \
}



#define _RCP_CCOEFF_FP32_SHIFT_LOAD_5 {                               \
    k_value = kernel[ker_dex];                                        \
    _SLIDING_WINDOW_FP32_LOAD_5_(_proc_reg, _store_reg, tmp2);        \
    tmp2 = _mm256_sub_ps(_proc_reg, _sub_src);                                          \
    _accumulator = _mm256_fmadd_ps(tmp2, _mm256_set1_ps(k_value), _accumulator);        \
    ++ker_dex;                                                        \
}



#define _RCP_CCOEFF_FP32_SHIFT_LOAD_6 {                               \
    k_value = kernel[ker_dex];                                        \
    _SLIDING_WINDOW_FP32_LOAD_6_(_proc_reg, _store_reg, tmp2);        \
    tmp2 = _mm256_sub_ps(_proc_reg, _sub_src);                                          \
    _accumulator = _mm256_fmadd_ps(tmp2, _mm256_set1_ps(k_value), _accumulator);        \
    ++ker_dex;                                                        \
}



#define _RCP_CCOEFF_FP32_SHIFT_LOAD_7 {                               \
    k_value = kernel[ker_dex];                                        \
    _SLIDING_WINDOW_FP32_LOAD_7_(_proc_reg, _store_reg, tmp2);        \
    tmp2 = _mm256_sub_ps(_proc_reg, _sub_src);                                          \
    _accumulator = _mm256_fmadd_ps(tmp2, _mm256_set1_ps(k_value), _accumulator);        \
    ++ker_dex;                                                        \
}




_THREAD_CALL_ __m256 
decx::rcp::CPUK::_rcp_CCOEFF_fp32_loop_in_kernel_Nregs(const float* __restrict     src, 
                                              const float* __restrict     kernel, 
                                              const float* __restrict     I_src,
                                              const uint2           ker_dims, 
                                              const ushort          reg_WL, 
                                              const size_t          Wsrc,
                                              const size_t          W_Isrc,
                                              const uint            _loop)
{
    register __m256 _proc_reg, _store_reg, tmp2, _sub_src,
        _accumulator = _mm256_set1_ps(0);

    float k_value;      // kernel value
    uint ker_dex = 0;

    _proc_reg = _mm256_add_ps(_mm256_loadu_ps(I_src + ker_dims.y * (W_Isrc << 3) + ker_dims.x), 
                              _mm256_loadu_ps(I_src - (W_Isrc << 3) - 1));
    _store_reg = _mm256_add_ps(_mm256_loadu_ps(I_src + ker_dims.y * (W_Isrc << 3) - 1), 
                               _mm256_loadu_ps(I_src + ker_dims.x - (W_Isrc << 3)));
    _sub_src = _mm256_sub_ps(_proc_reg, _store_reg);
    _sub_src = _mm256_div_ps(_sub_src, _mm256_set1_ps(ker_dims.x * ker_dims.y));

    for (int i = 0; i < ker_dims.y; ++i) 
    {
        _proc_reg = _mm256_load_ps(src + i * (Wsrc << 3));
        _store_reg = _mm256_load_ps(src + i * (Wsrc << 3) + 8);
        k_value = kernel[ker_dex];
        // first multiply-add with the first element of kernel on every row
        _accumulator = _mm256_fmadd_ps(_proc_reg, _mm256_set1_ps(k_value), _accumulator);
        ++ker_dex;

        _RCP_CCOEFF_FP32_SHIFT_LOAD_0;         _RCP_CCOEFF_FP32_SHIFT_LOAD_1;
        _RCP_CCOEFF_FP32_SHIFT_LOAD_2;         _RCP_CCOEFF_FP32_SHIFT_LOAD_3;
        _RCP_CCOEFF_FP32_SHIFT_LOAD_4;         _RCP_CCOEFF_FP32_SHIFT_LOAD_5;
        _RCP_CCOEFF_FP32_SHIFT_LOAD_6;         _RCP_CCOEFF_FP32_SHIFT_LOAD_7;

        for (uint _L = 0; _L < _loop; ++_L) {
            _store_reg = _mm256_load_ps(src + i * (Wsrc << 3) + 16 + (_L << 3));
            _RCP_CCOEFF_FP32_SHIFT_LOAD_0;         _RCP_CCOEFF_FP32_SHIFT_LOAD_1;
            _RCP_CCOEFF_FP32_SHIFT_LOAD_2;         _RCP_CCOEFF_FP32_SHIFT_LOAD_3;
            _RCP_CCOEFF_FP32_SHIFT_LOAD_4;         _RCP_CCOEFF_FP32_SHIFT_LOAD_5;
            _RCP_CCOEFF_FP32_SHIFT_LOAD_6;         _RCP_CCOEFF_FP32_SHIFT_LOAD_7;
        }
        _store_reg = _mm256_load_ps(src + i * (Wsrc << 3) + 16 + (_loop << 3));

        for (int j = 0; j < reg_WL; ++j) {
            _RCP_CCOEFF_FP32_SHIFT_LOAD_GEN_(j);
        }
    }
    return _accumulator;
}



_THREAD_CALL_ __m256 
decx::rcp::CPUK::_rcp_CCOEFF_fp32_loop_in_kernel_2regs(const float* __restrict     src, 
                                              const float* __restrict     kernel,
                                              const float* __restrict     I_src,
                                              const uint2           ker_dims, 
                                              const size_t          Wsrc,
                                              const size_t          W_Isrc)
{
    register __m256 _proc_reg, _store_reg, tmp2, _sub_src,
        _accumulator = _mm256_set1_ps(0);

    float k_value;      // kernel value
    uint ker_dex = 0;

    const float* Isrc_origin = I_src - (W_Isrc << 3) - 1;
    _proc_reg = _mm256_add_ps(_mm256_loadu_ps(Isrc_origin + ker_dims.y * (W_Isrc << 3) + ker_dims.x),
        _mm256_loadu_ps(Isrc_origin));
    _store_reg = _mm256_add_ps(_mm256_loadu_ps(Isrc_origin + ker_dims.y * (W_Isrc << 3)),
        _mm256_loadu_ps(Isrc_origin + ker_dims.x));
    _sub_src = _mm256_sub_ps(_proc_reg, _store_reg);
    _sub_src = _mm256_div_ps(_sub_src, _mm256_set1_ps(ker_dims.x * ker_dims.y));

    for (int i = 0; i < ker_dims.y; ++i) 
    {
        _proc_reg = _mm256_load_ps(src + i * (Wsrc << 3));
        _store_reg = _mm256_load_ps(src + i * (Wsrc << 3) + 8);
        k_value = kernel[ker_dex];
        // first multiply-add with the first element of kernel on every row
        _accumulator = _mm256_fmadd_ps(_proc_reg, _mm256_set1_ps(k_value), _accumulator);
        ++ker_dex;

        for (int _internal_dex = 0; _internal_dex < ker_dims.x - 1; ++_internal_dex)
        {
            k_value = kernel[ker_dex];
            switch (_internal_dex)
            {
            case 7:
                tmp2 = _mm256_sub_ps(_proc_reg, _sub_src);
                _accumulator = _mm256_fmadd_ps(tmp2, _mm256_set1_ps(k_value), _accumulator);
                break;

            default:
                _proc_reg = _mm256_permute_ps(_proc_reg, _MM_SHUFFLE(0, 3, 2, 1));
                tmp2 = _mm256_permute2f128_ps(_proc_reg, _mm256_permutevar8x32_ps(_store_reg, _mm256_set1_epi32(_internal_dex)), 0b00100001);
                _proc_reg = _mm256_blend_ps(_proc_reg, tmp2, 0b10001000);
                tmp2 = _mm256_sub_ps(_proc_reg, _sub_src);
                _accumulator = _mm256_fmadd_ps(tmp2, _mm256_set1_ps(k_value), _accumulator);
                break;
            }
            
            ++ker_dex;
        }
    }
    return _accumulator;
}



#define _RCP_CCOEFF_NORM_FP32_SHIFT_LOAD_GEN_(dex){                           \
    k_value = kernel[ker_dex];                                                \
    _SLIDING_WINDOW_FP32_GENERAL_(dex, _proc_reg, _store_reg, tmp2);          \
    tmp2 = _mm256_sub_ps(_proc_reg, _sub_src);                                          \
    _accumulator = _mm256_fmadd_ps(tmp2, _mm256_set1_ps(k_value), _accumulator);        \
    _src_sq_sum = _mm256_fmadd_ps(tmp2, tmp2, _src_sq_sum);                             \
    ++ker_dex;                                                                \
}



#define _RCP_CCOEFF_NORM_FP32_SHIFT_LOAD_0 {                                            \
    k_value = kernel[ker_dex];                                                          \
    _SLIDING_WINDOW_FP32_LOAD_0_(_proc_reg, _store_reg, tmp2);                          \
    tmp2 = _mm256_sub_ps(_proc_reg, _sub_src);                                          \
    _accumulator = _mm256_fmadd_ps(tmp2, _mm256_set1_ps(k_value), _accumulator);        \
    _src_sq_sum = _mm256_fmadd_ps(tmp2, tmp2, _src_sq_sum);                             \
    ++ker_dex;                                                                          \
}


#define _RCP_CCOEFF_NORM_FP32_SHIFT_LOAD_1 {                                                                        \
    k_value = kernel[ker_dex];                                                                                      \
    _SLIDING_WINDOW_FP32_LOAD_1_(_proc_reg, _store_reg, tmp2);                                                      \
    tmp2 = _mm256_sub_ps(_proc_reg, _sub_src);                                          \
    _accumulator = _mm256_fmadd_ps(tmp2, _mm256_set1_ps(k_value), _accumulator);        \
    _src_sq_sum = _mm256_fmadd_ps(tmp2, tmp2, _src_sq_sum);                             \
    ++ker_dex;                                                                                                      \
}


#define _RCP_CCOEFF_NORM_FP32_SHIFT_LOAD_2 {                                                                        \
    k_value = kernel[ker_dex];                                                                                      \
    _SLIDING_WINDOW_FP32_LOAD_2_(_proc_reg, _store_reg, tmp2);                                                      \
    tmp2 = _mm256_sub_ps(_proc_reg, _sub_src);                                          \
    _accumulator = _mm256_fmadd_ps(tmp2, _mm256_set1_ps(k_value), _accumulator);        \
    _src_sq_sum = _mm256_fmadd_ps(tmp2, tmp2, _src_sq_sum);                             \
    ++ker_dex;                                                                                                      \
}



#define _RCP_CCOEFF_NORM_FP32_SHIFT_LOAD_3 {                                                                        \
    k_value = kernel[ker_dex];                                                                                      \
    _SLIDING_WINDOW_FP32_LOAD_3_(_proc_reg, _store_reg, tmp2);                                                      \
    tmp2 = _mm256_sub_ps(_proc_reg, _sub_src);                                          \
    _accumulator = _mm256_fmadd_ps(tmp2, _mm256_set1_ps(k_value), _accumulator);        \
    _src_sq_sum = _mm256_fmadd_ps(tmp2, tmp2, _src_sq_sum);                             \
    ++ker_dex;                                                                                                      \
}



#define _RCP_CCOEFF_NORM_FP32_SHIFT_LOAD_4 {                                                                        \
    k_value = kernel[ker_dex];                                                                                      \
    _SLIDING_WINDOW_FP32_LOAD_4_(_proc_reg, _store_reg, tmp2);                                                      \
    tmp2 = _mm256_sub_ps(_proc_reg, _sub_src);                                          \
    _accumulator = _mm256_fmadd_ps(tmp2, _mm256_set1_ps(k_value), _accumulator);        \
    _src_sq_sum = _mm256_fmadd_ps(tmp2, tmp2, _src_sq_sum);                             \
    ++ker_dex;                                                                                                      \
}



#define _RCP_CCOEFF_NORM_FP32_SHIFT_LOAD_5 {                                                                        \
    k_value = kernel[ker_dex];                                                                                      \
    _SLIDING_WINDOW_FP32_LOAD_5_(_proc_reg, _store_reg, tmp2);                                                      \
    tmp2 = _mm256_sub_ps(_proc_reg, _sub_src);                                          \
    _accumulator = _mm256_fmadd_ps(tmp2, _mm256_set1_ps(k_value), _accumulator);        \
    _src_sq_sum = _mm256_fmadd_ps(tmp2, tmp2, _src_sq_sum);                             \
    ++ker_dex;                                                                                                      \
}



#define _RCP_CCOEFF_NORM_FP32_SHIFT_LOAD_6 {                                                                        \
    k_value = kernel[ker_dex];                                                                                      \
    _SLIDING_WINDOW_FP32_LOAD_6_(_proc_reg, _store_reg, tmp2);                                                      \
    tmp2 = _mm256_sub_ps(_proc_reg, _sub_src);                                          \
    _accumulator = _mm256_fmadd_ps(tmp2, _mm256_set1_ps(k_value), _accumulator);        \
    _src_sq_sum = _mm256_fmadd_ps(tmp2, tmp2, _src_sq_sum);                             \
    ++ker_dex;                                                                                                      \
}



#define _RCP_CCOEFF_NORM_FP32_SHIFT_LOAD_7 {                                                                        \
    k_value = kernel[ker_dex];                                                                                      \
    _SLIDING_WINDOW_FP32_LOAD_7_(_proc_reg, _store_reg, tmp2);                                                      \
    tmp2 = _mm256_sub_ps(_proc_reg, _sub_src);                                          \
    _accumulator = _mm256_fmadd_ps(tmp2, _mm256_set1_ps(k_value), _accumulator);        \
    _src_sq_sum = _mm256_fmadd_ps(tmp2, tmp2, _src_sq_sum);                             \
    ++ker_dex;                                                                                                      \
}



_THREAD_CALL_ __m256 
decx::rcp::CPUK::_rcp_CCOEFF_NORM_fp32_loop_in_kernel_Nregs(const float* __restrict     src, 
                                              const float* __restrict     kernel, 
                                              const float* __restrict     I_src,
                                              const float           _sqrt_k_sum,
                                              const uint2           ker_dims, 
                                              const ushort          reg_WL, 
                                              const size_t          Wsrc,
                                              const size_t          W_Isrc,
                                              const uint            _loop)
{
    register __m256 _proc_reg, _store_reg, tmp2, _sub_src,
        _accumulator = _mm256_set1_ps(0),
        _src_sq_sum = _mm256_set1_ps(0);

    const float* Isrc_origin = I_src - (W_Isrc << 3) - 1;
    _proc_reg = _mm256_add_ps(_mm256_loadu_ps(Isrc_origin + ker_dims.y * (W_Isrc << 3) + ker_dims.x),
        _mm256_loadu_ps(Isrc_origin));
    _store_reg = _mm256_add_ps(_mm256_loadu_ps(Isrc_origin + ker_dims.y * (W_Isrc << 3)),
        _mm256_loadu_ps(Isrc_origin + ker_dims.x));
    _sub_src = _mm256_sub_ps(_proc_reg, _store_reg);
    _sub_src = _mm256_div_ps(_sub_src, _mm256_set1_ps(ker_dims.x * ker_dims.y));

    float k_value;      // kernel value
    uint ker_dex = 0;

    for (int i = 0; i < ker_dims.y; ++i) 
    {
        _proc_reg = _mm256_load_ps(src + i * (Wsrc << 3));
        _store_reg = _mm256_load_ps(src + i * (Wsrc << 3) + 8);
        k_value = kernel[ker_dex];
        // first multiply-add with the first element of kernel on every row
        _accumulator = _mm256_fmadd_ps(_proc_reg, _mm256_set1_ps(k_value), _accumulator);
        ++ker_dex;

        _RCP_CCOEFF_NORM_FP32_SHIFT_LOAD_0;         _RCP_CCOEFF_NORM_FP32_SHIFT_LOAD_1;
        _RCP_CCOEFF_NORM_FP32_SHIFT_LOAD_2;         _RCP_CCOEFF_NORM_FP32_SHIFT_LOAD_3;
        _RCP_CCOEFF_NORM_FP32_SHIFT_LOAD_4;         _RCP_CCOEFF_NORM_FP32_SHIFT_LOAD_5;
        _RCP_CCOEFF_NORM_FP32_SHIFT_LOAD_6;         _RCP_CCOEFF_NORM_FP32_SHIFT_LOAD_7;

        for (uint _L = 0; _L < _loop; ++_L) {
            _store_reg = _mm256_load_ps(src + i * (Wsrc << 3) + 16 + (_L << 3));
            _RCP_CCOEFF_NORM_FP32_SHIFT_LOAD_0;         _RCP_CCOEFF_NORM_FP32_SHIFT_LOAD_1;
            _RCP_CCOEFF_NORM_FP32_SHIFT_LOAD_2;         _RCP_CCOEFF_NORM_FP32_SHIFT_LOAD_3;
            _RCP_CCOEFF_NORM_FP32_SHIFT_LOAD_4;         _RCP_CCOEFF_NORM_FP32_SHIFT_LOAD_5;
            _RCP_CCOEFF_NORM_FP32_SHIFT_LOAD_6;         _RCP_CCOEFF_NORM_FP32_SHIFT_LOAD_7;
        }
        _store_reg = _mm256_load_ps(src + i * (Wsrc << 3) + 16 + (_loop << 3));

        for (int j = 0; j < reg_WL; ++j) {
            _RCP_CCOEFF_NORM_FP32_SHIFT_LOAD_GEN_(j);
        }
    }
    _accumulator = _mm256_div_ps(_accumulator, _mm256_mul_ps(_mm256_set1_ps(_sqrt_k_sum), _mm256_sqrt_ps(_src_sq_sum)));
    return _accumulator;
}



_THREAD_CALL_ __m256 
decx::rcp::CPUK::_rcp_CCOEFF_NORM_fp32_loop_in_kernel_2regs(const float* __restrict     src, 
                                              const float* __restrict     kernel, 
                                              const float* __restrict     I_src,
                                              const float           _sqrt_k_sum,
                                              const uint2           ker_dims, 
                                              const size_t          Wsrc,
                                              const size_t          W_Isrc)
{
    register __m256 _proc_reg, _store_reg, tmp2, _sub_src,
        _accumulator = _mm256_set1_ps(0),
        _src_sq_sum = _mm256_set1_ps(0);

    float k_value;      // kernel value
    uint ker_dex = 0;

    const float* Isrc_origin = I_src - (W_Isrc << 3) - 1;
    _proc_reg = _mm256_add_ps(_mm256_loadu_ps(Isrc_origin + ker_dims.y * (W_Isrc << 3) + ker_dims.x),
        _mm256_loadu_ps(Isrc_origin));
    _store_reg = _mm256_add_ps(_mm256_loadu_ps(Isrc_origin + ker_dims.y * (W_Isrc << 3)),
        _mm256_loadu_ps(Isrc_origin + ker_dims.x));
    _sub_src = _mm256_sub_ps(_proc_reg, _store_reg);
    _sub_src = _mm256_div_ps(_sub_src, _mm256_set1_ps(ker_dims.x * ker_dims.y));

    for (int i = 0; i < ker_dims.y; ++i) 
    {
        _proc_reg = _mm256_load_ps(src + i * (Wsrc << 3));
        _store_reg = _mm256_load_ps(src + i * (Wsrc << 3) + 8);
        k_value = kernel[ker_dex];
        // first multiply-add with the first element of kernel on every row
        _accumulator = _mm256_fmadd_ps(_proc_reg, _mm256_set1_ps(k_value), _accumulator);
        ++ker_dex;

        for (int _internal_dex = 0; _internal_dex < ker_dims.x - 1; ++_internal_dex)
        {
            k_value = kernel[ker_dex];
            switch (_internal_dex)
            {
            case 7:
                tmp2 = _mm256_sub_ps(_proc_reg, _sub_src);
                _accumulator = _mm256_fmadd_ps(tmp2, _mm256_set1_ps(k_value), _accumulator);
                _src_sq_sum = _mm256_fmadd_ps(_store_reg, _store_reg, _src_sq_sum);
                break;

            default:
                _RCP_CCOEFF_NORM_FP32_SHIFT_LOAD_GEN_(_internal_dex);
                break;
            }
            
            ++ker_dex;
        }
    }
    _accumulator = _mm256_div_ps(_accumulator, _mm256_mul_ps(_mm256_set1_ps(_sqrt_k_sum), _mm256_sqrt_ps(_src_sq_sum)));
    return _accumulator;
}


// --------------------------------------------------- uint8 ----------------------------------------------------------



namespace decx
{
    namespace rcp {
        namespace CPUK {
            // *** ATTENTION *** ! -> In this model, kernel should be stored linearly (pitch = width)
            /*
            * In this model, we only pay attention to the width of kernel, regardless its height
            * Since only the kernel width affects the behaviours during loading data from src matrix
            */


            /*
            * @param Wsrc : width of src matrix, in double (1 double = 8 uint8_t)
            * @param Wdst : width of dst matrix, in float
            */
            static _THREAD_CALL_
            __m256 _rcp2_CCOEFF_uint8_loop_in_kernel(const double* src, const float* kernel, const float* _Isrc, const uint2 ker_dims,
                    const ushort reg_WL, const size_t Wsrc, const uint W_Isrc, const uint _loop);



            static _THREAD_CALL_
            __m256 _rcp2_CCOEFF_NORM_uint8_loop_in_kernel(const double* src, const float* kernel, const float* _Isrc, 
                const float _k_sq_sum, const uint2 ker_dims,
                    const ushort reg_WL, const size_t Wsrc, const uint W_Isrc, const uint _loop);

        }
    }
}



_THREAD_CALL_ __m256
decx::rcp::CPUK::_rcp2_CCOEFF_uint8_loop_in_kernel(const double* __restrict     src,
                                                       const float* __restrict     kernel, 
                                                       const float* __restrict         I_src,
                                                       const uint2           ker_dims, 
                                                       const ushort          reg_WL, 
                                                       const size_t          Wsrc,
                                                       const uint            W_Isrc,
                                                       const uint            _loop)
{
    uint8_t _store_reg[40];
    register __m128i _proc_reg;
    __m256 _sub_src;
    decx::utils::simd::_mmv256 reg1, reg2;

    const float* Isrc_origin = (float*)I_src - (W_Isrc << 3) - 1;
    reg1._vf = _mm256_add_ps(_mm256_loadu_ps(Isrc_origin + ker_dims.y * (W_Isrc << 3) + ker_dims.x),
        _mm256_loadu_ps(Isrc_origin));
    reg2._vf = _mm256_add_ps(_mm256_loadu_ps(Isrc_origin + ker_dims.y * (W_Isrc << 3)),
        _mm256_loadu_ps(Isrc_origin + ker_dims.x));
    _sub_src = _mm256_sub_ps(reg1._vf, reg2._vf);
    _sub_src = _mm256_div_ps(_sub_src, _mm256_set1_ps(ker_dims.x * ker_dims.y));

    __m256 _accumulator = _mm256_set1_ps(0);

    uint k_value;      // kernel value
    uint ker_dex = 0;

    for (int i = 0; i < ker_dims.y; ++i) 
    {
        for (uint j = 0; j < _loop; ++j) {
            _mm256_store_pd((double*)_store_reg, _mm256_loadu_pd(src + i * Wsrc + j * 3));
#ifdef _MSC_VER
            _proc_reg = _mm_loadu_epi8(_store_reg);
#endif
#ifdef __GNUC__
            _proc_reg = _mm_castpd_si128(_mm_loadu_pd((double*)_store_reg));
#endif
            reg1._vf = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_proc_reg));
            reg2._vf = _mm256_sub_ps(reg1._vf, _sub_src);
            _accumulator = _mm256_fmadd_ps(reg2._vf, _mm256_set1_ps(kernel[ker_dex]), _accumulator);
            ++ker_dex;

            for (int k = 0; k < 23; ++k) {
                _SLIDING_WINDOW_UINT8_SHIFT_(_proc_reg, k + 1);
                reg1._vf = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_proc_reg));
                reg2._vf = _mm256_sub_ps(reg1._vf, _sub_src);
                _accumulator = _mm256_fmadd_ps(reg2._vf, _mm256_set1_ps(kernel[ker_dex]), _accumulator);
                ++ker_dex;
            }
        }

        if (reg_WL != 0) {
            _mm256_store_pd((double*)_store_reg, _mm256_loadu_pd(src + i * Wsrc + _loop * 3));
#ifdef _MSC_VER
            _proc_reg = _mm_loadu_epi8(_store_reg);
#endif
#ifdef __GNUC__
            _proc_reg = _mm_castpd_si128(_mm_loadu_pd((double*)_store_reg));
#endif

            reg1._vf = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_proc_reg));
            reg2._vf = _mm256_sub_ps(reg1._vf, _sub_src);
            _accumulator = _mm256_fmadd_ps(reg2._vf, _mm256_set1_ps(kernel[ker_dex]), _accumulator);
            ++ker_dex;

            for (int j = 0; j < reg_WL - 1; ++j) {
                _SLIDING_WINDOW_UINT8_SHIFT_(_proc_reg, j + 1);
                reg1._vf = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_proc_reg));
                reg2._vf = _mm256_sub_ps(reg1._vf, _sub_src);
                _accumulator = _mm256_fmadd_ps(reg2._vf, _mm256_set1_ps(kernel[ker_dex]), _accumulator);
                ++ker_dex;
            }
        }
    }
    return _accumulator;
}



_THREAD_CALL_ __m256
decx::rcp::CPUK::_rcp2_CCOEFF_NORM_uint8_loop_in_kernel(const double* __restrict     src,
                                                   const float* __restrict     kernel, 
                                                   const float* __restrict         I_src,
                                                   const float           _k_sq_sum,
                                                   const uint2           ker_dims, 
                                                   const ushort          reg_WL, 
                                                   const size_t          Wsrc,
                                                   const uint            W_Isrc,
                                                   const uint            _loop)
{
    uint8_t _store_reg[40];
    register __m128i _proc_reg;
    __m256 _sub_src, _src_sq_sum = _mm256_set1_ps(0);
    decx::utils::simd::_mmv256 reg1, reg2;

    const float* Isrc_origin = (float*)I_src - (W_Isrc << 3) - 1;
    reg1._vf = _mm256_add_ps(_mm256_loadu_ps(Isrc_origin + ker_dims.y * (W_Isrc << 3) + ker_dims.x),
        _mm256_loadu_ps(Isrc_origin));
    reg2._vf = _mm256_add_ps(_mm256_loadu_ps(Isrc_origin + ker_dims.y * (W_Isrc << 3)),
        _mm256_loadu_ps(Isrc_origin + ker_dims.x));
    _sub_src = _mm256_sub_ps(reg1._vf, reg2._vf);
    _sub_src = _mm256_div_ps(_sub_src, _mm256_set1_ps(ker_dims.x * ker_dims.y));

    __m256 _accumulator = _mm256_set1_ps(0);

    uint k_value;      // kernel value
    uint ker_dex = 0;

    for (int i = 0; i < ker_dims.y; ++i) 
    {
        for (uint j = 0; j < _loop; ++j) {
            _mm256_store_pd((double*)_store_reg, _mm256_loadu_pd(src + i * Wsrc + j * 3));
#ifdef _MSC_VER
            _proc_reg = _mm_loadu_epi8(_store_reg);
#endif
#ifdef __GNUC__
            _proc_reg = _mm_castpd_si128(_mm_loadu_pd((double*)_store_reg));
#endif
            reg1._vf = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_proc_reg));
            _src_sq_sum = _mm256_fmadd_ps(reg1._vf, reg1._vf, _src_sq_sum);
            reg2._vf = _mm256_sub_ps(reg1._vf, _sub_src);
            _accumulator = _mm256_fmadd_ps(reg2._vf, _mm256_set1_ps(kernel[ker_dex]), _accumulator);
            ++ker_dex;

            for (int k = 0; k < 23; ++k) {
                _SLIDING_WINDOW_UINT8_SHIFT_(_proc_reg, k + 1);
                reg1._vf = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_proc_reg));
                _src_sq_sum = _mm256_fmadd_ps(reg1._vf, reg1._vf, _src_sq_sum);
                reg2._vf = _mm256_sub_ps(reg1._vf, _sub_src);
                _accumulator = _mm256_fmadd_ps(reg2._vf, _mm256_set1_ps(kernel[ker_dex]), _accumulator);
                ++ker_dex;
            }
        }

        if (reg_WL != 0) {
            _mm256_store_pd((double*)_store_reg, _mm256_loadu_pd(src + i * Wsrc + _loop * 3));
#ifdef _MSC_VER
            _proc_reg = _mm_loadu_epi8(_store_reg);
#endif
#ifdef __GNUC__
            _proc_reg = _mm_castpd_si128(_mm_loadu_pd((double*)_store_reg));
#endif

            reg1._vf = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_proc_reg));
            _src_sq_sum = _mm256_fmadd_ps(reg1._vf, reg1._vf, _src_sq_sum);
            reg2._vf = _mm256_sub_ps(reg1._vf, _sub_src);
            _accumulator = _mm256_fmadd_ps(reg2._vf, _mm256_set1_ps(kernel[ker_dex]), _accumulator);
            ++ker_dex;

            for (int j = 0; j < reg_WL - 1; ++j) {
                _SLIDING_WINDOW_UINT8_SHIFT_(_proc_reg, j + 1);
                reg1._vf = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_proc_reg));
                _src_sq_sum = _mm256_fmadd_ps(reg1._vf, reg1._vf, _src_sq_sum);
                reg2._vf = _mm256_sub_ps(reg1._vf, _sub_src);
                _accumulator = _mm256_fmadd_ps(reg2._vf, _mm256_set1_ps(kernel[ker_dex]), _accumulator);
                ++ker_dex;
            }
        }
    }
    return _mm256_div_ps(_accumulator, _mm256_mul_ps(_mm256_set1_ps(_k_sq_sum), _mm256_sqrt_ps(_src_sq_sum)));
}



#endif