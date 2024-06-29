/**
*   ----------------------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ----------------------------------------------------------------------------------
* 
* This is a part of the open source project named "DECX", a high-performance scientific
* computational library. This project follows the MIT License. For more information 
* please visit https://github.com/param0037/DECX.
* 
* Copyright (c) 2021 Wayne Anderson
* 
* Permission is hereby granted, free of charge, to any person obtaining a copy of this 
* software and associated documentation files (the "Software"), to deal in the Software 
* without restriction, including without limitation the rights to use, copy, modify, 
* merge, publish, distribute, sublicense, and/or sell copies of the Software, and to 
* permit persons to whom the Software is furnished to do so, subject to the following 
* conditions:
* 
* The above copyright notice and this permission notice shall be included in all copies 
* or substantial portions of the Software.
* 
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
* INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR 
* PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE 
* FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR 
* OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
* DEALINGS IN THE SOFTWARE.
*/


#ifndef _BILATERAL_FILTER_EXEC_H_
#define _BILATERAL_FILTER_EXEC_H_

#include "../../../core/basic.h"
#include "../../../DSP/convolution/CPU/sliding_window/uint8/conv2_uint8_K_loop_core.h"
#include "../../../core/thread_management/thread_pool.h"
#include "../../../core/utils/fragment_arrangment.h"
#include "../../../core/allocators.h"


#define _BILATERAL_UINT8_V16_CALC_ {                                                                                \
    reg1._vi = _mm256_cvtepu8_epi16(_proc_reg);                                                                     \
    reg2._vi = _mm256_cvtepu8_epi16(_proc_reg);                                                                     \
    reg1._vi = _mm256_abs_epi16(_mm256_sub_epi16(reg1._vi, _I_u16_v16));                                            \
    _distX = _exp_vals_dist[labs(i - ker_dims.y / 2)];                                                              \
    _distY = _exp_vals_dist[labs(_Y - ker_dims.x / 2)];                                                             \
                                                                                                                    \
    _W_reg = _mm256_mul_ps(_mm256_set1_ps(_distX * _distY),                                                         \
    _mm256_i32gather_ps(_exp_chart_diff, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(reg1._vi)), 4));              \
    _accu_W._v1 = _mm256_add_ps(_W_reg, _accu_W._v1);                                                               \
    _accumulator._v1 = _mm256_fmadd_ps(                                                                             \
        _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(reg2._vi))),                                \
        _W_reg, _accumulator._v1);                                                                                  \
                                                                                                                    \
    _W_reg = _mm256_mul_ps(_mm256_set1_ps(_distX * _distY),                                                         \
        _mm256_i32gather_ps(_exp_chart_diff, _mm256_cvtepi16_epi32(_mm256_extractf128_si256(reg1._vi, 1)), 4));     \
    _accu_W._v2 = _mm256_add_ps(_W_reg, _accu_W._v2);                                                               \
    _accumulator._v2 = _mm256_fmadd_ps(                                                                             \
        _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extractf128_si256(reg2._vi, 1))),                           \
        _W_reg, _accumulator._v2);                                                                                  \
    ++_Y;                                                                                                           \
}




#define _BILATERAL_UINT8_V8_CALC_ {                                                                                 \
    reg1._vi = _mm_cvtepu8_epi16(_proc_reg);                                                                        \
    reg2._vf = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(reg1._vi));                                                 \
    reg1._vi = _mm_abs_epi16(_mm_sub_epi16(reg1._vi, _I_u16_v8));                                                   \
                                                                                                                    \
    _distX = _exp_vals_dist[labs(i - ker_dims.y / 2)];                                                              \
    _distY = _exp_vals_dist[labs(_Y - ker_dims.x / 2)];                                                             \
                                                                                                                    \
    _W_reg = _mm256_mul_ps(_mm256_set1_ps(_distX * _distY),                                                         \
    _mm256_i32gather_ps(_exp_chart_diff, _mm256_cvtepi16_epi32(reg1._vi), 4));                                      \
                                                                                                                    \
    _accu_W = _mm256_add_ps(_W_reg, _accu_W);                                                                       \
    _accumulator = _mm256_fmadd_ps(                                                                                 \
    _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(reg2._vi))),                                    \
    _W_reg, _accumulator);                                                                                          \
    ++_Y;                                                                                                           \
}




#define _BILATERAL_UINT8_V16_LOAD_CVT_(_shif) {         \
    _SLIDING_WINDOW_UINT8_SHIFT_(_proc_reg, _shif);     \
    _BILATERAL_UINT8_V16_CALC_;                         \
}


#define _BILATERAL_UINT8_V8_LOAD_CVT_(_shif) {          \
    _SLIDING_WINDOW_UINT8_SHIFT_(_proc_reg, _shif);     \
    _BILATERAL_UINT8_V8_CALC_;                          \
}



namespace decx
{
    namespace vis 
    {
        _THREAD_FUNCTION_ void
            _bilateral_uint8_ST(const double* src, const float* _exp_chart_dist, const float* _exp_chart_diff,
                double* dst, const uint2 proc_dim, const uint2 ker_dims, const uint Wsrc,
                const uint Wdst, const ushort reg_WL, const uint _loop);


        void _bilateral_uint8_caller(const double* src, const float* _exp_chart_dist, const float* _exp_chart_diff, double* dst,
            const uint2 proc_dim, const uint2 ker_dims, const uint Wsrc, const uint Wdst, const ushort reg_WL,
            decx::utils::_thr_1D* t1D, decx::utils::frag_manager* f_mgr, const uint _loop);


        _THREAD_FUNCTION_ void
            _bilateral_uchar4_ST(const float* src, const float* _exp_chart_dist, const float* _exp_chart_diff,
                float* dst, const uint2 proc_dim, const uint2 ker_dims, const uint Wsrc,
                const uint Wdst, const ushort reg_WL, const uint _loop);


        void _bilateral_uchar4_caller(const float* src, const float* _exp_chart_dist, const float* _exp_chart_diff, float* dst,
            const uint2 proc_dim, const uint2 ker_dims, const uint Wsrc, const uint Wdst, const ushort reg_WL,
            decx::utils::_thr_1D* t1D, decx::utils::frag_manager* f_mgr, const uint _loop);
    }
}




#endif