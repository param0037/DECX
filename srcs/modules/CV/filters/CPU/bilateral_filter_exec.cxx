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


#include "bilateral_filter_exec.h"


namespace decx{
namespace vis {
    namespace CPUK
    {
        // *** ATTENTION *** ! -> In this model, kernel should be stored linearly (pitch = width)
        /*
        * In this model, we only pay attention to the width of kernel, regardless its height
        * Since only the kernel width affects the behaviours during loading data from src matrix
        */

        //static _THREAD_CALL_ void _bilateral_calc_v16(const float* _exp_chart_dist, const float* _exp_chart_diff, const __m256i _I_u16_v16,
        //    const __m128i _proc_reg, const uint32_t i, const uint32_t _Y, const uint2 ker_dims, decx::conv::_v256_2f32* _accu_W,
        //    decx::conv::_v256_2f32* _accumulator);

        /*
        * @param Wsrc : width of src matrix, in double (1 double = 8 uint8_t)
        * @param Wdst : width of dst matrix, in float
        */
        static _THREAD_CALL_ decx::conv::_v256_2f32 
        _bilateral_uint8_f32_loop_in_kernel(const double* src, const float* _exp_chart_dist, const float* _exp_chart_diff,
            const uint2 ker_dims, const ushort reg_WL, const uint64_t Wsrc, const uint32_t _loop);



        static _THREAD_CALL_ decx::conv::_v256_2f32
        _bilateral_uchar4_f32_loop_in_kernel(const float* src, const float* _exp_chart_dist, const float* _exp_chart_diff,
            const uint2 ker_dims, const ushort reg_WL, const uint64_t Wsrc, const uint32_t _loop);
    }
}
}




namespace decx
{
namespace vis {
    namespace CPUK
    {
        static _THREAD_CALL_ void _bilateral_rect_fixed_uint8_ST(const double* src, const float* _exp_vals_dist, const float* _exp_chart_diff, double* dst,
            const uint2 ker_dims, const uint32_t Wsrc, const uint32_t Wdst, const ushort reg_WL, const uint32_t _loop);



        static _THREAD_CALL_ void _bilateral_rect_flex_uint8_ST(const double* src, const float* _exp_vals_dist, const float* _exp_chart_diff, double* dst,
            const uint2 proc_dim, const uint2 ker_dims, const uint32_t Wsrc, const uint32_t Wdst, const ushort reg_WL, const uint32_t _loop);



        static _THREAD_CALL_ void _bilateral_rect_fixed_uchar4_ST(const float* src, const float* _exp_vals_dist, const float* _exp_chart_diff, float* dst,
            const uint2 ker_dims, const uint32_t Wsrc, const uint32_t Wdst, const ushort reg_WL, const uint32_t _loop);



        static _THREAD_CALL_ void _bilateral_rect_flex_uchar4_ST(const float* src, const float* _exp_vals_dist, const float* _exp_chart_diff, float* dst,
            const uint2 proc_dim, const uint2 ker_dims, const uint32_t Wsrc, const uint32_t Wdst, const ushort reg_WL, const uint32_t _loop);
    }
}
}



_THREAD_CALL_ decx::conv::_v256_2f32
decx::vis::CPUK::_bilateral_uint8_f32_loop_in_kernel(const double* __restrict       src,
                                                      const float* __restrict       _exp_vals_dist,
                                                      const float* __restrict       _exp_chart_diff,
                                                      const uint2                   ker_dims, 
                                                      const ushort                  reg_WL, 
                                                      const uint64_t                  Wsrc,
                                                      const uint32_t                    _loop)
{
    uint8_t _store_reg[32];
    __m128i _proc_reg;

    const __m256i _I_u16_v16 = _mm256_cvtepu8_epi16(_mm_castpd_si128(_mm_loadu_pd(
        (double*)((uint8_t*)src + (ker_dims.y / 2) * Wsrc * 8 + (ker_dims.x / 2)) )));

    decx::utils::simd::xmm256_reg reg1, reg2, reg3;
    decx::conv::_v256_2f32 _accu_W, _accumulator;
    _accu_W._v1 = _mm256_setzero_ps();
    _accu_W._v2 = _mm256_setzero_ps();
    _accumulator._v1 = _mm256_setzero_ps();
    _accumulator._v2 = _mm256_setzero_ps();

    float _distX, _distY;
    uint32_t _Y = 0;

    __m256 _W_reg;

    for (uint32_t i = 0; i < ker_dims.y; ++i) 
    {
        _Y = 0;
        for (uint32_t j = 0; j < _loop; ++j) 
        {
            _mm256_store_pd((double*)_store_reg, _mm256_loadu_pd(src + i * Wsrc + j * 2));
#ifdef _MSC_VER
            _proc_reg = _mm_loadu_epi8(_store_reg);
#endif
#ifdef __GNUC__
            _proc_reg = _mm_castpd_si128(_mm_loadu_pd((double*)_store_reg));
#endif
            _BILATERAL_UINT8_V16_CALC_;

            for (int k = 0; k < 15; ++k) {
                _BILATERAL_UINT8_V16_LOAD_CVT_(k + 1);
            }
        }

        if (reg_WL != 0) {
            _mm256_store_pd((double*)_store_reg, _mm256_loadu_pd(src + i * Wsrc + _loop * 2));
#ifdef _MSC_VER
            _proc_reg = _mm_loadu_epi8(_store_reg);
#endif
#ifdef __GNUC__
            _proc_reg = _mm_castpd_si128(_mm_loadu_pd((double*)_store_reg));
#endif
            _BILATERAL_UINT8_V16_CALC_;

            for (int j = 0; j < reg_WL - 1; ++j) {
                _BILATERAL_UINT8_V16_LOAD_CVT_(j + 1);
            }
        }
    }
    _accumulator._v1 = _mm256_div_ps(_accumulator._v1, _accu_W._v1);
    _accumulator._v2 = _mm256_div_ps(_accumulator._v2, _accu_W._v2);
    return _accumulator;
}





_THREAD_CALL_ decx::conv::_v256_2f32
decx::vis::CPUK::_bilateral_uchar4_f32_loop_in_kernel(const float* __restrict       src,
                                                      const float* __restrict       _exp_vals_dist,
                                                      const float* __restrict       _exp_chart_diff,
                                                      const uint2                   ker_dims, 
                                                      const ushort                  reg_WL, 
                                                      const uint64_t                  Wsrc,
                                                      const uint32_t                    _loop)
{
    __m128i _proc_reg, aux_reg;
    
    const __m256i _I_u16_v16 = _mm256_cvtepu8_epi16(_mm_castps_si128(_mm_loadu_ps(
        src + (ker_dims.y / 2) * Wsrc + (ker_dims.x / 2) )));

    decx::utils::simd::xmm256_reg reg1, reg2, reg3;
    decx::conv::_v256_2f32 _accu_W, _accumulator;
    _accu_W._v1 = _mm256_setzero_ps();
    _accu_W._v2 = _mm256_setzero_ps();
    _accumulator._v1 = _mm256_setzero_ps();
    _accumulator._v2 = _mm256_setzero_ps();

    float _distX, _distY;
    uint32_t _Y = 0;

    __m256 _W_reg;

    for (int i = 0; i < ker_dims.y; ++i) 
    {
        _Y = 0;
        for (uint32_t j = 0; j < _loop; ++j) 
        {
            if (j == 0) {
                _proc_reg = _mm_castps_si128(_mm_load_ps(src + i * Wsrc));
            }
            else {
                _proc_reg = _mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(aux_reg), 0b00111001));
            }
            aux_reg = _mm_castps_si128(_mm_load_ps(src + i * Wsrc + (j << 2) + 4));

            _BILATERAL_UINT8_V16_CALC_;

            for (int k = 0; k < 3; ++k) {
                _proc_reg = _mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(_proc_reg), 0b00111001));
                aux_reg = _mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(aux_reg), 0b00111001));
                _proc_reg = _mm_castps_si128(_mm_blend_ps(_mm_castsi128_ps(_proc_reg), _mm_castsi128_ps(aux_reg), 0b1000));

                _BILATERAL_UINT8_V16_CALC_;
            }
        }

        if (_loop == 0) {
            _proc_reg = _mm_castps_si128(_mm_load_ps(src));
        }
        else {
            _proc_reg = _mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(aux_reg), 0b00111001));
        }
        if (reg_WL != 0) {
            aux_reg = _mm_castps_si128(_mm_load_ps(src + i * Wsrc + (_loop << 2) + 4));

            _BILATERAL_UINT8_V16_CALC_;

            for (int j = 0; j < reg_WL - 1; ++j) {
                _proc_reg = _mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(_proc_reg), 0b00111001));
                aux_reg = _mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(aux_reg), 0b00111001));
                _proc_reg = _mm_castps_si128(_mm_blend_ps(_mm_castsi128_ps(_proc_reg), _mm_castsi128_ps(aux_reg), 0b1000));

                _BILATERAL_UINT8_V16_CALC_;
            }
        }
    }

    _accumulator._v1 = _mm256_div_ps(_accumulator._v1, _accu_W._v1);
    _accumulator._v2 = _mm256_div_ps(_accumulator._v2, _accu_W._v2);

    return _accumulator;
}




_THREAD_CALL_ void 
decx::vis::CPUK::_bilateral_rect_fixed_uint8_ST(const double* __restrict   src,
                                        const float* __restrict       _exp_vals_dist,
                                        const float* __restrict       _exp_chart_diff,
                                        double* __restrict               dst,
                                        const uint2                     ker_dims,
                                        const uint32_t                      Wsrc,
                                        const uint32_t                      Wdst,
                                        const ushort                    reg_WL,
                                        const uint32_t                      _loop)
{
    decx::conv::_v256_2f32 res_vec8;
    uint64_t dex_src = 0, dex_dst = 0;

    __m256i _iv1, _iv2;

    for (int i = 0; i < _BLOCKED_CONV2_UINT8_H_; ++i) {
#pragma unroll _BLOCKED_CONV2_UINT8_W_
        for (int j = 0; j < _BLOCKED_CONV2_UINT8_W_; ++j) {
            res_vec8 = decx::vis::CPUK::_bilateral_uint8_f32_loop_in_kernel(src + dex_src,
                _exp_vals_dist, _exp_chart_diff, ker_dims, reg_WL, Wsrc << 1, _loop);

            _iv1 = _mm256_cvtps_epi32(res_vec8._v1);
            _iv2 = _mm256_cvtps_epi32(res_vec8._v2);
            _iv1 = _mm256_packs_epi32(_iv1, _iv2);
            _iv2 = _mm256_permutevar8x32_epi32(_mm256_packus_epi16(_iv1, _iv1), _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
            
            _mm_store_pd(dst + dex_dst, _mm_castsi128_pd(_mm256_castsi256_si128(_iv2)));

            dex_src += 2;
            dex_dst += 2;
        }
        dex_dst += (Wdst - _BLOCKED_CONV2_UINT8_W_) * 2;
        dex_src += (Wsrc - _BLOCKED_CONV2_UINT8_W_) * 2;
    }
}




_THREAD_CALL_ void 
decx::vis::CPUK::_bilateral_rect_flex_uint8_ST(const double* __restrict     src,
                                               const float* __restrict      _exp_vals_dist,
                                               const float* __restrict      _exp_chart_diff,
                                               double* __restrict           dst,
                                               const uint2                  proc_dim,
                                               const uint2                  ker_dims,
                                               const uint32_t                   Wsrc,
                                               const uint32_t                   Wdst,
                                               const ushort                 reg_WL,
                                               const uint32_t                   _loop)
{
    decx::conv::_v256_2f32 res_vec8;
    uint64_t dex_src = 0, dex_dst = 0;

    __m256i _iv1, _iv2;

    for (int i = 0; i < proc_dim.y; ++i) {
        for (int j = 0; j < proc_dim.x; ++j) {
            res_vec8 = decx::vis::CPUK::_bilateral_uint8_f32_loop_in_kernel(src + dex_src, 
                _exp_vals_dist, _exp_chart_diff, ker_dims, reg_WL, Wsrc << 1, _loop);
            _iv1 = _mm256_cvtps_epi32(res_vec8._v1);
            _iv2 = _mm256_cvtps_epi32(res_vec8._v2);
            _iv1 = _mm256_packs_epi32(_iv1, _iv2);
            _iv2 = _mm256_permutevar8x32_epi32(_mm256_packus_epi16(_iv1, _iv1), _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));

            _mm_store_pd(dst + dex_dst, _mm_castsi128_pd(_mm256_castsi256_si128(_iv2)));

            dex_src += 2;
            dex_dst += 2;
        }
        dex_dst += (Wdst - proc_dim.x) * 2;
        dex_src += (Wsrc - proc_dim.x) * 2;
    }
}





_THREAD_CALL_ void 
decx::vis::CPUK::_bilateral_rect_fixed_uchar4_ST(const float* __restrict   src,
                                        const float* __restrict       _exp_vals_dist,
                                        const float* __restrict       _exp_chart_diff,
                                        float* __restrict               dst,
                                        const uint2                     ker_dims,
                                        const uint32_t                      Wsrc,
                                        const uint32_t                      Wdst,
                                        const ushort                    reg_WL,
                                        const uint32_t                      _loop)
{
    decx::conv::_v256_2f32 res_vec8;
    uint64_t dex_src = 0, dex_dst = 0;

    __m256i _iv1, _iv2;

    for (int i = 0; i < _BLOCKED_CONV2_UINT8_H_; ++i) {
#pragma unroll _BLOCKED_CONV2_UINT8_W_
        for (int j = 0; j < _BLOCKED_CONV2_UINT8_W_; ++j) {
            res_vec8 = decx::vis::CPUK::_bilateral_uchar4_f32_loop_in_kernel(src + dex_src,
                _exp_vals_dist, _exp_chart_diff, ker_dims, reg_WL, Wsrc, _loop);

            _iv1 = _mm256_cvtps_epi32(res_vec8._v1);
            _iv2 = _mm256_cvtps_epi32(res_vec8._v2);
            _iv1 = _mm256_packs_epi32(_iv1, _iv2);
            _iv2 = _mm256_permutevar8x32_epi32(_mm256_packus_epi16(_iv1, _iv1), _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));

            _mm_store_ps(dst + dex_dst, _mm_castsi128_ps(_mm256_castsi256_si128(_iv2)));

            dex_src += 4;
            dex_dst += 4;
        }
        dex_dst += (Wdst - _BLOCKED_CONV2_UINT8_W_ * 4);
        dex_src += (Wsrc - _BLOCKED_CONV2_UINT8_W_ * 4);
    }
}






_THREAD_CALL_ void 
decx::vis::CPUK::_bilateral_rect_flex_uchar4_ST(const float* __restrict     src,
                                               const float* __restrict      _exp_vals_dist,
                                               const float* __restrict      _exp_chart_diff,
                                               float* __restrict           dst,
                                               const uint2                  proc_dim,
                                               const uint2                  ker_dims,
                                               const uint32_t                   Wsrc,
                                               const uint32_t                   Wdst,
                                               const ushort                 reg_WL,
                                               const uint32_t                   _loop)
{
    decx::conv::_v256_2f32 res_vec8;
    uint64_t dex_src = 0, dex_dst = 0;

    __m256i _iv1, _iv2;

    for (int i = 0; i < proc_dim.y; ++i) {
        for (int j = 0; j < proc_dim.x; ++j) {
            res_vec8 = decx::vis::CPUK::_bilateral_uchar4_f32_loop_in_kernel(src + dex_src, 
                _exp_vals_dist, _exp_chart_diff, ker_dims, reg_WL, Wsrc, _loop);
            _iv1 = _mm256_cvtps_epi32(res_vec8._v1);
            _iv2 = _mm256_cvtps_epi32(res_vec8._v2);
            _iv1 = _mm256_packs_epi32(_iv1, _iv2);
            _iv2 = _mm256_permutevar8x32_epi32(_mm256_packus_epi16(_iv1, _iv1), _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));

            _mm_store_ps(dst + dex_dst, _mm_castsi128_ps(_mm256_castsi256_si128(_iv2)));

            dex_src += 4;
            dex_dst += 4;
        }
        dex_dst += (Wdst - proc_dim.x * 4);
        dex_src += (Wsrc - proc_dim.x * 4);
    }
}




_THREAD_FUNCTION_
void decx::vis::_bilateral_uint8_ST(const double* __restrict     src,
                             const float* _exp_chart_dist,
                             const float* _exp_chart_diff,
                             double* __restrict     dst, 
                             const uint2           proc_dim, 
                             const uint2           ker_dims,
                             const uint32_t            Wsrc,
                             const uint32_t            Wdst,
                             const ushort          reg_WL,
                             const uint32_t            _loop)
{
    __m256 res_vec8;
    uint64_t dex_src = 0, dex_dst = 0;

    decx::utils::frag_manager f_mgrH, f_mgrW;
    decx::utils::frag_manager_gen_from_fragLen(&f_mgrH, proc_dim.y, _BLOCKED_CONV2_UINT8_H_);
    decx::utils::frag_manager_gen_from_fragLen(&f_mgrW, proc_dim.x, _BLOCKED_CONV2_UINT8_W_);

    const uint32_t _loopH = f_mgrH.is_left ? f_mgrH.frag_num - 1 : f_mgrH.frag_num;
    
    for (int i = 0; i < _loopH; ++i) 
    {
        const uint32_t _loopW = f_mgrW.is_left ? f_mgrW.frag_num - 1 : f_mgrW.frag_num;

        for (int k = 0; k < _loopW; ++k) {
            decx::vis::CPUK::_bilateral_rect_fixed_uint8_ST(
                DECX_PTR_SHF_XY_SAME_TYPE(src, i * _BLOCKED_CONV2_UINT8_H_, k * 2 * _BLOCKED_CONV2_UINT8_W_, Wsrc * 2),
                _exp_chart_dist, _exp_chart_diff,
                DECX_PTR_SHF_XY_SAME_TYPE(dst, i * _BLOCKED_CONV2_UINT8_H_, k * 2 * _BLOCKED_CONV2_UINT8_W_, Wdst * 2),
                ker_dims, Wsrc, Wdst, reg_WL, _loop);
        }
        if (f_mgrW.is_left) {
            const uint32_t _sum_prev_lenW = proc_dim.x - f_mgrW.frag_left_over;
            decx::vis::CPUK::_bilateral_rect_flex_uint8_ST(
                DECX_PTR_SHF_XY_SAME_TYPE(src, i * _BLOCKED_CONV2_UINT8_H_, _sum_prev_lenW * 2, Wsrc * 2),
                _exp_chart_dist, _exp_chart_diff,
                DECX_PTR_SHF_XY_SAME_TYPE(dst, i * _BLOCKED_CONV2_UINT8_H_, _sum_prev_lenW * 2, Wdst * 2),
                make_uint2(f_mgrW.frag_left_over, _BLOCKED_CONV2_UINT8_H_), ker_dims,
                Wsrc, Wdst, reg_WL, _loop);
        }
    }
    
    if (f_mgrH.is_left)
    {
        const uint32_t _sum_prev_lenH = proc_dim.y - f_mgrH.frag_left_over;
        const uint32_t _loopW = f_mgrW.is_left ? f_mgrW.frag_num - 1 : f_mgrW.frag_num;

        for (int k = 0; k < _loopW; ++k) {
            decx::vis::CPUK::_bilateral_rect_flex_uint8_ST(
                DECX_PTR_SHF_XY_SAME_TYPE(src, _sum_prev_lenH, k * _BLOCKED_CONV2_UINT8_W_ * 2, Wsrc * 2),
                _exp_chart_dist, _exp_chart_diff,
                DECX_PTR_SHF_XY_SAME_TYPE(dst, _sum_prev_lenH, k * _BLOCKED_CONV2_UINT8_W_ * 2, Wdst * 2),
                make_uint2(_BLOCKED_CONV2_UINT8_W_, f_mgrH.frag_left_over),
                ker_dims, Wsrc, Wdst, reg_WL, _loop);
        }
        if (f_mgrW.is_left) {
            const uint32_t _sum_prev_lenW = proc_dim.x - f_mgrW.frag_left_over;
            decx::vis::CPUK::_bilateral_rect_flex_uint8_ST(
                DECX_PTR_SHF_XY_SAME_TYPE(src, _sum_prev_lenH, _sum_prev_lenW * 2, Wsrc * 2),
                _exp_chart_dist, _exp_chart_diff,
                DECX_PTR_SHF_XY_SAME_TYPE(dst, _sum_prev_lenH, _sum_prev_lenW * 2, Wdst * 2),
                make_uint2(f_mgrW.frag_left_over, f_mgrH.frag_left_over), ker_dims,
                Wsrc, Wdst, reg_WL, _loop);
        }
    }
}





_THREAD_FUNCTION_
void decx::vis::_bilateral_uchar4_ST(const float* __restrict        src,
                                     const float*                   _exp_chart_dist,
                                     const float*                   _exp_chart_diff,
                                     float* __restrict              dst, 
                                     const uint2                    proc_dim, 
                                     const uint2                    ker_dims,
                                     const uint32_t                     Wsrc,
                                     const uint32_t                     Wdst,
                                     const ushort                   reg_WL,
                                     const uint32_t                     _loop)
{
    __m256 res_vec8;
    uint64_t dex_src = 0, dex_dst = 0;

    decx::utils::frag_manager f_mgrH, f_mgrW;
    decx::utils::frag_manager_gen_from_fragLen(&f_mgrH, proc_dim.y, _BLOCKED_CONV2_UINT8_H_);
    decx::utils::frag_manager_gen_from_fragLen(&f_mgrW, proc_dim.x, _BLOCKED_CONV2_UINT8_W_);

    const uint32_t _loopH = f_mgrH.is_left ? f_mgrH.frag_num - 1 : f_mgrH.frag_num;
    
    for (int i = 0; i < _loopH; ++i) 
    {
        const uint32_t _loopW = f_mgrW.is_left ? f_mgrW.frag_num - 1 : f_mgrW.frag_num;

        for (int k = 0; k < _loopW; ++k) {
            decx::vis::CPUK::_bilateral_rect_fixed_uchar4_ST(
                DECX_PTR_SHF_XY_SAME_TYPE(src, i * _BLOCKED_CONV2_UINT8_H_, k * _BLOCKED_CONV2_UINT8_W_ * 4, Wsrc),
                _exp_chart_dist, _exp_chart_diff,
                DECX_PTR_SHF_XY_SAME_TYPE(dst, i * _BLOCKED_CONV2_UINT8_H_, k * _BLOCKED_CONV2_UINT8_W_ * 4, Wdst),
                ker_dims, Wsrc, Wdst, reg_WL, _loop);
        }
        if (f_mgrW.is_left) {
            const uint32_t _sum_prev_lenW = proc_dim.x - f_mgrW.frag_left_over;
            decx::vis::CPUK::_bilateral_rect_flex_uchar4_ST(
                DECX_PTR_SHF_XY_SAME_TYPE(src, i * _BLOCKED_CONV2_UINT8_H_, _sum_prev_lenW * 4, Wsrc),
                _exp_chart_dist, _exp_chart_diff,
                DECX_PTR_SHF_XY_SAME_TYPE(dst, i * _BLOCKED_CONV2_UINT8_H_, _sum_prev_lenW * 4, Wdst),
                make_uint2(f_mgrW.frag_left_over, _BLOCKED_CONV2_UINT8_H_), ker_dims,
                Wsrc, Wdst, reg_WL, _loop);
        }
    }
    
    if (f_mgrH.is_left)
    {
        const uint32_t _sum_prev_lenH = proc_dim.y - f_mgrH.frag_left_over;
        const uint32_t _loopW = f_mgrW.is_left ? f_mgrW.frag_num - 1 : f_mgrW.frag_num;

        for (int k = 0; k < _loopW; ++k) {
            decx::vis::CPUK::_bilateral_rect_flex_uchar4_ST(
                DECX_PTR_SHF_XY_SAME_TYPE(src, _sum_prev_lenH, k * _BLOCKED_CONV2_UINT8_W_ * 4, Wsrc),
                _exp_chart_dist, _exp_chart_diff,
                DECX_PTR_SHF_XY_SAME_TYPE(dst, _sum_prev_lenH, k * _BLOCKED_CONV2_UINT8_W_ * 4, Wdst),
                make_uint2(_BLOCKED_CONV2_UINT8_W_, f_mgrH.frag_left_over),
                ker_dims, Wsrc, Wdst, reg_WL, _loop);
        }
        if (f_mgrW.is_left) {
            const uint32_t _sum_prev_lenW = proc_dim.x - f_mgrW.frag_left_over;
            decx::vis::CPUK::_bilateral_rect_flex_uchar4_ST(
                DECX_PTR_SHF_XY_SAME_TYPE(src, _sum_prev_lenH, _sum_prev_lenW * 4, Wsrc),
                _exp_chart_dist, _exp_chart_diff,
                DECX_PTR_SHF_XY_SAME_TYPE(dst, _sum_prev_lenH, _sum_prev_lenW * 4, Wdst),
                make_uint2(f_mgrW.frag_left_over, f_mgrH.frag_left_over), ker_dims,
                Wsrc, Wdst, reg_WL, _loop);
        }
    }
}






void decx::vis::_bilateral_uint8_caller(const double*               src,
                                        const float*                _exp_chart_dist,
                                        const float*                _exp_chart_diff,
                                        double*                     dst, 
                                        const uint2                 proc_dim, 
                                        const uint2                 neighbor_dims,
                                        const uint32_t                  Wsrc,
                                        const uint32_t                  Wdst,
                                        const ushort                reg_WL,
                                        decx::utils::_thr_1D*       t1D,
                                        decx::utils::frag_manager*  f_mgr,
                                        const uint32_t                  _loop)
{
    const double* tmp_src_ptr = src;
    double *tmp_dst_ptr = dst;
    uint64_t frag_src = (uint64_t)f_mgr->frag_len * (uint64_t)Wsrc * 2;
    uint64_t frag_dst = (uint64_t)f_mgr->frag_len * (uint64_t)Wdst * 2;

    for (int i = 0; i < t1D->total_thread - 1; ++i) {
        t1D->_async_thread[i] = decx::cpu::register_task_default(decx::vis::_bilateral_uint8_ST,
            tmp_src_ptr, _exp_chart_dist, _exp_chart_diff, tmp_dst_ptr,
            make_uint2(proc_dim.x, f_mgr->frag_len), neighbor_dims, Wsrc, Wdst, reg_WL, _loop);

        tmp_src_ptr += frag_src;
        tmp_dst_ptr += frag_dst;
    }
    const uint32_t _L = f_mgr->is_left ? f_mgr->frag_left_over : f_mgr->frag_len;
    t1D->_async_thread[t1D->total_thread - 1] = decx::cpu::register_task_default(decx::vis::_bilateral_uint8_ST,
        tmp_src_ptr, _exp_chart_dist, _exp_chart_diff, tmp_dst_ptr,
        make_uint2(proc_dim.x, _L), neighbor_dims, Wsrc, Wdst, reg_WL, _loop);

    t1D->__sync_all_threads();
}





void decx::vis::_bilateral_uchar4_caller(const float*               src,
                                        const float*                _exp_chart_dist,
                                        const float*                _exp_chart_diff,
                                        float*                      dst, 
                                        const uint2                 proc_dim, 
                                        const uint2                 neighbor_dims,
                                        const uint32_t                  Wsrc,
                                        const uint32_t                  Wdst,
                                        const ushort                reg_WL,
                                        decx::utils::_thr_1D*       t1D,
                                        decx::utils::frag_manager*  f_mgr,
                                        const uint32_t                  _loop)
{
    const float* tmp_src_ptr = src;
    float *tmp_dst_ptr = dst;
    uint64_t frag_src = (uint64_t)f_mgr->frag_len * (uint64_t)Wsrc;
    uint64_t frag_dst = (uint64_t)f_mgr->frag_len * (uint64_t)Wdst;

    for (int i = 0; i < t1D->total_thread - 1; ++i) {
        t1D->_async_thread[i] = decx::cpu::register_task_default(decx::vis::_bilateral_uchar4_ST,
            tmp_src_ptr, _exp_chart_dist, _exp_chart_diff, tmp_dst_ptr,
            make_uint2(proc_dim.x, f_mgr->frag_len), neighbor_dims, Wsrc, Wdst, reg_WL, _loop);

        tmp_src_ptr += frag_src;
        tmp_dst_ptr += frag_dst;
    }
    const uint32_t _L = f_mgr->is_left ? f_mgr->frag_left_over : f_mgr->frag_len;
    t1D->_async_thread[t1D->total_thread - 1] = decx::cpu::register_task_default(decx::vis::_bilateral_uchar4_ST,
        tmp_src_ptr, _exp_chart_dist, _exp_chart_diff, tmp_dst_ptr,
        make_uint2(proc_dim.x, _L), neighbor_dims, Wsrc, Wdst, reg_WL, _loop);

    t1D->__sync_all_threads();
}