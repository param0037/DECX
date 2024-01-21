/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#include "extend_reflect_exec.h"


namespace decx
{
    namespace bp {
        namespace CPUK 
        {
            _THREAD_CALL_ inline void
                _rev_load_reflection_region_b32_v8(const float* __restrict src, float* __restrict dst, const uint32_t reflect_depth_v8)
            {
                __m256 _reg;
                const __m256i _rev_var = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);
                for (uint32_t i = 0; i < reflect_depth_v8; ++i) {
                    _reg = _mm256_load_ps(src + (reflect_depth_v8 - i - 1) * 8);
                    _mm256_store_ps(dst + i * 8, _mm256_permutevar8x32_ps(_reg, _rev_var));
                }
            }



            _THREAD_CALL_ inline void
                _rev_load_reflection_region_b64_v4(const double* __restrict src, double* __restrict dst, const uint32_t reflect_depth_v8)
            {
                decx::utils::simd::xmm256_reg _reg;
                const __m256i _rev_var = _mm256_setr_epi32(6, 7, 4, 5, 2, 3, 0, 1);
                for (uint32_t i = 0; i < reflect_depth_v8; ++i) {
                    _reg._vd = _mm256_load_pd(src + (reflect_depth_v8 - i - 1) * 4);
                    _mm256_store_pd(dst + i * 4, _mm256_castps_pd(_mm256_permutevar8x32_ps(_reg._vf, _rev_var)));
                }
            }


            _THREAD_CALL_ inline void
                _rev_load_reflection_region_b8_v16(const uint8_t* __restrict src, uint8_t* __restrict dst, const uint32_t reflect_depth_v16)
            {
                __m128 _reg;
                const __m128i _rev_var = _mm_setr_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
                for (uint32_t i = 0; i < reflect_depth_v16; ++i) {
                    _reg = _mm_load_ps((float*)(src + (reflect_depth_v16 - i - 1) * 16));
                    _mm_store_ps((float*)(dst + i * 16), _mm_castsi128_ps(_mm_shuffle_epi8(_mm_castps_si128(_reg), _rev_var)/*_mm_set1_epi8(37)*/));
                }
            }


            _THREAD_CALL_ inline void
                _rev_load_reflection_region_b16_v8(const uint16_t* __restrict src, uint16_t* __restrict dst, const uint32_t reflect_depth_v16)
            {
                __m128 _reg;
                const __m128i _rev_var = _mm_setr_epi8(14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1);
                for (uint32_t i = 0; i < reflect_depth_v16; ++i) {
                    _reg = _mm_load_ps((float*)(src + (reflect_depth_v16 - i - 1) * 8));
                    _mm_store_ps((float*)(dst + i * 8), _mm_castsi128_ps(_mm_shuffle_epi8(_mm_castps_si128(_reg), _rev_var)/*_mm_set1_epi8(37)*/));
                }
            }


            _THREAD_CALL_ inline void
                _store_buf_to_dst_b32_v8(const float* __restrict src, float* __restrict dst, const uint32_t store_num_v8)
            {
                for (uint32_t i = 0; i < store_num_v8; ++i) {
                    _mm256_store_ps(dst + i * 8, _mm256_loadu_ps(src + i * 8));
                }
            }


            _THREAD_CALL_ inline void
                _store_buf_to_dst_b64_v4(const double* __restrict src, double* __restrict dst, const uint32_t store_num_v8)
            {
                for (uint32_t i = 0; i < store_num_v8; ++i) {
                    _mm256_store_pd(dst + i * 4, _mm256_loadu_pd(src + i * 4));
                }
            }



            _THREAD_CALL_ inline void
                _store_buf_to_dst_b8_v16(const uint8_t* __restrict src, uint8_t* __restrict dst, const uint32_t store_num_v16)
            {
                for (uint32_t i = 0; i < store_num_v16; ++i) {
                    _mm_store_ps((float*)(dst + i * 16), _mm_loadu_ps((float*)(src + i * 16)));
                }
            }


            _THREAD_CALL_ inline void
                _store_buf_to_dst_b16_v8(const uint16_t* __restrict src, uint16_t* __restrict dst, const uint32_t store_num_v8)
            {
                for (uint32_t i = 0; i < store_num_v8; ++i) {
                    _mm_store_ps((float*)(dst + i * 8), _mm_loadu_ps((float*)(src + i * 8)));
                }
            }

        }
    }
}




_THREAD_FUNCTION_ void
decx::bp::CPUK::_extend_reflect1D_b32(const float* __restrict   src,
                                      float* __restrict         buffer,
                                      float* __restrict         dst,
                                      const decx::bp::extend_reflect_exec_params* b_rfct,
                                      const size_t              _actual_w_v1,
                                      const size_t              _original_w_v8)
{
    const __m256i f = decx::bp::e_rfct_exep_get_shufflevar_f_b32(b_rfct);
    const __m256i b = decx::bp::e_rfct_exep_get_shufflevar_b_b32(b_rfct);
    const __m256i blend = decx::bp::e_rfct_exep_get_blend_b32(b_rfct);

    __m256 reg0, reg1, store;

    decx::bp::CPUK::_rev_load_reflection_region_b32_v8(src, buffer, b_rfct->_actual_load_num_L / 8);
    decx::bp::CPUK::_store_buf_to_dst_b32_v8(buffer + b_rfct->_L_v8_reflectL, dst, b_rfct->_actual_load_num_L / 8);
    reg1 = _mm256_load_ps(src);
    _mm256_storeu_ps(dst + b_rfct->_left, reg1);

    for (uint32_t i = 1; i < _original_w_v8; ++i) {
        reg0 = reg1;
        reg1 = _mm256_load_ps(src + i * 8);
        store = _mm256_blendv_ps(_mm256_permutevar8x32_ps(reg0, f), _mm256_permutevar8x32_ps(reg1, b), _mm256_castsi256_ps(blend));
        _mm256_store_ps(dst + b_rfct->_actual_load_num_L + i * 8 - 8, store);
    }
    reg0 = reg1;
    reg1 = _mm256_setzero_ps();
    store = _mm256_blendv_ps(_mm256_permutevar8x32_ps(reg0, f), _mm256_permutevar8x32_ps(reg1, b), _mm256_castsi256_ps(blend));
    _mm256_store_ps(dst + b_rfct->_actual_load_num_L + _original_w_v8 * 8 - 8, store);

    decx::bp::CPUK::_rev_load_reflection_region_b32_v8(src + (_original_w_v8) * 8 - b_rfct->_actual_load_num_R, 
                                                       buffer, 
                                                       b_rfct->_actual_load_num_R / 8);
    memcpy(dst + _actual_w_v1 + b_rfct->_left, 
           buffer + b_rfct->_rightmost_0num_src + 1, 
           b_rfct->_right * sizeof(float));
}




_THREAD_FUNCTION_ void
decx::bp::CPUK::_extend_H_reflect2D_b32(const float* __restrict   src,
                                      float* __restrict         buffer,
                                      float* __restrict         dst,
                                      const decx::bp::extend_reflect_exec_params* b_rfct,
                                      const uint32_t            Wsrc,
                                      const uint32_t            Wdst,
                                      const uint32_t            _actual_w_v1,
                                      const uint2               _original_dims_v8)
{
    
    const __m256i f = decx::bp::e_rfct_exep_get_shufflevar_f_b32(b_rfct);
    const __m256i b = decx::bp::e_rfct_exep_get_shufflevar_b_b32(b_rfct);
    const __m256i blend = decx::bp::e_rfct_exep_get_blend_b32(b_rfct);

    __m256 reg0, reg1, store;

    size_t dex_src = 0, dex_dst = 0;

    for (uint32_t i = 0; i < _original_dims_v8.y; ++i) 
    {
        dex_src = i * Wsrc;
        dex_dst = i * Wdst;
        _rev_load_reflection_region_b32_v8(src + dex_src, buffer, b_rfct->_actual_load_num_L / 8);
        _store_buf_to_dst_b32_v8(buffer + b_rfct->_L_v8_reflectL, dst + dex_dst, b_rfct->_actual_load_num_L / 8);
        reg1 = _mm256_load_ps(src + dex_src);
        _mm256_storeu_ps(dst + dex_dst + b_rfct->_left, reg1);

        dex_src += 8;
        dex_dst += b_rfct->_actual_load_num_L;

        for (uint32_t j = 1; j < _original_dims_v8.x; ++j) {
            reg0 = reg1;
            reg1 = _mm256_load_ps(src + dex_src);
            store = _mm256_blendv_ps(_mm256_permutevar8x32_ps(reg0, f), _mm256_permutevar8x32_ps(reg1, b), _mm256_castsi256_ps(blend));
            _mm256_store_ps(dst + dex_dst, store);

            dex_src += 8;
            dex_dst += 8;
        }

        reg0 = reg1;
        reg1 = _mm256_setzero_ps();
        store = _mm256_blendv_ps(_mm256_permutevar8x32_ps(reg0, f), _mm256_permutevar8x32_ps(reg1, b), _mm256_castsi256_ps(blend));
        _mm256_store_ps(dst + dex_dst, store);

        decx::bp::CPUK::_rev_load_reflection_region_b32_v8(src + dex_src - b_rfct->_actual_load_num_R, 
                                                           buffer, 
                                                           b_rfct->_actual_load_num_R / 8);
        memcpy(dst + i * Wdst + b_rfct->_left + _actual_w_v1, 
               buffer + b_rfct->_rightmost_0num_src + 1, 
               b_rfct->_right * sizeof(float));
    }
}




// --------------------------------------------- 8-bit ------------------------------------------------



_THREAD_FUNCTION_ void decx::bp::CPUK::_extend_reflect1D_b8(const uint8_t* src, uint8_t* buffer, uint8_t* dst,
    const decx::bp::extend_reflect_exec_params* b_rfct, const size_t _actual_w_v1, const size_t _original_w_v8)
{
    const __m128i f = decx::bp::e_rfct_exep_get_shufflevar_f_b8(b_rfct);
    const __m128i b = decx::bp::e_rfct_exep_get_shufflevar_b_b8(b_rfct);
    const __m128i blend = decx::bp::e_rfct_exep_get_blend_b8(b_rfct);

    decx::utils::simd::xmm128_reg reg0, reg1, store;

    decx::bp::CPUK::_rev_load_reflection_region_b8_v16(src, buffer, b_rfct->_actual_load_num_L / 16);
    decx::bp::CPUK::_store_buf_to_dst_b8_v16(buffer + b_rfct->_L_v8_reflectL, dst, b_rfct->_actual_load_num_L / 16);
    reg1._vf = _mm_load_ps((float*)src);
    _mm_storeu_ps((float*)(dst + b_rfct->_left), reg1._vf);

    for (uint32_t i = 1; i < _original_w_v8; ++i) {
        reg0 = reg1;
        reg1._vf = _mm_load_ps((float*)(src + i * 16));
        store._vi = _mm_blendv_epi8(_mm_shuffle_epi8(reg0._vi, f), _mm_shuffle_epi8(reg1._vi, b), blend);
        _mm_store_ps((float*)(dst + b_rfct->_actual_load_num_L + i * 16 - 16), store._vf);
    }
    reg0 = reg1;
    reg1._vi = _mm_setzero_si128();
    store._vi = _mm_blendv_epi8(_mm_shuffle_epi8(reg0._vi, f), _mm_shuffle_epi8(reg1._vi, b), blend);
    _mm_store_ps((float*)(dst + b_rfct->_actual_load_num_L + _original_w_v8 * 16 - 16), store._vf);

    decx::bp::CPUK::_rev_load_reflection_region_b8_v16(src + (_original_w_v8) * 16 - b_rfct->_actual_load_num_R, 
                                                       buffer, 
                                                       b_rfct->_actual_load_num_R / 16);
    memcpy(dst + _actual_w_v1 + b_rfct->_left, 
           buffer + b_rfct->_rightmost_0num_src + 1, 
           b_rfct->_right * sizeof(uint8_t));
}






_THREAD_FUNCTION_ void
decx::bp::CPUK::_extend_H_reflect2D_b8(const uint8_t* __restrict   src,
                                      uint8_t* __restrict         buffer,
                                      uint8_t* __restrict         dst,
                                      const decx::bp::extend_reflect_exec_params* b_rfct,
                                      const uint32_t            Wsrc,
                                      const uint32_t            Wdst,
                                      const uint32_t            _actual_w_v1,
                                      const uint2               _original_dims_v16)
{
    
    const __m128i f = decx::bp::e_rfct_exep_get_shufflevar_f_b8(b_rfct);
    const __m128i b = decx::bp::e_rfct_exep_get_shufflevar_b_b8(b_rfct);
    const __m128i blend = decx::bp::e_rfct_exep_get_blend_b8(b_rfct);

    decx::utils::simd::xmm128_reg reg0, reg1, store;

    size_t dex_src = 0, dex_dst = 0;

    for (uint32_t i = 0; i < _original_dims_v16.y; ++i) 
    {
        dex_src = i * Wsrc;
        dex_dst = i * Wdst;
        _rev_load_reflection_region_b8_v16(src + dex_src, buffer, b_rfct->_actual_load_num_L / 16);
        _store_buf_to_dst_b8_v16(buffer + b_rfct->_L_v8_reflectL, dst + dex_dst, b_rfct->_actual_load_num_L / 16);
        reg1._vf = _mm_load_ps((float*)(src + dex_src));
        _mm_storeu_ps((float*)(dst + dex_dst + b_rfct->_left), reg1._vf);

        dex_src += 16;
        dex_dst += b_rfct->_actual_load_num_L;

        for (uint32_t j = 1; j < _original_dims_v16.x; ++j) {
            reg0 = reg1;
            reg1._vf = _mm_load_ps((float*)(src + dex_src));
            store._vi = _mm_blendv_epi8(_mm_shuffle_epi8(reg0._vi, f), _mm_shuffle_epi8(reg1._vi, b), blend);
            _mm_store_ps((float*)(dst + dex_dst), store._vf);

            dex_src += 16;
            dex_dst += 16;
        }

        reg0 = reg1;
        reg1._vi = _mm_setzero_si128();
        store._vi = _mm_blendv_epi8(_mm_shuffle_epi8(reg0._vi, f), _mm_shuffle_epi8(reg1._vi, b), blend);
        _mm_store_ps((float*)(dst + dex_dst), store._vf);

        decx::bp::CPUK::_rev_load_reflection_region_b8_v16(src + dex_src - b_rfct->_actual_load_num_R, 
                                                           buffer, 
                                                           b_rfct->_actual_load_num_R / 16);
        memcpy(dst + i * Wdst + b_rfct->_left + _actual_w_v1, 
               buffer + b_rfct->_rightmost_0num_src + 1, 
               b_rfct->_right * sizeof(uint8_t));
    }
}



_THREAD_FUNCTION_ void 
decx::bp::CPUK::_extend_V_reflect2D_m256(float* __restrict   dst,        // point A
                                        const uint32_t      _top, 
                                        const uint32_t      _bottom,
                                        const uint          Hsrc, 
                                        const uint32_t      Wdst)       // in float
{
    __m256 recv;
    size_t _take = 0, _put = 0;

    for (uint i = 0, _i = _top * 2; i < _top; ++i, --_i) 
    {
        _take = _i * Wdst;
        _put = i * Wdst;
        for (uint j = 0; j < Wdst / 8; ++j) {
            recv = _mm256_load_ps(dst + _take);
            _mm256_store_ps(dst + _put, recv);
            _take += 8;
            _put += 8;
        }
    }

    for (uint i = Hsrc + _top, _i = Hsrc + _top - 2; i < Hsrc + _top + _bottom; ++i, --_i) {
        _take = _i * Wdst;
        _put = i * Wdst;
        for (uint j = 0; j < Wdst / 8; ++j) {
            recv = _mm256_load_ps(dst + _take);
            _mm256_store_ps(dst + _put, recv);
            _take += 8;
            _put += 8;
        }
    }
}



// -------------------------------------------------------- 16 bit ----------------------------------------------------------------



_THREAD_FUNCTION_ void decx::bp::CPUK::_extend_reflect1D_b16(const uint16_t* src, uint16_t* buffer, uint16_t* dst,
    const decx::bp::extend_reflect_exec_params* b_rfct, const size_t _actual_w_v1, const size_t _original_w_v8)
{
    const __m128i f = decx::bp::e_rfct_exep_get_shufflevar_f_b16(b_rfct);
    const __m128i b = decx::bp::e_rfct_exep_get_shufflevar_b_b16(b_rfct);
    const __m128i blend = decx::bp::e_rfct_exep_get_blend_b16(b_rfct);

    decx::utils::simd::xmm128_reg reg0, reg1, store;

    decx::bp::CPUK::_rev_load_reflection_region_b16_v8(src, buffer, b_rfct->_actual_load_num_L / 8);
    decx::bp::CPUK::_store_buf_to_dst_b16_v8(buffer + b_rfct->_L_v8_reflectL, dst, b_rfct->_actual_load_num_L / 8);
    reg1._vf = _mm_load_ps((float*)src);
    _mm_storeu_ps((float*)(dst + b_rfct->_left), reg1._vf);

    for (uint32_t i = 1; i < _original_w_v8; ++i) {
        reg0 = reg1;
        reg1._vf = _mm_load_ps((float*)(src + i * 8));
        store._vi = _mm_blendv_epi8(_mm_shuffle_epi8(reg0._vi, f), _mm_shuffle_epi8(reg1._vi, b), blend);
        _mm_store_ps((float*)(dst + b_rfct->_actual_load_num_L + i * 8 - 8), store._vf);
    }
    reg0 = reg1;
    reg1._vi = _mm_setzero_si128();
    store._vi = _mm_blendv_epi8(_mm_shuffle_epi8(reg0._vi, f), _mm_shuffle_epi8(reg1._vi, b), blend);
    _mm_store_ps((float*)(dst + b_rfct->_actual_load_num_L + _original_w_v8 * 8 - 8), store._vf);

    decx::bp::CPUK::_rev_load_reflection_region_b16_v8(src + (_original_w_v8) * 8 - b_rfct->_actual_load_num_R, 
                                                       buffer, 
                                                       b_rfct->_actual_load_num_R / 8);
    memcpy(dst + _actual_w_v1 + b_rfct->_left, 
           buffer + b_rfct->_rightmost_0num_src + 1, 
           b_rfct->_right * sizeof(uint16_t));
}





_THREAD_FUNCTION_ void
decx::bp::CPUK::_extend_H_reflect2D_b16(const uint16_t* __restrict   src,
                                       uint16_t* __restrict         buffer,
                                       uint16_t* __restrict         dst,
                                       const decx::bp::extend_reflect_exec_params* b_rfct,
                                       const uint32_t            Wsrc,
                                       const uint32_t            Wdst,
                                       const uint32_t            _actual_w_v1,
                                       const uint2               _original_dims_v16)
{
    
    const __m128i f = decx::bp::e_rfct_exep_get_shufflevar_f_b16(b_rfct);
    const __m128i b = decx::bp::e_rfct_exep_get_shufflevar_b_b16(b_rfct);
    const __m128i blend = decx::bp::e_rfct_exep_get_blend_b16(b_rfct);

    decx::utils::simd::xmm128_reg reg0, reg1, store;

    size_t dex_src = 0, dex_dst = 0;

    for (uint32_t i = 0; i < _original_dims_v16.y; ++i) 
    {
        dex_src = i * Wsrc;
        dex_dst = i * Wdst;
        _rev_load_reflection_region_b16_v8(src + dex_src, buffer, b_rfct->_actual_load_num_L / 8);
        _store_buf_to_dst_b16_v8(buffer + b_rfct->_L_v8_reflectL, dst + dex_dst, b_rfct->_actual_load_num_L / 8);
        reg1._vf = _mm_load_ps((float*)(src + dex_src));
        _mm_storeu_ps((float*)(dst + dex_dst + b_rfct->_left), reg1._vf);

        dex_src += 8;
        dex_dst += b_rfct->_actual_load_num_L;

        for (uint32_t j = 1; j < _original_dims_v16.x; ++j) {
            reg0 = reg1;
            reg1._vf = _mm_load_ps((float*)(src + dex_src));
            store._vi = _mm_blendv_epi8(_mm_shuffle_epi8(reg0._vi, f), _mm_shuffle_epi8(reg1._vi, b), blend);
            _mm_store_ps((float*)(dst + dex_dst), store._vf);

            dex_src += 8;
            dex_dst += 8;
        }

        reg0 = reg1;
        reg1._vi = _mm_setzero_si128();
        store._vi = _mm_blendv_epi8(_mm_shuffle_epi8(reg0._vi, f), _mm_shuffle_epi8(reg1._vi, b), blend);
        _mm_store_ps((float*)(dst + dex_dst), store._vf);

        decx::bp::CPUK::_rev_load_reflection_region_b16_v8(src + dex_src - b_rfct->_actual_load_num_R, 
                                                           buffer, 
                                                           b_rfct->_actual_load_num_R / 8);
        memcpy(dst + i * Wdst + b_rfct->_left + _actual_w_v1, 
               buffer + b_rfct->_rightmost_0num_src + 1, 
               b_rfct->_right * sizeof(uint16_t));
    }
}


// --------------------------------------------------------- 64 bit ----------------------------------------------------------------




_THREAD_FUNCTION_ void
decx::bp::CPUK::_extend_reflect1D_b64(const double* __restrict   src,
                                      double* __restrict         buffer,
                                      double* __restrict         dst,
                                      const decx::bp::extend_reflect_exec_params* b_rfct,
                                      const size_t              _actual_w_v1,
                                      const size_t              _original_w_v4)
{
    const __m256i f = decx::bp::e_rfct_exep_get_shufflevar_f_b64(b_rfct);
    const __m256i b = decx::bp::e_rfct_exep_get_shufflevar_b_b64(b_rfct);
    const __m256i blend = decx::bp::e_rfct_exep_get_blend_b64(b_rfct);

    decx::utils::simd::xmm256_reg reg0, reg1, store;

    decx::bp::CPUK::_rev_load_reflection_region_b64_v4(src, buffer, b_rfct->_actual_load_num_L / 4);
    decx::bp::CPUK::_store_buf_to_dst_b64_v4(buffer + b_rfct->_L_v8_reflectL, dst, b_rfct->_actual_load_num_L / 4);
    reg1._vd = _mm256_load_pd(src);
    _mm256_storeu_pd(dst + b_rfct->_left, reg1._vd);

    for (uint32_t i = 1; i < _original_w_v4; ++i) {
        reg0 = reg1;
        reg1._vd = _mm256_load_pd(src + i * 4);
        store._vd = _mm256_blendv_pd(
            _mm256_castps_pd(_mm256_permutevar8x32_ps(reg0._vf, f)), 
            _mm256_castps_pd(_mm256_permutevar8x32_ps(reg1._vf, b)), 
            _mm256_castsi256_pd(blend));
        _mm256_store_pd(dst + b_rfct->_actual_load_num_L + i * 4 - 4, store._vd);
    }
    reg0 = reg1;
    reg1._vd = _mm256_setzero_pd();
    store._vd = _mm256_blendv_pd(
        _mm256_castps_pd(_mm256_permutevar8x32_ps(reg0._vf, f)),
        _mm256_castps_pd(_mm256_permutevar8x32_ps(reg1._vf, b)),
        _mm256_castsi256_pd(blend));
    _mm256_store_pd(dst + b_rfct->_actual_load_num_L + _original_w_v4 * 4 - 4, store._vd);

    decx::bp::CPUK::_rev_load_reflection_region_b64_v4(src + (_original_w_v4) * 4 - b_rfct->_actual_load_num_R, 
                                                       buffer, 
                                                       b_rfct->_actual_load_num_R / 4);
    memcpy(dst + _actual_w_v1 + b_rfct->_left, 
           buffer + b_rfct->_rightmost_0num_src + 1, 
           b_rfct->_right * sizeof(double));
}





_THREAD_FUNCTION_ void
decx::bp::CPUK::_extend_H_reflect2D_b64(const double* __restrict   src,
                                        double* __restrict         buffer,
                                        double* __restrict         dst,
                                        const decx::bp::extend_reflect_exec_params* b_rfct,
                                        const uint32_t            Wsrc,
                                        const uint32_t            Wdst,
                                        const uint32_t            _actual_w_v1,
                                        const uint2               _original_dims_v8)
{
    
    const __m256i f = decx::bp::e_rfct_exep_get_shufflevar_f_b64(b_rfct);
    const __m256i b = decx::bp::e_rfct_exep_get_shufflevar_b_b64(b_rfct);
    const __m256i blend = decx::bp::e_rfct_exep_get_blend_b64(b_rfct);

    decx::utils::simd::xmm256_reg reg0, reg1, store;

    size_t dex_src = 0, dex_dst = 0;

    for (uint32_t i = 0; i < _original_dims_v8.y; ++i) 
    {
        dex_src = i * Wsrc;
        dex_dst = i * Wdst;
        _rev_load_reflection_region_b64_v4(src + dex_src, buffer, b_rfct->_actual_load_num_L / 4);
        _store_buf_to_dst_b64_v4(buffer + b_rfct->_L_v8_reflectL, dst + dex_dst, b_rfct->_actual_load_num_L / 4);
        reg1._vd = _mm256_load_pd(src + dex_src);
        _mm256_storeu_pd(dst + dex_dst + b_rfct->_left, reg1._vd);

        dex_src += 4;
        dex_dst += b_rfct->_actual_load_num_L;

        for (uint32_t j = 1; j < _original_dims_v8.x; ++j) {
            reg0 = reg1;
            reg1._vd = _mm256_load_pd(src + dex_src);
            store._vd = _mm256_blendv_pd(
                _mm256_castps_pd(_mm256_permutevar8x32_ps(reg0._vf, f)),
                _mm256_castps_pd(_mm256_permutevar8x32_ps(reg1._vf, b)),
                _mm256_castsi256_pd(blend));
            _mm256_store_pd(dst + dex_dst, store._vd);

            dex_src += 4;
            dex_dst += 4;
        }

        reg0 = reg1;
        reg1._vd = _mm256_setzero_pd();
        store._vd = _mm256_blendv_pd(
            _mm256_castps_pd(_mm256_permutevar8x32_ps(reg0._vf, f)),
            _mm256_castps_pd(_mm256_permutevar8x32_ps(reg1._vf, b)),
            _mm256_castsi256_pd(blend));
        _mm256_store_pd(dst + dex_dst, store._vd);

        decx::bp::CPUK::_rev_load_reflection_region_b64_v4(src + dex_src - b_rfct->_actual_load_num_R, 
                                                           buffer, 
                                                           b_rfct->_actual_load_num_R / 4);
        memcpy(dst + i * Wdst + b_rfct->_left + _actual_w_v1, 
               buffer + b_rfct->_rightmost_0num_src + 1, 
               b_rfct->_right * sizeof(double));
    }
}