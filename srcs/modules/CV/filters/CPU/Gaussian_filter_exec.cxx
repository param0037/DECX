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


#include "Gaussian_filter_exec.h"


namespace decx
{
    namespace vis {
        namespace CPUK {
            _THREAD_CALL_ decx::conv::_v256_2f32
                _gaussian_uint8_fp32_H_Kloop(const double* src, const float* _gaussian_K_H,
                    const uint32_t Wker, const ushort reg_WL, const uint _loop);


            _THREAD_CALL_ decx::conv::_v256_2f32
                _gaussian_uchar4_fp32_H_Kloop(const float* src, const float* _gaussian_K_H,
                    const uint32_t Wker, const ushort reg_WL, const uint _loop);


            _THREAD_FUNCTION_ void
                _gaussian_V_uint8_fp32(const float* src, const float* _gaussian_K_H, double* dst,
                    const uint32_t Hker, const uint32_t Wsrc, const uint32_t Wdst, const uint2 proc_dims);
        }
    }
}




_THREAD_CALL_ decx::conv::_v256_2f32
decx::vis::CPUK::_gaussian_uint8_fp32_H_Kloop(const double* __restrict       src,
                                              const float* __restrict        kernel,
                                              const uint32_t                 Wker, 
                                              const ushort                   reg_WL,
                                              const uint                     _loop)
{
    uint8_t _store_reg[32];
    register __m128i _proc_reg;
    __m256 reg1, reg2;

    decx::conv::_v256_2f32 _accumulator;
    _accumulator._v1 = _mm256_set1_ps(0);
    _accumulator._v2 = _mm256_set1_ps(0);

    uint k_value;      // kernel value
    uint ker_dex = 0;

    for (uint j = 0; j < _loop; ++j) {
        _mm256_store_pd((double*)_store_reg, _mm256_loadu_pd(src + j * 2));
#ifdef _MSC_VER
        _proc_reg = _mm_loadu_epi8(_store_reg);
#endif
#ifdef __GNUC__
        _proc_reg = _mm_castpd_si128(_mm_loadu_pd((double*)_store_reg));
#endif

        reg1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_proc_reg));
        reg2 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_castpd_si128(_mm_permute_pd(_mm_castsi128_pd(_proc_reg), 0b01))));
        _accumulator._v1 = _mm256_fmadd_ps(reg1, _mm256_set1_ps(kernel[ker_dex]), _accumulator._v1);
        _accumulator._v2 = _mm256_fmadd_ps(reg2, _mm256_set1_ps(kernel[ker_dex]), _accumulator._v2);
        ++ker_dex;

        for (int k = 0; k < 15; ++k) {
            _CONV2_REGS_UINT8_F32_SHIFT_FMADD16_(k + 1);
        }
    }

    if (reg_WL != 0) {
        _mm256_store_pd((double*)_store_reg, _mm256_loadu_pd(src + _loop * 2));
#ifdef _MSC_VER
        _proc_reg = _mm_loadu_epi8(_store_reg);
#endif
#ifdef __GNUC__
        _proc_reg = _mm_castpd_si128(_mm_loadu_pd((double*)_store_reg));
#endif

        reg1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_proc_reg));
        reg2 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_castpd_si128(_mm_permute_pd(_mm_castsi128_pd(_proc_reg), 0b01))));
        _accumulator._v1 = _mm256_fmadd_ps(reg1, _mm256_set1_ps(kernel[ker_dex]), _accumulator._v1);
        _accumulator._v2 = _mm256_fmadd_ps(reg2, _mm256_set1_ps(kernel[ker_dex]), _accumulator._v2);
        ++ker_dex;

        for (int j = 0; j < reg_WL - 1; ++j) {
            _CONV2_REGS_UINT8_F32_SHIFT_FMADD16_(j + 1);
        }
    }
    return _accumulator;
}




_THREAD_CALL_ decx::conv::_v256_2f32
decx::vis::CPUK::_gaussian_uchar4_fp32_H_Kloop(const float* __restrict       src,
                                              const float* __restrict        kernel,
                                              const uint32_t                 Wker, 
                                              const ushort                   reg_WL,
                                              const uint                     _loop)
{
    __m128i _proc_reg, aux_reg;
    __m256 reg1, reg2;

    decx::conv::_v256_2f32 _accumulator;
    _accumulator._v1 = _mm256_set1_ps(0);
    _accumulator._v2 = _mm256_set1_ps(0);

    uint k_value;      // kernel value
    uint ker_dex = 0;

    for (uint j = 0; j < _loop; ++j) {
        if (j == 0) {
            _proc_reg = _mm_castps_si128(_mm_load_ps(src));
        }
        else {
            _proc_reg = _mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(aux_reg), 0b00111001));
        }
        aux_reg = _mm_castps_si128(_mm_load_ps(src + (j << 2) + 4));

        reg1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_proc_reg));
        reg2 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_castpd_si128(_mm_permute_pd(_mm_castsi128_pd(_proc_reg), 0b01))));
        _accumulator._v1 = _mm256_fmadd_ps(reg1, _mm256_set1_ps(kernel[ker_dex]), _accumulator._v1);
        _accumulator._v2 = _mm256_fmadd_ps(reg2, _mm256_set1_ps(kernel[ker_dex]), _accumulator._v2);
        ++ker_dex;

        for (int k = 0; k < 3; ++k) {
            _proc_reg = _mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(_proc_reg), 0b00111001));
            aux_reg = _mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(aux_reg), 0b00111001));
            _proc_reg = _mm_castps_si128(_mm_blend_ps(_mm_castsi128_ps(_proc_reg), _mm_castsi128_ps(aux_reg), 0b1000));

            reg1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_proc_reg));
            reg2 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_castpd_si128(_mm_permute_pd(_mm_castsi128_pd(_proc_reg), 0b01))));
            _accumulator._v1 = _mm256_fmadd_ps(reg1, _mm256_set1_ps(kernel[ker_dex]), _accumulator._v1);
            _accumulator._v2 = _mm256_fmadd_ps(reg2, _mm256_set1_ps(kernel[ker_dex]), _accumulator._v2);
            ++ker_dex;
        }
    }

    if (_loop == 0) {
        _proc_reg = _mm_castps_si128(_mm_load_ps(src));
    }
    else {
        _proc_reg = _mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(aux_reg), 0b00111001));
    }
    if (reg_WL != 0) {
        aux_reg = _mm_castps_si128(_mm_load_ps(src + (_loop << 2) + 4));

        reg1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_proc_reg));
        reg2 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_castpd_si128(_mm_permute_pd(_mm_castsi128_pd(_proc_reg), 0b01))));
        _accumulator._v1 = _mm256_fmadd_ps(reg1, _mm256_set1_ps(kernel[ker_dex]), _accumulator._v1);
        _accumulator._v2 = _mm256_fmadd_ps(reg2, _mm256_set1_ps(kernel[ker_dex]), _accumulator._v2);
        ++ker_dex;

        for (int j = 0; j < reg_WL - 1; ++j) {
            _proc_reg = _mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(_proc_reg), 0b00111001));
            aux_reg = _mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(aux_reg), 0b00111001));
            _proc_reg = _mm_castps_si128(_mm_blend_ps(_mm_castsi128_ps(_proc_reg), _mm_castsi128_ps(aux_reg), 0b1000));

            reg1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_proc_reg));
            reg2 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_castpd_si128(_mm_permute_pd(_mm_castsi128_pd(_proc_reg), 0b01))));
            _accumulator._v1 = _mm256_fmadd_ps(reg1, _mm256_set1_ps(kernel[ker_dex]), _accumulator._v1);
            _accumulator._v2 = _mm256_fmadd_ps(reg2, _mm256_set1_ps(kernel[ker_dex]), _accumulator._v2);
            ++ker_dex;
        }
    }
    return _accumulator;
}



_THREAD_FUNCTION_ void
decx::vis::CPUK::_gaussian_V_uint8_fp32(const float* __restrict src,
                                        const float* __restrict _gaussian_K_H,
                                        double* __restrict dst,
                                        const uint32_t Hker,
                                        const uint32_t Wsrc, 
                                        const uint32_t Wdst, 
                                        const uint2 proc_dims)
{
    size_t dex_src = 0, dex_dst = 0;
    const size_t _shf_dex_src = (size_t)Hker * (size_t)Wsrc - 16;

    decx::conv::_v256_2f32 _accumulator;
    _accumulator._v1 = _mm256_set1_ps(0);
    _accumulator._v2 = _mm256_set1_ps(0);
    decx::utils::simd::xmm256_reg recv;
    decx::utils::simd::xmm256_reg reg1, reg2;

    for (uint32_t i = 0; i < proc_dims.y; ++i) {
        dex_src = i * Wsrc;
        dex_dst = i * Wdst;
        for (uint32_t j = 0; j < proc_dims.x; ++j) 
        {
            _accumulator._v1 = _mm256_set1_ps(0);
            _accumulator._v2 = _mm256_set1_ps(0);
            for (uint32_t k = 0; k < Hker; ++k) {
                recv._vf = _mm256_loadu_ps(src + dex_src);
                _accumulator._v1 = _mm256_fmadd_ps(recv._vf, _mm256_set1_ps(_gaussian_K_H[k]), _accumulator._v1);
                recv._vf = _mm256_loadu_ps(src + dex_src + 8);
                _accumulator._v2 = _mm256_fmadd_ps(recv._vf, _mm256_set1_ps(_gaussian_K_H[k]), _accumulator._v2);
                dex_src += Wsrc;
            }

            reg1._vi = _mm256_cvtps_epi32(_accumulator._v1);
            reg2._vi = _mm256_cvtps_epi32(_accumulator._v2);
            reg1._vi = _mm256_packs_epi32(reg1._vi, reg2._vi);
            reg2._vi = _mm256_permutevar8x32_epi32(_mm256_packus_epi16(reg1._vi, reg1._vi), _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));

            _mm_store_pd(dst + dex_dst, _mm_castsi128_pd(_mm256_castsi256_si128(reg2._vi)));

            dex_dst += 2;
            dex_src -= _shf_dex_src;
        }
    }
}


namespace decx
{
    namespace vis {
        namespace CPUK
        {
            static _THREAD_FUNCTION_ void _gaussian_H_uint8_fp32_ST(const double* src, const float* kernel, float* dst,
                const uint2 proc_dim, const uint32_t Wker, const uint Wsrc, const uint Wdst, const ushort reg_WL, const uint _loop);


            static _THREAD_FUNCTION_ void _gaussian_H_uchar4_fp32_ST(const float* src, const float* kernel, float* dst,
                const uint2 proc_dim, const uint32_t Wker, const uint Wsrc, const uint Wdst, const ushort reg_WL, const uint _loop);
        }
    }
}



static _THREAD_CALL_ void 
decx::vis::CPUK::_gaussian_H_uint8_fp32_ST(const double* src, const float* kernel, float* dst,
    const uint2 proc_dim, const uint32_t Wker, const uint Wsrc, const uint Wdst, const ushort reg_WL, const uint _loop)
{
    decx::conv::_v256_2f32 res_vec8;
    size_t dex_src = 0, dex_dst = 0;

    __m256i _iv1, _iv2;

    for (int i = 0; i < proc_dim.y; ++i) 
    {
        dex_src = i * Wsrc;
        dex_dst = i * Wdst;
        for (int j = 0; j < proc_dim.x; ++j) {
            res_vec8 = decx::vis::CPUK::_gaussian_uint8_fp32_H_Kloop(src + dex_src,
                kernel, Wker, reg_WL, _loop);

            _mm256_store_ps(dst + dex_dst, res_vec8._v1);
            _mm256_store_ps(dst + dex_dst + 8, res_vec8._v2);

            dex_src += 2;
            dex_dst += 16;
        }
    }
}




static _THREAD_CALL_ void 
decx::vis::CPUK::_gaussian_H_uchar4_fp32_ST(const float* src, const float* kernel, float* dst,
    const uint2 proc_dim, const uint32_t Wker, const uint Wsrc, const uint Wdst, const ushort reg_WL, const uint _loop)
{
    decx::conv::_v256_2f32 res_vec8;
    size_t dex_src = 0, dex_dst = 0;

    __m256i _iv1, _iv2;

    for (int i = 0; i < proc_dim.y; ++i) 
    {
        dex_src = i * Wsrc;
        dex_dst = i * Wdst;
        for (int j = 0; j < proc_dim.x; ++j) {
            res_vec8 = decx::vis::CPUK::_gaussian_uchar4_fp32_H_Kloop(src + dex_src,
                kernel, Wker, reg_WL, _loop);

            _mm256_store_ps(dst + dex_dst, res_vec8._v1);
            _mm256_store_ps(dst + dex_dst + 8, res_vec8._v2);

            dex_src += 4;
            dex_dst += 16;
        }
    }
}




void decx::vis::_gaussian_H_uint8_caller(const double* src, 
    const float* kernel, float* dst, const uint2 proc_dim, const uint32_t Wker,
    const uint Wsrc, const uint Wdst, const ushort reg_WL, decx::utils::_thr_1D* t1D,
    const uint _loop)
{
    const double* tmp_src_ptr = src;
    float* tmp_dst_ptr = dst;

    decx::utils::frag_manager f_mgr;
    decx::utils::frag_manager_gen(&f_mgr, proc_dim.y, decx::cpu::_get_permitted_concurrency());

    size_t frag_src = (size_t)f_mgr.frag_len * (size_t)Wsrc;
    size_t frag_dst = (size_t)f_mgr.frag_len * (size_t)Wdst;

    for (int i = 0; i < t1D->total_thread - 1; ++i) {
        t1D->_async_thread[i] = decx::cpu::register_task_default(decx::vis::CPUK::_gaussian_H_uint8_fp32_ST,
            tmp_src_ptr, kernel, tmp_dst_ptr,
            make_uint2(proc_dim.x, f_mgr.frag_len), Wker, Wsrc, Wdst, reg_WL, _loop);

        tmp_src_ptr += frag_src;
        tmp_dst_ptr += frag_dst;
    }
    const uint32_t _L = f_mgr.is_left ? f_mgr.frag_left_over : f_mgr.frag_len;
    t1D->_async_thread[t1D->total_thread - 1] = decx::cpu::register_task_default(decx::vis::CPUK::_gaussian_H_uint8_fp32_ST,
        tmp_src_ptr, kernel, tmp_dst_ptr,
        make_uint2(proc_dim.x, _L), Wker, Wsrc, Wdst, reg_WL, _loop);

    t1D->__sync_all_threads();
}




void decx::vis::_gaussian_H_uchar4_caller(const float* src, 
    const float* kernel, float* dst, const uint2 proc_dim, const uint32_t Wker,
    const uint Wsrc, const uint Wdst, const ushort reg_WL, decx::utils::_thr_1D* t1D,
    const uint _loop)
{
    const float* tmp_src_ptr = src;
    float* tmp_dst_ptr = dst;

    decx::utils::frag_manager f_mgr;
    decx::utils::frag_manager_gen(&f_mgr, proc_dim.y, decx::cpu::_get_permitted_concurrency());

    size_t frag_src = (size_t)f_mgr.frag_len * (size_t)Wsrc;
    size_t frag_dst = (size_t)f_mgr.frag_len * (size_t)Wdst;

    for (int i = 0; i < t1D->total_thread - 1; ++i) {
        t1D->_async_thread[i] = decx::cpu::register_task_default(decx::vis::CPUK::_gaussian_H_uchar4_fp32_ST,
            tmp_src_ptr, kernel, tmp_dst_ptr,
            make_uint2(proc_dim.x, f_mgr.frag_len), Wker, Wsrc, Wdst, reg_WL, _loop);

        tmp_src_ptr += frag_src;
        tmp_dst_ptr += frag_dst;
    }
    const uint32_t _L = f_mgr.is_left ? f_mgr.frag_left_over : f_mgr.frag_len;
    t1D->_async_thread[t1D->total_thread - 1] = decx::cpu::register_task_default(decx::vis::CPUK::_gaussian_H_uchar4_fp32_ST,
        tmp_src_ptr, kernel, tmp_dst_ptr,
        make_uint2(proc_dim.x, _L), Wker, Wsrc, Wdst, reg_WL, _loop);

    t1D->__sync_all_threads();
}



void decx::vis::_gaussian_V_uint8_caller(const float* src, const float* kernel, double* dst, const uint2 proc_dim, const uint32_t Hker,
    const uint Wsrc, const uint Wdst, decx::utils::_thr_1D* t1D)
{
    const float* tmp_src_ptr = src;
    double* tmp_dst_ptr = dst;

    decx::utils::frag_manager f_mgr;
    decx::utils::frag_manager_gen(&f_mgr, proc_dim.y, decx::cpu::_get_permitted_concurrency());

    size_t frag_src = (size_t)f_mgr.frag_len * (size_t)Wsrc;
    size_t frag_dst = (size_t)f_mgr.frag_len * (size_t)Wdst;

    for (int i = 0; i < t1D->total_thread - 1; ++i) {
        t1D->_async_thread[i] = decx::cpu::register_task_default(decx::vis::CPUK::_gaussian_V_uint8_fp32,
            tmp_src_ptr, kernel, tmp_dst_ptr,
            Hker, Wsrc, Wdst, make_uint2(proc_dim.x, f_mgr.frag_len));

        tmp_src_ptr += frag_src;
        tmp_dst_ptr += frag_dst;
    }
    const uint32_t _L = f_mgr.is_left ? f_mgr.frag_left_over : f_mgr.frag_len;
    t1D->_async_thread[t1D->total_thread - 1] = decx::cpu::register_task_default(decx::vis::CPUK::_gaussian_V_uint8_fp32,
        tmp_src_ptr, kernel, tmp_dst_ptr,
        Hker, Wsrc, Wdst, make_uint2(proc_dim.x, _L));

    t1D->__sync_all_threads();
}