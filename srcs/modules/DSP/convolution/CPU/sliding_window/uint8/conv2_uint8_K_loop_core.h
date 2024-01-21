/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _CONV2_UINT8_K_LOOP_CORE_H_
#define _CONV2_UINT8_K_LOOP_CORE_H_


#include "../../../../../core/thread_management/thread_pool.h"
#include "../../../../../core/thread_management/thread_arrange.h"
#include "../../../../../DSP/regional/regional_comparision/CPU/rcp_sliding_window_avx_ops.h"


#define _BLOCKED_CONV2_UINT8_H_ 8
#define _BLOCKED_CONV2_UINT8_W_ 8


#define _CONV2_REGS_UINT8_I32_SHIFT_FMADD16_(_shf) {        \
    _SLIDING_WINDOW_UINT8_SHIFT_(_proc_reg, _shf);    \
    reg1 = _mm256_cvtepu8_epi16(_proc_reg);                                         \
    reg1 = _mm256_mullo_epi16(reg1, _mm256_set1_epi16(kernel[ker_dex]));            \
    reg2 = _mm256_permute4x64_epi64(reg1, 0b01001110);                              \
    _accumulator._v1 = _mm256_add_epi32(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(reg1)), _accumulator._v1);         \
    _accumulator._v2 = _mm256_add_epi32(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(reg2)), _accumulator._v2);         \
    ++ker_dex;      \
}


#define _CONV2_REGS_UINT8_I32_SHIFT_FMADD8_(_shf) {        \
    _SLIDING_WINDOW_UINT8_SHIFT_(_proc_reg, _shf);    \
    reg1 = _mm256_cvtepu8_epi16(_proc_reg);                                         \
    reg1 = _mm256_mullo_epi16(reg1, _mm256_set1_epi16(kernel[ker_dex]));            \
    _accumulator = _mm256_add_epi32(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(reg1)), _accumulator);         \
    ++ker_dex;      \
}



#define _CONV2_REGS_UINT8_F32_SHIFT_FMADD16_(_shf) {        \
    _SLIDING_WINDOW_UINT8_SHIFT_(_proc_reg, _shf);    \
    reg1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_proc_reg));            \
    reg2 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_castpd_si128(_mm_permute_pd(_mm_castsi128_pd(_proc_reg), 0b01))));      \
    _accumulator._v1 = _mm256_fmadd_ps(reg1, _mm256_set1_ps(kernel[ker_dex]), _accumulator._v1);                             \
    _accumulator._v2 = _mm256_fmadd_ps(reg2, _mm256_set1_ps(kernel[ker_dex]), _accumulator._v2);                             \
    ++ker_dex;      \
}


#define _CONV2_REGS_UINT8_F32_SHIFT_FMADD8_(_shf) {        \
    _SLIDING_WINDOW_UINT8_SHIFT_(_proc_reg, _shf);    \
    reg1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_proc_reg));            \
    _accumulator = _mm256_fmadd_ps(reg1, _mm256_set1_ps(kernel[ker_dex]), _accumulator);         \
    ++ker_dex;      \
}



namespace decx
{
    namespace conv {

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
            decx::conv::_v256_2i32 _conv2_uint8_i32_loop_in_kernel_16(const double* src, const uint8_t* kernel, const uint2  ker_dims,
                    const ushort reg_WL, const size_t Wsrc, const uint _loop);


            static _THREAD_CALL_
            __m256i _conv2_uint8_i32_loop_in_kernel_8(const double* src, const uint8_t* kernel, const uint2  ker_dims,
                    const ushort reg_WL, const size_t Wsrc, const uint _loop);


            /*
            * @param Wsrc : width of src matrix, in double (1 double = 8 uint8_t)
            * @param Wdst : width of dst matrix, in float
            */
            static _THREAD_CALL_
            decx::conv::_v256_2f32 _conv2_uint8_f32_loop_in_kernel_16(const double* src, const float* kernel, const uint2  ker_dims,
                    const ushort reg_WL, const size_t Wsrc, const uint _loop);


            static _THREAD_CALL_
            __m256 _conv2_uint8_f32_loop_in_kernel_8(const double* src, const float* kernel, const uint2  ker_dims,
                    const ushort reg_WL, const size_t Wsrc, const uint _loop);
        }
    }
}



_THREAD_CALL_ decx::conv::_v256_2i32
decx::conv::CPUK::_conv2_uint8_i32_loop_in_kernel_16(const double* __restrict     src,
                                              const uint8_t* __restrict     kernel, 
                                              const uint2           ker_dims, 
                                              const ushort          reg_WL, 
                                              const size_t          Wsrc,
                                              const uint            _loop)
{
    uint8_t _store_reg[32];
    register __m128i _proc_reg;
    __m256i reg1, reg2;

    decx::conv::_v256_2i32 _accumulator;
    _accumulator._v1 = _mm256_set1_epi32(0);
    _accumulator._v2 = _mm256_set1_epi32(0);

    uint k_value;      // kernel value
    uint ker_dex = 0;

    for (int i = 0; i < ker_dims.y; ++i) 
    {
        for (uint j = 0; j < _loop; ++j) {
            _mm256_store_pd((double*)_store_reg, _mm256_loadu_pd(src + i * Wsrc + j * 2));
#ifdef _MSC_VER
            _proc_reg = _mm_loadu_epi8(_store_reg);
#endif
#ifdef __GNUC__
            _proc_reg = _mm_castpd_si128(_mm_loadu_pd((double*)_store_reg));
#endif

            reg1 = _mm256_cvtepu8_epi16(_proc_reg);
            reg1 = _mm256_mullo_epi16(reg1, _mm256_set1_epi16(kernel[ker_dex]));
            reg2 = _mm256_permute4x64_epi64(reg1, 0b01001110);      // the lane 1
            _accumulator._v1 = _mm256_add_epi32(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(reg1)), _accumulator._v1);
            _accumulator._v2 = _mm256_add_epi32(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(reg2)), _accumulator._v2);
            ++ker_dex;

            for (int k = 0; k < 15; ++k) {
                _CONV2_REGS_UINT8_I32_SHIFT_FMADD16_(k + 1);
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

            reg1 = _mm256_cvtepu8_epi16(_proc_reg);
            reg1 = _mm256_mullo_epi16(reg1, _mm256_set1_epi16(kernel[ker_dex]));
            reg2 = _mm256_permute4x64_epi64(reg1, 0b01001110);      // the lane 1
            _accumulator._v1 = _mm256_add_epi32(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(reg1)), _accumulator._v1);
            _accumulator._v2 = _mm256_add_epi32(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(reg2)), _accumulator._v2);
            ++ker_dex;

            for (int j = 0; j < reg_WL - 1; ++j) {
                _CONV2_REGS_UINT8_I32_SHIFT_FMADD16_(j + 1);
            }
        }
    }
    return _accumulator;
}


_THREAD_CALL_ __m256i 
decx::conv::CPUK::_conv2_uint8_i32_loop_in_kernel_8(const double* __restrict        src,
                                                    const uint8_t* __restrict       kernel, 
                                                    const uint2                     ker_dims, 
                                                    const ushort                    reg_WL, 
                                                    const size_t                    Wsrc,
                                                    const uint                      _loop)
{
    uint8_t _store_reg[32];
    register __m128i _proc_reg;
    __m256i reg1;

    __m256i _accumulator = _mm256_set1_epi32(0);

    uint k_value;      // kernel value
    uint ker_dex = 0;

    for (int i = 0; i < ker_dims.y; ++i) 
    {
        for (uint j = 0; j < _loop; ++j) {
            _mm256_store_pd((double*)_store_reg, _mm256_loadu_pd(src + i * Wsrc + j * 2));
#ifdef _MSC_VER
            _proc_reg = _mm_loadu_epi8(_store_reg);
#endif
#ifdef __GNUC__
            _proc_reg = _mm_castpd_si128(_mm_loadu_pd((double*)_store_reg));
#endif

            reg1 = _mm256_cvtepu8_epi16(_proc_reg);
            reg1 = _mm256_mullo_epi16(reg1, _mm256_set1_epi16(kernel[ker_dex]));
            _accumulator = _mm256_add_epi32(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(reg1)), _accumulator);
            ++ker_dex;

            for (int k = 0; k < 15; ++k) {
                _CONV2_REGS_UINT8_I32_SHIFT_FMADD8_(k + 1);
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

            reg1 = _mm256_cvtepu8_epi16(_proc_reg);
            reg1 = _mm256_mullo_epi16(reg1, _mm256_set1_epi16(kernel[ker_dex]));
            _accumulator = _mm256_add_epi32(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(reg1)), _accumulator);
            ++ker_dex;

            for (int j = 0; j < reg_WL - 1; ++j) {
                _CONV2_REGS_UINT8_I32_SHIFT_FMADD8_(j + 1);
            }
        }
    }
    return _accumulator;
}




_THREAD_CALL_ decx::conv::_v256_2f32
decx::conv::CPUK::_conv2_uint8_f32_loop_in_kernel_16(const double* __restrict     src,
                                              const float* __restrict     kernel, 
                                              const uint2           ker_dims, 
                                              const ushort          reg_WL, 
                                              const size_t          Wsrc,
                                              const uint            _loop)
{
    uint8_t _store_reg[32];
    register __m128i _proc_reg;
    __m256 reg1, reg2;

    decx::conv::_v256_2f32 _accumulator;
    _accumulator._v1 = _mm256_set1_ps(0);
    _accumulator._v2 = _mm256_set1_ps(0);

    uint k_value;      // kernel value
    uint ker_dex = 0;

    for (int i = 0; i < ker_dims.y; ++i) 
    {
        for (uint j = 0; j < _loop; ++j) {
            _mm256_store_pd((double*)_store_reg, _mm256_loadu_pd(src + i * Wsrc + j * 2));
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
            _mm256_store_pd((double*)_store_reg, _mm256_loadu_pd(src + i * Wsrc + _loop * 2));
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
    }
    return _accumulator;
}



_THREAD_CALL_ __m256 
decx::conv::CPUK::_conv2_uint8_f32_loop_in_kernel_8(const double* __restrict     src,
                                                    const float* __restrict     kernel, 
                                                    const uint2           ker_dims, 
                                                    const ushort          reg_WL, 
                                                    const size_t          Wsrc,
                                                    const uint            _loop)
{
    uint8_t _store_reg[32];
    register __m128i _proc_reg;
    __m256 reg1;

    __m256 _accumulator = _mm256_set1_ps(0);

    uint k_value;      // kernel value
    uint ker_dex = 0;

    for (int i = 0; i < ker_dims.y; ++i) 
    {
        for (uint j = 0; j < _loop; ++j) {
            _mm256_store_pd((double*)_store_reg, _mm256_loadu_pd(src + i * Wsrc + j * 2));
#ifdef _MSC_VER
            _proc_reg = _mm_loadu_epi8(_store_reg);
#endif
#ifdef __GNUC__
            _proc_reg = _mm_castpd_si128(_mm_loadu_pd((double*)_store_reg));
#endif

            reg1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_proc_reg));
            _accumulator = _mm256_fmadd_ps(reg1, _mm256_set1_ps(kernel[ker_dex]), _accumulator);
            ++ker_dex;

            for (int k = 0; k < 15; ++k) {
                _CONV2_REGS_UINT8_F32_SHIFT_FMADD8_(k + 1);
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

            reg1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_proc_reg));
            _accumulator = _mm256_fmadd_ps(reg1, _mm256_set1_ps(kernel[ker_dex]), _accumulator);
            ++ker_dex;

            for (int j = 0; j < reg_WL - 1; ++j) {
                _CONV2_REGS_UINT8_F32_SHIFT_FMADD8_(j + 1);
            }
        }
    }
    return _accumulator;
}



#endif