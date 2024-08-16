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


#ifndef _GEMM_CPLXF_KERNEL_AARCH64_H_
#define _GEMM_CPLXF_KERNEL_AARCH64_H_

#include "../../../../../../common/basic.h"
#include "../../../../../../common/SIMD/arm64/CPU_cpf32_neon.h"
#include "../../../../../../common/SIMD/intrinsics_ops.h"


namespace decx
{
namespace dsp{
    namespace CPUK{
        inline _THREAD_CALL_
        float32x4_t _cp2_mul_cp2_fp32_unshuffled(const float32x4_t __x, const float32x4_t __y)
        {
            decx::utils::simd::xmm128_reg rr_ii;
            rr_ii._vf = vmulq_f32(__x, __y);
            float32x4_t ri_ir = vmulq_f32(__x, vrev64q_f32(__y));

            uint64x2_t _sign_inv = vdupq_n_u64(0x8000000000000000);
            rr_ii._vui = veorq_u32(rr_ii._vui, vreinterpretq_u64_u32(_sign_inv));      // Invert the sign of imaginary parts

            return vpaddq_f32(rr_ii._vf, ri_ir);
            //float32x4_t _res_vec = vpaddq_f32(rr_ii._vf, ri_ir);
            // R1, R2, I1, I2 -> R1, I1, R2
            //return vzip1q_f32(_res_vec, vextq_f32(_res_vec, _res_vec, 2));
        }
    }
}
namespace blas {
    namespace CPUK 
    {
        // [C11 C12]
        template <bool _ABC>
        static _THREAD_CALL_ void GEMM_cplxf_dp_kernel_strassen1x2(const double* __restrict A_line, const double* __restrict B_lane,
            double* __restrict dst, const uint32_t _linear, const bool _first = false, const double* __restrict C = NULL);

        // [C11 C21]
        template <bool _ABC>
        static _THREAD_CALL_ void GEMM_cplxf_dp_kernel_strassen2x1(const double* __restrict A_line, const double* __restrict B_lane,
            double* __restrict dst, const uint32_t _linear, const uint32_t pitchA_v1, const uint32_t pitchdst_v1,
            const bool _first = false, const double* __restrict C = NULL);


        template <bool _ABC>
        static _THREAD_CALL_ void GEMM_cplxf_dp_kernel_strassen1x1(const double* __restrict A_line, const double* __restrict B_lane,
            double* __restrict dst, const uint32_t _linear, const bool _first = false, const double* __restrict C = NULL);


        // [C11 C12; C21 C22]
        template <bool _ABC>
        static _THREAD_CALL_ void GEMM_cplxf_dp_kernel_strassen2x2(const double* __restrict A_line, const double* __restrict B_lane,
            double* __restrict dst, const uint32_t _linear, const uint32_t pitchA_v1, const uint32_t pitchdst_v1,
            const bool _first = false, const double* __restrict C = NULL);

        /**
        * The layout of dst and C should be completely consistant. Normally it will be, by the definition of GEMM.
        */
        template <bool _ABC>
        static _THREAD_FUNCTION_ void GEMM_cplxf_block_kernel(const double* __restrict A, const double* __restrict B,
            double* __restrict dst, const uint2 proc_dims_v8, const decx::utils::frag_manager* fmgrL,
            const uint32_t pitchA_v1, const uint32_t Llen, const uint32_t pitchdst_v1, const double* __restrict C = NULL);


        template <bool _ABC>
        static _THREAD_FUNCTION_ void GEMM_cplxf_kernel(const double* __restrict A, const double* __restrict B,
            double* __restrict dst, const decx::blas::GEMM_blocking_config* config,
            const uint32_t pitchA_v1, const uint32_t Llen, const uint32_t pitchdst_v1, const double* __restrict C = NULL);
    }
}
}


template <bool _ABC>
static _THREAD_CALL_ void decx::blas::CPUK::
GEMM_cplxf_dp_kernel_strassen1x2(const double* __restrict A_line,   const double* __restrict B_lane,
                                double* __restrict dst,             const uint32_t _linear, 
                                const bool _first,                  const double* __restrict C)
{
    uint32_t B_dex = 0;
    decx::utils::simd::xmm256_reg _accu;
    const uint32_t _L_v2 = decx::utils::fast_uint_ceil2<uint32_t>(_linear);

    if (!_first) {
        _accu._vd = vld1q_f64_x2(dst);
    }
    else {
        if constexpr (_ABC) {
            /**
             * Since matrix C is layouted as normal, in Strassen's Algorithm for avx2 (4x cplxf),
             * the data has to be rearranged to the same form (layout) as that in dst (and _accu registers).
             * Use vld2q_f64() to load the elements interleaved, so the shuffle is done.
            */
            _accu._vd = vld2q_f64(C);
            /**
             * During the accumulation, the complex numbers are stored as [Re1, Re2, Im1, Im2]
            */
            _accu._vd.val[0] = decx::utils::simd::vswap_middle_f32(_accu._vd.val[0]);
            _accu._vd.val[1] = decx::utils::simd::vswap_middle_f32(_accu._vd.val[1]);
        }
        else {
            _accu._vui.val[0] = veorq_u32(_accu._vui.val[0], _accu._vui.val[0]);
            _accu._vui.val[1] = veorq_u32(_accu._vui.val[1], _accu._vui.val[1]);
        }
    }

    /**
    * The pitch of matrix A allows access of data where row address is width + 1 
    * if width is not aligned to 4 in de::CPf datatype.
    */
    for (uint32_t i = 0; i < _L_v2; ++i)
    {
        float32x4_t A_row0 = vreinterpretq_f64_f32(vld1q_f64(A_line + i * 2));
        
        float32x4_t A11_v2 = vcombine_f32(vget_low_f32(A_row0), vget_low_f32(A_row0));
        float32x4_t A12_v2 = vcombine_f32(vget_high_f32(A_row0), vget_high_f32(A_row0));

        if ((i == _L_v2 - 1) && (_linear & 1)) {
            A12_v2 = vreinterpretq_u32_f32(veorq_u32(vreinterpretq_f32_u32(A12_v2), vreinterpretq_f32_u32(A12_v2)));
        }

        // __m256 B1 = _mm256_castpd_ps(_mm256_load_pd(B_lane + B_dex));
        // __m256 B2 = _mm256_castpd_ps(_mm256_load_pd(B_lane + B_dex + 12));

        float32x4x4_t B2x2_v = vld1q_f32_x4((float*)(B_lane + B_dex));

        float32x4_t tmp = decx::dsp::CPUK::_cp2_mul_cp2_fp32_unshuffled(vaddq_f32(B2x2_v.val[0], B2x2_v.val[3]), A11_v2);   // M1
        _accu._vf.val[0] = vaddq_f32(_accu._vf.val[0], tmp);        // C11 += M1

        // M2 = 0   nop
        // B1 = _mm256_castpd_ps(_mm256_load_pd(B_lane + B_dex + 8));      // B21
        // M4 = 0   nop

        tmp = decx::dsp::CPUK::_cp2_mul_cp2_fp32_unshuffled(vaddq_f32(B2x2_v.val[2], B2x2_v.val[3]), A12_v2);       // M7                                             // M7
        _accu._vf.val[0] = vaddq_f32(_accu._vf.val[0], tmp);        // C11 += M7

        //B1 = _mm256_castpd_ps(_mm256_load_pd(B_lane + B_dex + 4));      // B12
        tmp = decx::dsp::CPUK::_cp2_mul_cp2_fp32_unshuffled(A11_v2, vsubq_f32(B2x2_v.val[1], B2x2_v.val[3]));       // M3
        _accu._vf.val[1] = vaddq_f32(_accu._vf.val[1], tmp);        // C12 += M3

        tmp = decx::dsp::CPUK::_cp2_mul_cp2_fp32_unshuffled(vaddq_f32(A11_v2, A12_v2), B2x2_v.val[3]);      // M5
        _accu._vf.val[0] = vsubq_f32(_accu._vf.val[0], tmp);        // C11 -= M5
        _accu._vf.val[1] = vaddq_f32(_accu._vf.val[1], tmp);        // C12 += M5

        // M6 != 0, but C22 doen't exist, so nop

        B_dex += 8;
    }
    // Swap back to form of [R1, I1, R2, I2]
    _accu._vd.val[0] = decx::utils::simd::vswap_middle_f32(_accu._vd.val[0]);
    _accu._vd.val[1] = decx::utils::simd::vswap_middle_f32(_accu._vd.val[1]);
    vst2q_f64(dst, _accu._vd);
}



// template <bool _ABC>
// static _THREAD_CALL_ void decx::blas::CPUK::
// GEMM_cplxf_dp_kernel_strassen2x1(const double* __restrict A_line,       const double* __restrict B_lane,
//                                  double* __restrict dst,                const uint32_t _linear,
//                                  const uint32_t pitchA_v1,              const uint32_t pitchdst_v1,            
//                                  const bool _first,                     const double* __restrict C)
// {
//     uint32_t B_dex = 0;
//     decx::utils::simd::xmm128_reg _accu[4];
//     const uint32_t _L_v2 = decx::utils::fast_uint_ceil2<uint32_t>(_linear);

//     if (!_first) {
//         _accu[0]._vd = _mm_load_pd(dst);                 _accu[1]._vd = _mm_load_pd(dst + 2);
//         _accu[2]._vd = _mm_load_pd(dst + pitchdst_v1);   _accu[3]._vd = _mm_load_pd(dst + pitchdst_v1 + 2);
//     }
//     else {
//         if constexpr (_ABC) {
//             /**
//             * Since matrix C is layouted as normal, in Strassen's Algorithm for avx2 (4x cplxf),
//             * the data has to be rearranged to the same form (layout) as that in dst (and _accu registers).
//             */
//             // First row
//             __m256d recv = _mm256_load_pd(C);
//             recv = _mm256_permute4x64_pd(recv, 0b11011000);
//             _accu[0]._vd = _mm256_castpd256_pd128(recv); _accu[1]._vd = _mm256_extractf128_pd(recv, 1);
//             // Second row
//             recv = _mm256_load_pd(C + pitchdst_v1);
//             recv = _mm256_permute4x64_pd(recv, 0b11011000);
//             _accu[2]._vd = _mm256_castpd256_pd128(recv); _accu[3]._vd = _mm256_extractf128_pd(recv, 1);
//         }
//         else {
//             _accu[0]._vd = _mm_setzero_pd();             _accu[1]._vd = _mm_setzero_pd();
//             _accu[2]._vd = _mm_setzero_pd();             _accu[3]._vd = _mm_setzero_pd();
//         }
//     }

//     /**
//     * The pitch of matrix A allows access of data where row address is width + 1 
//     * if width is not aligned to 4 in de::CPf datatype.
//     */
//     for (uint32_t i = 0; i < _L_v2; ++i)
//     {
//         __m128d A_row0 = _mm_load_pd(A_line + i * 2);
//         __m128d A_row1 = _mm_load_pd(A_line + i * 2 + pitchA_v1);

//         __m128 A11 = _mm_castpd_ps(_mm_permute_pd(A_row0, 0b00));
//         __m128 A12 = _mm_castpd_ps(_mm_permute_pd(A_row0, 0b11));
//         __m128 A21 = _mm_castpd_ps(_mm_permute_pd(A_row1, 0b00));
//         __m128 A22 = _mm_castpd_ps(_mm_permute_pd(A_row1, 0b11));

//         if ((i == _L_v2 - 1) && (_linear & 1)) {
//             A12 = _mm_setzero_ps();
//             A22 = _mm_setzero_ps();
//         }

//         __m128 B1 = _mm_castpd_ps(_mm_load_pd(B_lane + B_dex));
//         __m128 B2 = _mm_castpd_ps(_mm_load_pd(B_lane + B_dex + 10));

//         __m128 tmp = decx::dsp::CPUK::_cp2_mul_cp2_fp32(_mm_add_ps(B1, B2),
//                                                         _mm_add_ps(A11, A22));   // M1
//         _accu[0]._vf = _mm_add_ps(_accu[0]._vf, tmp);        // C11 += M1
//         _accu[3]._vf = _mm_add_ps(_accu[3]._vf, tmp);        // C22 += M1

//         tmp = decx::dsp::CPUK::_cp2_mul_cp2_fp32(B1, _mm_add_ps(A21, A22));      // M2
//         _accu[2]._vf = _mm_add_ps(_accu[2]._vf, tmp);        // C21 += M2
//         _accu[3]._vf = _mm_sub_ps(_accu[3]._vf, tmp);        // C22 -= M2

//         B1 = _mm_castpd_ps(_mm_load_pd(B_lane + B_dex + 8));      // B21
//         tmp = decx::dsp::CPUK::_cp2_mul_cp2_fp32(A22, _mm_sub_ps(B1,
//                 _mm_castpd_ps(_mm_load_pd(B_lane + B_dex))));                     // M4
//         _accu[0]._vf = _mm_add_ps(_accu[0]._vf, tmp);        // C21 += M4
//         _accu[2]._vf = _mm_add_ps(_accu[2]._vf, tmp);        // C22 += M4

//         tmp = decx::dsp::CPUK::_cp2_mul_cp2_fp32(_mm_add_ps(B1, B2),
//                 _mm_sub_ps(A12, A22));                                               // M7
//         _accu[0]._vf = _mm_add_ps(_accu[0]._vf, tmp);        // C11 += M7

//         B1 = _mm_castpd_ps(_mm_load_pd(B_lane + B_dex + 2));      // B12
//         tmp = decx::dsp::CPUK::_cp2_mul_cp2_fp32(A11, _mm_sub_ps(B1, B2));       // M3
//         _accu[1]._vf = _mm_add_ps(_accu[1]._vf, tmp);        // C12 += M3
//         _accu[3]._vf = _mm_add_ps(_accu[3]._vf, tmp);        // C22 += M3

//         tmp = decx::dsp::CPUK::_cp2_mul_cp2_fp32(_mm_add_ps(A11, A12), B2);      // M5
//         _accu[0]._vf = _mm_sub_ps(_accu[0]._vf, tmp);        // C11 -= M5
//         _accu[1]._vf = _mm_add_ps(_accu[1]._vf, tmp);        // C12 += M5

//         B2 = _mm_castpd_ps(_mm_load_pd(B_lane + B_dex));          // B11
//         tmp = decx::dsp::CPUK::_cp2_mul_cp2_fp32(_mm_add_ps(B2, B1),
//                 _mm_sub_ps(A21, A11));                                               // M6
//         _accu[3]._vf = _mm_add_ps(_accu[3]._vf, tmp);        // C22 += M6

//         B_dex += 16;
//     }

//     _mm_store_pd(dst, _accu[0]._vd);                 _mm_store_pd(dst + 2, _accu[1]._vd);
//     _mm_store_pd(dst + pitchdst_v1, _accu[2]._vd);   _mm_store_pd(dst + pitchdst_v1 + 2, _accu[3]._vd);
// }



// template <bool _ABC>
// static _THREAD_CALL_ void decx::blas::CPUK::
// GEMM_cplxf_dp_kernel_strassen1x1(const double* __restrict A_line,       const double* __restrict B_lane,
//                                  double* __restrict dst,                const uint32_t _linear,       
//                                  const bool _first,                     const double* __restrict C)
// {
//     uint32_t B_dex = 0;
//     decx::utils::simd::xmm128_reg _accu[2];
//     const uint32_t _L_v2 = decx::utils::fast_uint_ceil2<uint32_t>(_linear);

//     if (!_first) {
//         _accu[0]._vd = _mm_load_pd(dst);                 _accu[1]._vd = _mm_load_pd(dst + 2);
//     }
//     else {
//         if constexpr (_ABC) {
//             /**
//             * Since matrix C is layouted as normal, in Strassen's Algorithm for avx2 (4x cplxf),
//             * the data has to be rearranged to the same form (layout) as that in dst (and _accu registers).
//             */
//             __m256d recv = _mm256_load_pd(C);
//             recv = _mm256_permute4x64_pd(recv, 0b11011000);
//             _accu[0]._vd = _mm256_castpd256_pd128(recv); _accu[1]._vd = _mm256_extractf128_pd(recv, 1);
//         }
//         else {
//             _accu[0]._vd = _mm_setzero_pd();             _accu[1]._vd = _mm_setzero_pd();
//         }
//     }

//     /**
//     * The pitch of matrix A allows access of data where row address is width + 1 
//     * if width is not aligned to 4 in de::CPf datatype.
//     */
//     for (uint32_t i = 0; i < _L_v2; ++i)
//     {
//         __m128d A_row0 = _mm_load_pd(A_line + i * 2);

//         __m128 A11 = _mm_castpd_ps(_mm_permute_pd(A_row0, 0b00));
//         __m128 A12 = _mm_castpd_ps(_mm_permute_pd(A_row0, 0b11));

//         if ((i == _L_v2 - 1) && (_linear & 1)) {
//             A12 = _mm_setzero_ps();
//         }

//         __m128 B1 = _mm_castpd_ps(_mm_load_pd(B_lane + B_dex));
//         __m128 B2 = _mm_castpd_ps(_mm_load_pd(B_lane + B_dex + 10));

//         __m128 tmp = decx::dsp::CPUK::_cp2_mul_cp2_fp32(_mm_add_ps(B1, B2), A11);   // M1
//         _accu[0]._vf = _mm_add_ps(_accu[0]._vf, tmp);        // C11 += M1

//         // M2 = 0   nop
//         B1 = _mm_castpd_ps(_mm_load_pd(B_lane + B_dex + 8));      // B21
//         // M4 = 0   nop

//         tmp = decx::dsp::CPUK::_cp2_mul_cp2_fp32(_mm_add_ps(B1, B2), A12);      // M7
//         _accu[0]._vf = _mm_add_ps(_accu[0]._vf, tmp);        // C11 += M7

//         //B1 = _mm_castpd_ps(_mm_load_pd(B_lane + B_dex + 2));      // B12 = 0
//         tmp = decx::dsp::CPUK::_cp2_mul_cp2_fp32(A11, decx::utils::simd::_mm_signinv_ps(B2)); // M3
//         _accu[1]._vf = _mm_add_ps(_accu[1]._vf, tmp);        // C12 += M3

//         tmp = decx::dsp::CPUK::_cp2_mul_cp2_fp32(_mm_add_ps(A11, A12), B2);      // M5
//         _accu[0]._vf = _mm_sub_ps(_accu[0]._vf, tmp);        // C11 -= M5
//         _accu[1]._vf = _mm_add_ps(_accu[1]._vf, tmp);        // C12 += M5

//         // C22 dosen't exist    nop

//         B_dex += 16;
//     }

//     _mm_store_pd(dst, _accu[0]._vd);                 _mm_store_pd(dst + 2, _accu[1]._vd);
// }



template <bool _ABC>
static _THREAD_CALL_ void decx::blas::CPUK::
GEMM_cplxf_dp_kernel_strassen2x2(const double* __restrict A_line,       const double* __restrict B_lane,
                                 double* __restrict dst,                const uint32_t _linear,
                                 const uint32_t pitchA_v1,              const uint32_t pitchdst_v1,            
                                 const bool _first,                     const double* __restrict  C)
{
    uint32_t B_dex = 0;
    decx::utils::simd::xmm256_reg _accu[2];
    const uint32_t _L_v2 = decx::utils::fast_uint_ceil2<uint32_t>(_linear);

    if (!_first) {
        _accu[0]._vd = vld1q_f64_x2(dst);   // First row
        _accu[1]._vd = vld1q_f64_x2(dst + pitchdst_v1);     // Second row
    }
    else {
        if constexpr (_ABC) {
            /**
            * Since matrix C is layouted as normal, in Strassen's Algorithm for avx2 (4x cplxf),
            * the data has to be rearranged to the same form (layout) as that in dst (and _accu registers).
            */
            // First row
            _accu[0]._vd = vld1q_f64_x2(C);
            _accu[0]._vf.val[0] = decx::utils::simd::vswap_middle_f32(_accu[0]._vf.val[0]);
            _accu[0]._vf.val[1] = decx::utils::simd::vswap_middle_f32(_accu[0]._vf.val[1]);
            
            // Second row
            _accu[1]._vd = vld1q_f64_x2(C + pitchdst_v1);
            _accu[1]._vf.val[0] = decx::utils::simd::vswap_middle_f32(_accu[1]._vf.val[0]);
            _accu[1]._vf.val[1] = decx::utils::simd::vswap_middle_f32(_accu[1]._vf.val[1]);
        }
        else {
            _accu[0]._vmm128[0] = decx::utils::simd::vdupq_n_zeros(_accu[0]._vmm128[0]);
            _accu[0]._vmm128[1] = decx::utils::simd::vdupq_n_zeros(_accu[0]._vmm128[1]);
            _accu[1]._vmm128[0] = decx::utils::simd::vdupq_n_zeros(_accu[1]._vmm128[0]);
            _accu[1]._vmm128[1] = decx::utils::simd::vdupq_n_zeros(_accu[1]._vmm128[1]);
        }
    }

    /**
    * The pitch of matrix A allows access of data where row address is width + 1 
    * if width is not aligned to 4 in de::CPf datatype.
    */
    for (uint32_t i = 0; i < _L_v2; ++i)
    {
        float32x4_t A_row0 = vreinterpretq_f64_f32(vld1q_f64(A_line + i * 2));
        float32x4_t A_row1 = vreinterpretq_f64_f32(vld1q_f64(A_line + pitchA_v1 + i * 2));

        float32x4_t A11 = vcombine_f32(vget_low_f32(A_row0), vget_low_f32(A_row0));
        float32x4_t A12 = vcombine_f32(vget_high_f32(A_row0), vget_high_f32(A_row0));
        float32x4_t A21 = vcombine_f32(vget_low_f32(A_row1), vget_low_f32(A_row1));
        float32x4_t A22 = vcombine_f32(vget_high_f32(A_row1), vget_high_f32(A_row1));

        if ((i == _L_v2 - 1) && (_linear & 1)) {
            A12 = vreinterpretq_u32_f32(veorq_u32(vreinterpretq_f32_u32(A12), vreinterpretq_f32_u32(A12)));
            A22 = vreinterpretq_u32_f32(veorq_u32(vreinterpretq_f32_u32(A22), vreinterpretq_f32_u32(A22)));
        }

        float32x4x4_t B2x2_v2 = vld1q_f32_x4((float*)(B_lane + B_dex));

        float32x4_t tmp = decx::dsp::CPUK::_cp2_mul_cp2_fp32_unshuffled(vaddq_f32(B2x2_v2.val[0], B2x2_v2.val[3]),
                                                        vaddq_f32(A11, A22));   // M1
        _accu[0]._vf.val[0] = vaddq_f32(_accu[0]._vf.val[0], tmp);        // C11 += M1
        _accu[1]._vf.val[1] = vaddq_f32(_accu[1]._vf.val[1], tmp);        // C22 += M1

        tmp = decx::dsp::CPUK::_cp2_mul_cp2_fp32_unshuffled(B2x2_v2.val[0], vaddq_f32(A21, A22));      // M2
        _accu[1]._vf.val[0] = vaddq_f32(_accu[1]._vf.val[0], tmp);        // C21 += M2
        _accu[1]._vf.val[1] = vsubq_f32(_accu[1]._vf.val[1], tmp);        // C22 -= M2

        //B1 = _mm256_castpd_ps(_mm256_load_pd(B_lane + B_dex + 8));      // B21
        tmp = decx::dsp::CPUK::_cp2_mul_cp2_fp32_unshuffled(A22, vsubq_f32(B2x2_v2.val[2], B2x2_v2.val[0])); // M4
        _accu[0]._vf.val[0] = vaddq_f32(_accu[0]._vf.val[0], tmp);        // C21 += M4
        _accu[1]._vf.val[0] = vaddq_f32(_accu[1]._vf.val[0], tmp);        // C22 += M4

        tmp = decx::dsp::CPUK::_cp2_mul_cp2_fp32_unshuffled(vaddq_f32(B2x2_v2.val[2], B2x2_v2.val[3]),
            vsubq_f32(A12, A22)); // M7
        _accu[0]._vf.val[0] = vaddq_f32(_accu[0]._vf.val[0], tmp);        // C11 += M7

        tmp = decx::dsp::CPUK::_cp2_mul_cp2_fp32_unshuffled(A11, vsubq_f32(B2x2_v2.val[1], B2x2_v2.val[3]));       // M3
        _accu[0]._vf.val[1] = vaddq_f32(_accu[0]._vf.val[1], tmp);        // C12 += M3
        _accu[1]._vf.val[1] = vaddq_f32(_accu[1]._vf.val[1], tmp);        // C22 += M3

        tmp = decx::dsp::CPUK::_cp2_mul_cp2_fp32_unshuffled(vaddq_f32(A11, A12), B2x2_v2.val[3]);      // M5
        _accu[0]._vf.val[0] = vsubq_f32(_accu[0]._vf.val[0], tmp);        // C11 -= M5
        _accu[0]._vf.val[1] = vaddq_f32(_accu[0]._vf.val[1], tmp);        // C12 += M5

        tmp = decx::dsp::CPUK::_cp2_mul_cp2_fp32_unshuffled(vaddq_f32(B2x2_v2.val[0], B2x2_v2.val[1]),
            vsubq_f32(A21, A11));                                               // M6
        _accu[1]._vf.val[1] = vaddq_f32(_accu[1]._vf.val[1], tmp);        // C22 += M6

        B_dex += 8;
    }

    // First row
    vst1q_f64_x2(dst, _accu[0]._vd);
    
    // Second row
    vst1q_f64_x2(dst + pitchdst_v1, _accu[1]._vd);
}



template <bool _ABC>
static _THREAD_FUNCTION_ void 
decx::blas::CPUK::GEMM_cplxf_block_kernel(const double* __restrict A,               const double* __restrict B, 
                                         double* __restrict dst,                    const uint2 proc_dims_v2,
                                         const decx::utils::frag_manager* fmgrL,    const uint32_t pitchA_v1, 
                                         const uint32_t Llen,                       const uint32_t pitchdst_v1,
                                         const double* __restrict C)
{
    uint64_t A_dex = 0, B_dex = 0, dst_dex = 0;
    const uint32_t _H_v2 = proc_dims_v2.y / 2;

    for (uint32_t k = 0; k < fmgrL->frag_num; ++k) 
    {
        const uint32_t _L_frag = k == fmgrL->frag_num - 1 ? fmgrL->last_frag_len : fmgrL->frag_len;
        A_dex = fmgrL->frag_len * k;

        for (uint32_t i = 0; i < _H_v2; ++i) 
        {
            B_dex = fmgrL->frag_len * k * 4;
            dst_dex = i * pitchdst_v1 * 2;
            for (uint32_t j = 0; j < proc_dims_v2.x / 2; ++j) {
                decx::blas::CPUK::GEMM_cplxf_dp_kernel_strassen2x2<_ABC>(A + A_dex, B + B_dex,
                        dst + dst_dex, _L_frag, pitchA_v1, pitchdst_v1, k == 0, C + dst_dex);

                B_dex += Llen * 4;
                dst_dex += 4;
            }
            if (proc_dims_v2.x % 2) {
                // strassen2x1
                // decx::blas::CPUK::GEMM_cplxf_dp_kernel_strassen2x1<_ABC>(A + A_dex, B + B_dex,
                //         dst + dst_dex, _L_frag, pitchA_v1, pitchdst_v1, k == 0, C + dst_dex);
            }
            A_dex += pitchA_v1 * 2;
        }
        if (proc_dims_v2.y & 1) {
            B_dex = fmgrL->frag_len * k * 4;
            dst_dex = _H_v2 * pitchdst_v1 * 2;
            for (uint32_t j = 0; j < proc_dims_v2.x / 2; ++j) {
                decx::blas::CPUK::GEMM_cplxf_dp_kernel_strassen1x2<_ABC>(A + A_dex, B + B_dex,
                    dst + dst_dex, _L_frag, k == 0, C + dst_dex);

                B_dex += Llen * 4;
                dst_dex += 4;
            }
            if (proc_dims_v2.x % 2) {
                // strassen1x1
                // decx::blas::CPUK::GEMM_cplxf_dp_kernel_strassen1x1<_ABC>(A + A_dex, B + B_dex,
                //     dst + dst_dex, _L_frag, k == 0, C + dst_dex);
            }
        }
    }

    for (uint32_t i = 0; i < proc_dims_v2.y; ++i) {
        dst_dex = i * pitchdst_v1;
        for (uint32_t j = 0; j < proc_dims_v2.x / 2; ++j) {
            decx::utils::simd::xmm256_reg _reg;
            _reg._vd = vld1q_f64_x2(dst + dst_dex);
            _reg._vf.val[0] = decx::utils::simd::vswap_middle_f32(_reg._vf.val[0]);
            _reg._vf.val[1] = decx::utils::simd::vswap_middle_f32(_reg._vf.val[1]);

            vst1q_f64_x2(dst + dst_dex, _reg._vd);
            
            dst_dex += 4;
        }
        if (proc_dims_v2.x & 1) {
            decx::utils::simd::xmm128_reg _reg;
            _reg._vd = vld1q_f64(dst + dst_dex);
            _reg._vf = decx::utils::simd::vswap_middle_f32(_reg._vf);
            vst1q_f64(dst + dst_dex, _reg._vd);
        }
    }
}



template <bool _ABC>
static _THREAD_FUNCTION_ void 
decx::blas::CPUK::GEMM_cplxf_kernel(const double* __restrict A,     const double* __restrict B,
                                    double* __restrict dst,         const decx::blas::GEMM_blocking_config* config,
                                    const uint32_t pitchA_v1,       const uint32_t Llen, 
                                    const uint32_t pitchdst_v1,     const double* __restrict C)
{
    uint64_t A_dex = 0, B_dex = 0, dst_dex = 0;

    for (uint32_t i = 0; i < config->_fmgr_W.frag_num; ++i) 
    {
        B_dex = i * config->_fmgr_W.frag_len * Llen * 2;
        A_dex = 0;
        dst_dex = i * config->_fmgr_W.frag_len * 2;

        uint2 proc_dims = make_uint2(i < config->_fmgr_W.frag_num - 1 ? 
                                     config->_fmgr_W.frag_len : config->_fmgr_W.last_frag_len,
                                     config->_fmgr_H.frag_len);

        for (uint32_t j = 0; j < config->_fmgr_H.frag_num - 1; ++j) 
        {
            decx::blas::CPUK::GEMM_cplxf_block_kernel<_ABC>(A + A_dex, B + B_dex, dst + dst_dex,
                        proc_dims, &config->_fmgr_L, pitchA_v1, Llen, pitchdst_v1, C + dst_dex);

            A_dex += config->_fmgr_H.frag_len * pitchA_v1;
            dst_dex += config->_fmgr_H.frag_len * pitchdst_v1;
        }

        proc_dims.y = config->_fmgr_H.last_frag_len;
        decx::blas::CPUK::GEMM_cplxf_block_kernel<_ABC>(A + A_dex, B + B_dex, dst + dst_dex,
                    proc_dims, &config->_fmgr_L, pitchA_v1, Llen, pitchdst_v1, C + dst_dex);
    }
}


#endif
