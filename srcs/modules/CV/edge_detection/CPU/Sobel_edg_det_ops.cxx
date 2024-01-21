/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#include "edge_det_ops.h"



_THREAD_FUNCTION_ void
decx::vis::CPUK::Sobel_XY_uint8(const uint8_t* __restrict  src, 
                                 float* __restrict          G, 
                                 float* __restrict          dir, 
                                 const uint32_t             WG,           // in vec1
                                 const uint32_t             WD,           // in vec1
                                 const uint32_t             Wsrc,           // in vec1
                                 const uint2                proc_dims)
{
    size_t dex_G = 0, dex_src = 0, dex_D;

    decx::utils::simd::xmm128_reg recv, reg1, reg2, reg3, accuY, accuX;
    decx::utils::simd::xmm256_reg Gv8_fp32, RADv8_fp32, tmp, tmp1;
    const __m128i _shift_cont = _mm_setr_epi8(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0);

    for (int i = 0; i < proc_dims.y; ++i) 
    {
        dex_G = i * WG;
        dex_D = i * WD;
        dex_src = i * Wsrc;
        for (int j = 0; j < proc_dims.x; ++j) {
            accuY._vi = _mm_set1_epi16(0);      accuX._vi = _mm_set1_epi16(0);

            for (int k = 0; k < 3; ++k) 
            {
                recv._vf = _mm_loadu_ps((float*)(src + dex_src + ((Wsrc) * k)));

                reg1._vi = _mm_cvtepu8_epi16(recv._vi);
                reg2._vi = _mm_shuffle_epi8(recv._vi, _shift_cont);
                reg3._vi = _mm_cvtepu8_epi16(_mm_shuffle_epi8(reg2._vi, _shift_cont));
                reg2._vi = _mm_cvtepu8_epi16(reg2._vi);
                reg2._vi = _mm_add_epi16(_mm_slli_epi16(reg2._vi, 1), _mm_add_epi16(reg1._vi, reg3._vi));

                switch (k) {
                case 0:
                    accuY._vi = _mm_sub_epi16(accuY._vi, reg2._vi);
                    accuX._vi = _mm_add_epi16(_mm_sub_epi16(reg3._vi, reg1._vi), accuX._vi);
                    break;
                case 1:
                    accuX._vi = _mm_add_epi16(_mm_slli_epi16(_mm_sub_epi16(reg3._vi, reg1._vi), 1), accuX._vi);
                    break;
                case 2:
                    accuY._vi = _mm_add_epi16(accuY._vi, reg2._vi);
                    accuX._vi = _mm_add_epi16(_mm_sub_epi16(reg3._vi, reg1._vi), accuX._vi);
                    break;
                default:
                    break;
                }
            }
            tmp._vf = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(accuY._vi));
            tmp1._vf = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(accuX._vi));
#ifdef _MSC_VER
            RADv8_fp32._vf = _mm256_atan2_ps(tmp._vf, tmp1._vf);
#endif
#ifdef __GNUC__
            RADv8_fp32._vf = decx::utils::simd::_mm256_atan2_ps(tmp._vf, tmp1._vf);
#endif
            Gv8_fp32._vf = _mm256_fmadd_ps(tmp._vf, tmp._vf, _mm256_mul_ps(tmp1._vf, tmp1._vf));

            _mm256_storeu_ps(G + dex_G + 1 + WG, Gv8_fp32._vf);
            _mm256_store_ps(dir + dex_D, RADv8_fp32._vf);

            dex_src += 8;
            dex_G += 8;
            dex_D += 8;
        }
    }
}