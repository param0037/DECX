/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/


#include "edge_det_ops.h"
#include "../../../Vectorial/vector4.h"



_THREAD_FUNCTION_ void
decx::vis::CPUK::Sobel_XY_uint8_T(const uint8_t* __restrict  src, 
                                 float* __restrict          G, 
                                 float* __restrict          dir, 
                                 const uint                 Wsrc,           // in vec1
                                 const uint                 Wdst,           // in vec1
                                 const uint2                proc_dims)
{
    size_t dex_src = 0, dex_dst = 0;

    decx::utils::simd::xmm128_reg recv, reg1, reg2, reg3, accuY, accuX;
    decx::utils::simd::xmm256_reg Gv8_fp32, RADv8_fp32, tmp, tmp1;
    const __m128i _shift_cont = _mm_setr_epi8(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0);

    for (int j = 0; j < proc_dims.x; ++j) {
        accuY._vi = _mm_set1_epi16(0);      accuX._vi = _mm_set1_epi16(0);

        for (int k = 0; k < 2; ++k)
        {
            if (j == 0) {
                recv._vf = _mm_load_ps((float*)(src + dex_src + ((Wsrc) * k)));
                recv._vi = _mm_shuffle_epi8(recv._vi, _shift_cont);
            }
            else {
                recv._vf = _mm_loadu_ps((float*)(src + dex_src + ((Wsrc) * k)));
            }
            reg1._vi = _mm_cvtepu8_epi16(recv._vi);
            reg2._vi = _mm_shuffle_epi8(recv._vi, _shift_cont);
            reg3._vi = _mm_cvtepu8_epi16(_mm_shuffle_epi8(reg2._vi, _shift_cont));
            reg2._vi = _mm_cvtepu8_epi16(reg2._vi);
            reg2._vi = _mm_add_epi16(_mm_slli_epi16(reg2._vi, 1), _mm_add_epi16(reg1._vi, reg3._vi));

            switch (k) {
            case 0:
                accuX._vi = _mm_add_epi16(_mm_slli_epi16(_mm_sub_epi16(reg3._vi, reg1._vi), 1), accuX._vi);
                break;
            case 1:
                accuY._vi = _mm_add_epi16(accuY._vi, reg2._vi);
                accuX._vi = _mm_add_epi16(_mm_slli_epi16(_mm_sub_epi16(reg3._vi, reg1._vi), 1), accuX._vi);
                break;
            default:
                break;
            }
        }
        tmp._vf = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(accuY._vi));
        tmp1._vf = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(accuX._vi));
        RADv8_fp32._vf = _mm256_atan2_ps(tmp1._vf, tmp._vf);

        Gv8_fp32._vf = _mm256_fmadd_ps(tmp._vf, tmp._vf, _mm256_mul_ps(tmp1._vf, tmp1._vf));

        _mm256_store_ps(G + dex_dst, Gv8_fp32._vf);
        _mm256_store_ps(dir + dex_dst, RADv8_fp32._vf);

        dex_dst += 8;
        dex_src += ((j == 0) ? 7 : 8);
    }

    for (int i = 1; i < proc_dims.y; ++i) 
    {
        dex_src = i * Wsrc;
        dex_dst = i * Wdst;
        for (int j = 0; j < proc_dims.x; ++j) {
            accuY._vi = _mm_set1_epi16(0);      accuX._vi = _mm_set1_epi16(0);

            for (int k = 0; k < 3; ++k) 
            {
                if (j == 0) {
                    recv._vf = _mm_load_ps((float*)(src + dex_src + ((Wsrc) * k)));
                    recv._vi = _mm_shuffle_epi8(recv._vi, _shift_cont);
                }
                else {
                    recv._vf = _mm_loadu_ps((float*)(src + dex_src + ((Wsrc) * k)));
                }
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
            RADv8_fp32._vf = _mm256_atan2_ps(tmp1._vf, tmp._vf);
            
            Gv8_fp32._vf = _mm256_fmadd_ps(tmp._vf, tmp._vf, _mm256_mul_ps(tmp1._vf, tmp1._vf));

            _mm256_store_ps(G + dex_dst, Gv8_fp32._vf);
            _mm256_store_ps(dir + dex_dst, RADv8_fp32._vf);

            dex_dst += 8;
            dex_src += ((j == 0) ? 7 : 8);
        }
    }
}



_THREAD_FUNCTION_ void
decx::vis::CPUK::Sobel_XY_uint8(const uint8_t* __restrict  src, 
                                 float* __restrict          G, 
                                 float* __restrict          dir, 
                                 const uint                 Wsrc,           // in vec1
                                 const uint                 Wdst,           // in vec1
                                 const uint2                proc_dims)
{
    size_t dex_src = 0, dex_dst = 0;

    decx::utils::simd::xmm128_reg recv, reg1, reg2, reg3, accuY, accuX;
    decx::utils::simd::xmm256_reg Gv8_fp32, RADv8_fp32, tmp, tmp1;
    const __m128i _shift_cont = _mm_setr_epi8(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0);

    for (int i = 0; i < proc_dims.y; ++i) 
    {
        dex_src = i * Wsrc;
        dex_dst = i * Wdst;
        for (int j = 0; j < proc_dims.x; ++j) {
            accuY._vi = _mm_set1_epi16(0);      accuX._vi = _mm_set1_epi16(0);

            for (int k = 0; k < 3; ++k) 
            {
                if (j == 0) {
                    recv._vf = _mm_load_ps((float*)(src + dex_src + ((Wsrc) * k)));
                    recv._vi = _mm_shuffle_epi8(recv._vi, _shift_cont);
                }
                else {
                    recv._vf = _mm_loadu_ps((float*)(src + dex_src + ((Wsrc) * k)));
                }
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
            RADv8_fp32._vf = _mm256_atan2_ps(tmp1._vf, tmp._vf);
            
            Gv8_fp32._vf = _mm256_fmadd_ps(tmp._vf, tmp._vf, _mm256_mul_ps(tmp1._vf, tmp1._vf));

            _mm256_store_ps(G + dex_dst, Gv8_fp32._vf);
            _mm256_store_ps(dir + dex_dst, RADv8_fp32._vf);

            dex_dst += 8;
            dex_src += ((j == 0) ? 7 : 8);
        }
    }
}



_THREAD_FUNCTION_ void
decx::vis::CPUK::Sobel_XY_uint8_B(const uint8_t* __restrict  src, 
                                 float* __restrict          G, 
                                 float* __restrict          dir, 
                                 const uint                 Wsrc,           // in vec1
                                 const uint                 Wdst,           // in vec1
                                 const uint2                proc_dims)
{
    size_t dex_src = 0, dex_dst = 0;

    decx::utils::simd::xmm128_reg recv, reg1, reg2, reg3, accuY, accuX;
    decx::utils::simd::xmm256_reg Gv8_fp32, RADv8_fp32, tmp, tmp1;
    const __m128i _shift_cont = _mm_setr_epi8(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0);

    for (int i = 0; i < proc_dims.y - 1; ++i) 
    {
        dex_src = i * Wsrc;
        dex_dst = i * Wdst;
        for (int j = 0; j < proc_dims.x; ++j) {
            accuY._vi = _mm_set1_epi16(0);      accuX._vi = _mm_set1_epi16(0);

            for (int k = 0; k < 3; ++k) 
            {
                if (j == 0) {
                    recv._vf = _mm_load_ps((float*)(src + dex_src + ((Wsrc) * k)));
                    recv._vi = _mm_shuffle_epi8(recv._vi, _shift_cont);
                }
                else {
                    recv._vf = _mm_loadu_ps((float*)(src + dex_src + ((Wsrc) * k)));
                }
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
            RADv8_fp32._vf = _mm256_atan2_ps(tmp1._vf, tmp._vf);
            
            Gv8_fp32._vf = _mm256_fmadd_ps(tmp._vf, tmp._vf, _mm256_mul_ps(tmp1._vf, tmp1._vf));

            _mm256_store_ps(G + dex_dst, Gv8_fp32._vf);
            _mm256_store_ps(dir + dex_dst, RADv8_fp32._vf);

            dex_dst += 8;
            dex_src += ((j == 0) ? 7 : 8);
        }
    }

    for (int j = 0; j < proc_dims.x; ++j) {
        accuY._vi = _mm_set1_epi16(0);      accuX._vi = _mm_set1_epi16(0);
        dex_src = (proc_dims.y - 1) * Wsrc;
        dex_dst = (proc_dims.y - 1) * Wdst;
        for (int k = 0; k < 2; ++k)
        {
            if (j == 0) {
                recv._vf = _mm_load_ps((float*)(src + dex_src + ((Wsrc)*k)));
                recv._vi = _mm_shuffle_epi8(recv._vi, _shift_cont);
            }
            else {
                recv._vf = _mm_loadu_ps((float*)(src + dex_src + ((Wsrc)*k)));
            }
            reg1._vi = _mm_cvtepu8_epi16(recv._vi);
            reg2._vi = _mm_shuffle_epi8(recv._vi, _shift_cont);
            reg3._vi = _mm_cvtepu8_epi16(_mm_shuffle_epi8(reg2._vi, _shift_cont));
            reg2._vi = _mm_cvtepu8_epi16(reg2._vi);
            reg2._vi = _mm_add_epi16(_mm_slli_epi16(reg2._vi, 1), _mm_add_epi16(reg1._vi, reg3._vi));

            switch (k) {
            case 0:
                accuY._vi = _mm_sub_epi16(accuY._vi, reg2._vi);
                accuX._vi = _mm_add_epi16(_mm_slli_epi16(_mm_sub_epi16(reg3._vi, reg1._vi), 1), accuX._vi);
                break;
            case 1:
                accuX._vi = _mm_add_epi16(_mm_slli_epi16(_mm_sub_epi16(reg3._vi, reg1._vi), 1), accuX._vi);
                break;
            default:
                break;
            }
        }
        tmp._vf = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(accuY._vi));
        tmp1._vf = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(accuX._vi));
        RADv8_fp32._vf = _mm256_atan2_ps(tmp1._vf, tmp._vf);

        Gv8_fp32._vf = _mm256_fmadd_ps(tmp._vf, tmp._vf, _mm256_mul_ps(tmp1._vf, tmp1._vf));

        _mm256_store_ps(G + dex_dst, Gv8_fp32._vf);
        _mm256_store_ps(dir + dex_dst, RADv8_fp32._vf);

        dex_dst += 8;
        dex_src += ((j == 0) ? 7 : 8);
    }
}



_THREAD_FUNCTION_ void
decx::vis::CPUK::Sobel_XY_uint8_TB(const uint8_t* __restrict  src, 
                                 float* __restrict          G, 
                                 float* __restrict          dir, 
                                 const uint                 Wsrc,           // in vec1
                                 const uint                 Wdst,           // in vec1
                                 const uint2                proc_dims)
{
    size_t dex_src = 0, dex_dst = 0;

    decx::utils::simd::xmm128_reg recv, reg1, reg2, reg3, accuY, accuX;
    decx::utils::simd::xmm256_reg Gv8_fp32, RADv8_fp32, tmp, tmp1;
    const __m128i _shift_cont = _mm_setr_epi8(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0);

    for (int j = 0; j < proc_dims.x; ++j) {
        accuY._vi = _mm_set1_epi16(0);      accuX._vi = _mm_set1_epi16(0);

        for (int k = 0; k < 2; ++k)
        {
            if (j == 0) {
                recv._vf = _mm_load_ps((float*)(src + dex_src + ((Wsrc)*k)));
                recv._vi = _mm_shuffle_epi8(recv._vi, _shift_cont);
            }
            else {
                recv._vf = _mm_loadu_ps((float*)(src + dex_src + ((Wsrc)*k)));
            }
            reg1._vi = _mm_cvtepu8_epi16(recv._vi);
            reg2._vi = _mm_shuffle_epi8(recv._vi, _shift_cont);
            reg3._vi = _mm_cvtepu8_epi16(_mm_shuffle_epi8(reg2._vi, _shift_cont));
            reg2._vi = _mm_cvtepu8_epi16(reg2._vi);
            reg2._vi = _mm_add_epi16(_mm_slli_epi16(reg2._vi, 1), _mm_add_epi16(reg1._vi, reg3._vi));

            switch (k) {
            case 0:
                accuX._vi = _mm_add_epi16(_mm_slli_epi16(_mm_sub_epi16(reg3._vi, reg1._vi), 1), accuX._vi);
                break;
            case 1:
                accuY._vi = _mm_add_epi16(accuY._vi, reg2._vi);
                accuX._vi = _mm_add_epi16(_mm_slli_epi16(_mm_sub_epi16(reg3._vi, reg1._vi), 1), accuX._vi);
                break;
            default:
                break;
            }
        }
        tmp._vf = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(accuY._vi));
        tmp1._vf = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(accuX._vi));
        RADv8_fp32._vf = _mm256_atan2_ps(tmp1._vf, tmp._vf);

        Gv8_fp32._vf = _mm256_fmadd_ps(tmp._vf, tmp._vf, _mm256_mul_ps(tmp1._vf, tmp1._vf));

        _mm256_store_ps(G + dex_dst, Gv8_fp32._vf);
        _mm256_store_ps(dir + dex_dst, RADv8_fp32._vf);

        dex_dst += 8;
        dex_src += ((j == 0) ? 7 : 8);
    }

    for (int i = 1; i < proc_dims.y - 1; ++i)
    {
        dex_src = i * Wsrc;
        dex_dst = i * Wdst;
        for (int j = 0; j < proc_dims.x; ++j) {
            accuY._vi = _mm_set1_epi16(0);      accuX._vi = _mm_set1_epi16(0);

            for (int k = 0; k < 3; ++k) 
            {
                if (j == 0) {
                    recv._vf = _mm_load_ps((float*)(src + dex_src + ((Wsrc) * k)));
                    recv._vi = _mm_shuffle_epi8(recv._vi, _shift_cont);
                }
                else {
                    recv._vf = _mm_loadu_ps((float*)(src + dex_src + ((Wsrc) * k)));
                }
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
            RADv8_fp32._vf = _mm256_atan2_ps(tmp1._vf, tmp._vf);
            
            Gv8_fp32._vf = _mm256_fmadd_ps(tmp._vf, tmp._vf, _mm256_mul_ps(tmp1._vf, tmp1._vf));

            _mm256_store_ps(G + dex_dst, Gv8_fp32._vf);
            _mm256_store_ps(dir + dex_dst, RADv8_fp32._vf);

            dex_dst += 8;
            dex_src += ((j == 0) ? 7 : 8);
        }
    }

    for (int j = 0; j < proc_dims.x; ++j) {
        accuY._vi = _mm_set1_epi16(0);      accuX._vi = _mm_set1_epi16(0);
        dex_src = (proc_dims.y - 1) * Wsrc;
        dex_dst = (proc_dims.y - 1) * Wdst;
        for (int k = 0; k < 2; ++k)
        {
            if (j == 0) {
                recv._vf = _mm_load_ps((float*)(src + dex_src + ((Wsrc)*k)));
                recv._vi = _mm_shuffle_epi8(recv._vi, _shift_cont);
            }
            else {
                recv._vf = _mm_loadu_ps((float*)(src + dex_src + ((Wsrc)*k)));
            }
            reg1._vi = _mm_cvtepu8_epi16(recv._vi);
            reg2._vi = _mm_shuffle_epi8(recv._vi, _shift_cont);
            reg3._vi = _mm_cvtepu8_epi16(_mm_shuffle_epi8(reg2._vi, _shift_cont));
            reg2._vi = _mm_cvtepu8_epi16(reg2._vi);
            reg2._vi = _mm_add_epi16(_mm_slli_epi16(reg2._vi, 1), _mm_add_epi16(reg1._vi, reg3._vi));

            switch (k) {
            case 0:
                accuY._vi = _mm_sub_epi16(accuY._vi, reg2._vi);
                accuX._vi = _mm_add_epi16(_mm_slli_epi16(_mm_sub_epi16(reg3._vi, reg1._vi), 1), accuX._vi);
                break;
            case 1:
                accuX._vi = _mm_add_epi16(_mm_slli_epi16(_mm_sub_epi16(reg3._vi, reg1._vi), 1), accuX._vi);
                break;
            default:
                break;
            }
        }
        tmp._vf = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(accuY._vi));
        tmp1._vf = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(accuX._vi));
        RADv8_fp32._vf = _mm256_atan2_ps(tmp1._vf, tmp._vf);

        Gv8_fp32._vf = _mm256_fmadd_ps(tmp._vf, tmp._vf, _mm256_mul_ps(tmp1._vf, tmp1._vf));

        _mm256_store_ps(G + dex_dst, Gv8_fp32._vf);
        _mm256_store_ps(dir + dex_dst, RADv8_fp32._vf);

        dex_dst += 8;
        dex_src += ((j == 0) ? 7 : 8);
    }
}



#define _22_5DEG_RAD 0.392699075
#define _67_5DEG_RAD 1.178097225
#define _112_5DEG_RAD 1.963495375
#define _157_5DEG_RAD 2.748893525




_THREAD_FUNCTION_ void
decx::vis::CPUK::Edge_Detector_Post_processing(const float* __restrict      G_info_map, 
                                               const float* __restrict      dir_info_map, 
                                               uint8_t*                     dst, 
                                               const uint                   Wsrc, 
                                               const uint                   Wdst,
                                               const uint2                  proc_dims, 
                                               const float2                 _thres)
{
    size_t dex_src = 0, dex_dst = 0;
    bool is_max = false, is_keep = false;

    register decx::_Vector4f _neigbour_3[3];
    uint8_t res = 0;

    for (int i = 0; i < proc_dims.y; ++i) {
        dex_src = i * Wsrc;
        dex_dst = i * Wdst;
        for (int j = 0; j < proc_dims.x; ++j) 
        {
            res = 0;

            const float* _neigbour_start_ptr = G_info_map + dex_src - 1 - Wsrc;
            
            for (int _n = 0; _n < 3; ++_n) {
                _neigbour_3[_n]._vec = _mm_loadu_ps(_neigbour_start_ptr + _n * Wsrc);
            }
            decx::vis::_gradient_info _center;
            _center.gradient = _neigbour_3[1]._vec_sp[1];
            _center.direction_rad = dir_info_map[dex_src];
            float abs_dir = _center.direction_rad < 0 ? (3.1415926 + _center.direction_rad) : _center.direction_rad;
            
            if ((abs_dir <= _67_5DEG_RAD && abs_dir > _22_5DEG_RAD)) {
                is_max = (_center.gradient > _neigbour_3[0]._vec_sp[0]) && (_center.gradient > _neigbour_3[2]._vec_sp[2]);
            }
            else if ((abs_dir <= _112_5DEG_RAD && abs_dir > _67_5DEG_RAD)) {
                is_max = (_center.gradient > _neigbour_3[0]._vec_sp[1]) && (_center.gradient > _neigbour_3[2]._vec_sp[1]);
            }
            else if ((abs_dir <= _157_5DEG_RAD && abs_dir > _112_5DEG_RAD)) {
                is_max = (_center.gradient > _neigbour_3[0]._vec_sp[2]) && (_center.gradient > _neigbour_3[2]._vec_sp[0]);
            }
            else {
                is_max = (_center.gradient > _neigbour_3[1]._vec_sp[0]) && (_center.gradient > _neigbour_3[1]._vec_sp[2]);
            }

            if (is_max && (_center.gradient > _thres.x)) {
                if (_center.gradient < _thres.y) {
                    for (int _n = 0; _n < 9; ++_n) {
                        is_keep = is_keep && (_center.gradient > _neigbour_3[_n / 3]._vec_sp[_n % 3]);
                    }
                    res = is_keep ? 255 : 0;
                }
                else {
                    res = 255;
                }
            }
            dst[dex_dst] = res;
            ++dex_dst;
            ++dex_src;
        }
    }
}
