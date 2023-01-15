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


_THREAD_FUNCTION_ void
decx::vis::CPUK::Scharr_XY_uint8_T(const uint8_t* __restrict  src, 
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
            reg2._vi = _mm_add_epi16(_mm_mullo_epi16(reg2._vi, _mm_set1_epi16(10)), 
                _mm_mullo_epi16(_mm_add_epi16(reg1._vi, reg3._vi), _mm_set1_epi16(3)));

            switch (k) {
            case 0:
                accuX._vi = _mm_add_epi16(_mm_mullo_epi16(_mm_sub_epi16(reg3._vi, reg1._vi), _mm_set1_epi16(10)), accuX._vi);
                break;
            case 1:
                accuY._vi = _mm_add_epi16(accuY._vi, reg2._vi);
                accuX._vi = _mm_add_epi16(_mm_mullo_epi16(_mm_sub_epi16(reg3._vi, reg1._vi), _mm_set1_epi16(3)), accuX._vi);
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
                reg2._vi = _mm_add_epi16(_mm_mullo_epi16(reg2._vi, _mm_set1_epi16(10)),
                    _mm_mullo_epi16(_mm_add_epi16(reg1._vi, reg3._vi), _mm_set1_epi16(3)));

                switch (k) {
                case 0:
                    accuY._vi = _mm_sub_epi16(accuY._vi, reg2._vi);
                    accuX._vi = _mm_add_epi16(_mm_mullo_epi16(_mm_sub_epi16(reg3._vi, reg1._vi), _mm_set1_epi16(3)), accuX._vi);
                    break;
                case 1:
                    accuX._vi = _mm_add_epi16(_mm_mullo_epi16(_mm_sub_epi16(reg3._vi, reg1._vi), _mm_set1_epi16(10)), accuX._vi);
                    break;
                case 2:
                    accuY._vi = _mm_add_epi16(accuY._vi, reg2._vi);
                    accuX._vi = _mm_add_epi16(_mm_mullo_epi16(_mm_sub_epi16(reg3._vi, reg1._vi), _mm_set1_epi16(3)), accuX._vi);
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
decx::vis::CPUK::Scharr_XY_uint8(const uint8_t* __restrict  src, 
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
                reg2._vi = _mm_add_epi16(_mm_mullo_epi16(reg2._vi, _mm_set1_epi16(10)),
                    _mm_mullo_epi16(_mm_add_epi16(reg1._vi, reg3._vi), _mm_set1_epi16(3)));

                switch (k) {
                case 0:
                    accuY._vi = _mm_sub_epi16(accuY._vi, reg2._vi);
                    accuX._vi = _mm_add_epi16(_mm_mullo_epi16(_mm_sub_epi16(reg3._vi, reg1._vi), _mm_set1_epi16(3)), accuX._vi);
                    break;
                case 1:
                    accuX._vi = _mm_add_epi16(_mm_mullo_epi16(_mm_sub_epi16(reg3._vi, reg1._vi), _mm_set1_epi16(10)), accuX._vi);
                    break;
                case 2:
                    accuY._vi = _mm_add_epi16(accuY._vi, reg2._vi);
                    accuX._vi = _mm_add_epi16(_mm_mullo_epi16(_mm_sub_epi16(reg3._vi, reg1._vi), _mm_set1_epi16(3)), accuX._vi);
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
decx::vis::CPUK::Scharr_XY_uint8_B(const uint8_t* __restrict  src, 
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
                reg2._vi = _mm_add_epi16(_mm_mullo_epi16(reg2._vi, _mm_set1_epi16(10)),
                    _mm_mullo_epi16(_mm_add_epi16(reg1._vi, reg3._vi), _mm_set1_epi16(3)));

                switch (k) {
                case 0:
                    accuY._vi = _mm_sub_epi16(accuY._vi, reg2._vi);
                    accuX._vi = _mm_add_epi16(_mm_mullo_epi16(_mm_sub_epi16(reg3._vi, reg1._vi), _mm_set1_epi16(3)), accuX._vi);
                    break;
                case 1:
                    accuX._vi = _mm_add_epi16(_mm_mullo_epi16(_mm_sub_epi16(reg3._vi, reg1._vi), _mm_set1_epi16(10)), accuX._vi);
                    break;
                case 2:
                    accuY._vi = _mm_add_epi16(accuY._vi, reg2._vi);
                    accuX._vi = _mm_add_epi16(_mm_mullo_epi16(_mm_sub_epi16(reg3._vi, reg1._vi), _mm_set1_epi16(3)), accuX._vi);
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
            reg2._vi = _mm_add_epi16(_mm_mullo_epi16(reg2._vi, _mm_set1_epi16(10)),
                _mm_mullo_epi16(_mm_add_epi16(reg1._vi, reg3._vi), _mm_set1_epi16(3)));

            switch (k) {
            case 0:
                accuY._vi = _mm_sub_epi16(accuY._vi, reg2._vi);
                accuX._vi = _mm_add_epi16(_mm_mullo_epi16(_mm_sub_epi16(reg3._vi, reg1._vi), _mm_set1_epi16(3)), accuX._vi);
                break;
            case 1:
                accuX._vi = _mm_add_epi16(_mm_mullo_epi16(_mm_sub_epi16(reg3._vi, reg1._vi), _mm_set1_epi16(10)), accuX._vi);
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
decx::vis::CPUK::Scharr_XY_uint8_TB(const uint8_t* __restrict  src, 
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
            reg2._vi = _mm_add_epi16(_mm_mullo_epi16(reg2._vi, _mm_set1_epi16(10)),
                _mm_mullo_epi16(_mm_add_epi16(reg1._vi, reg3._vi), _mm_set1_epi16(3)));

            switch (k) {
            case 0:
                accuX._vi = _mm_add_epi16(_mm_mullo_epi16(_mm_sub_epi16(reg3._vi, reg1._vi), _mm_set1_epi16(10)), accuX._vi);
                break;
            case 1:
                accuY._vi = _mm_add_epi16(accuY._vi, reg2._vi);
                accuX._vi = _mm_add_epi16(_mm_mullo_epi16(_mm_sub_epi16(reg3._vi, reg1._vi), _mm_set1_epi16(3)), accuX._vi);
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
                reg2._vi = _mm_add_epi16(_mm_mullo_epi16(reg2._vi, _mm_set1_epi16(10)),
                    _mm_mullo_epi16(_mm_add_epi16(reg1._vi, reg3._vi), _mm_set1_epi16(3)));

                switch (k) {
                case 0:
                    accuY._vi = _mm_sub_epi16(accuY._vi, reg2._vi);
                    accuX._vi = _mm_add_epi16(_mm_mullo_epi16(_mm_sub_epi16(reg3._vi, reg1._vi), _mm_set1_epi16(3)), accuX._vi);
                    break;
                case 1:
                    accuX._vi = _mm_add_epi16(_mm_mullo_epi16(_mm_sub_epi16(reg3._vi, reg1._vi), _mm_set1_epi16(10)), accuX._vi);
                    break;
                case 2:
                    accuY._vi = _mm_add_epi16(accuY._vi, reg2._vi);
                    accuX._vi = _mm_add_epi16(_mm_mullo_epi16(_mm_sub_epi16(reg3._vi, reg1._vi), _mm_set1_epi16(3)), accuX._vi);
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
            reg2._vi = _mm_add_epi16(_mm_mullo_epi16(reg2._vi, _mm_set1_epi16(10)),
                _mm_mullo_epi16(_mm_add_epi16(reg1._vi, reg3._vi), _mm_set1_epi16(3)));

            switch (k) {
            case 0:
                accuY._vi = _mm_sub_epi16(accuY._vi, reg2._vi);
                accuX._vi = _mm_add_epi16(_mm_mullo_epi16(_mm_sub_epi16(reg3._vi, reg1._vi), _mm_set1_epi16(3)), accuX._vi);
                break;
            case 1:
                accuX._vi = _mm_add_epi16(_mm_mullo_epi16(_mm_sub_epi16(reg3._vi, reg1._vi), _mm_set1_epi16(10)), accuX._vi);
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
