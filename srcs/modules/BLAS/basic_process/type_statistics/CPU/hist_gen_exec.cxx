/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#include "hist_gen_exec.h"


//
//_THREAD_FUNCTION_ void
//decx::bp::CPUK::_histgen2D_u8(const uint8_t* __restrict     src,
//                              uint64_t* __restrict          _hist,
//                              const uint2                   proc_dims,
//                              const uint32_t                Wsrc,
//                              const __m256i                 _mask_row_end)
//{
//    uint64_t dex_src = 0;
//    decx::utils::simd::xmm128_reg recv;
//    decx::utils::simd::xmm256_reg reg;
//
//    const uint32_t _procW_v4 = decx::utils::ceil<uint32_t>(proc_dims.x, 4);
//    // Flush the histogram buffer
//    for (int i = 0; i < 256; ++i) {
//        _mm256_store_si256((__m256i*)_hist + i, _mm256_set1_epi64x(0));
//    }
//
//    for (uint32_t i = 0; i < proc_dims.y; ++i)
//    {
//        dex_src = i * Wsrc;
//        for (uint32_t j = 0; j < _procW_v4; ++j) 
//        {
//            recv._vf = _mm_loadu_ps((float*)(src + dex_src));
//            recv._vi = _mm_slli_epi32(_mm_cvtepu8_epi32(recv._vi), 2);
//            recv._vi = _mm_add_epi32(recv._vi, _mm_setr_epi32(0, 1, 2, 3));
//            
//            reg._vi = _mm256_i32gather_epi64((const int64_t*)_hist, recv._vi, 8);
//
//            if (j < _procW_v4 - 1) {
//                reg._vi = _mm256_add_epi64(reg._vi, _mm256_set1_epi64x(1));
//            }
//            else {
//                reg._vi = _mm256_add_epi64(reg._vi, _mask_row_end);
//            }
//            
//            _hist[_mm_extract_epi32(recv._vi, 0)] = (uint64_t)_mm256_extract_epi64(reg._vi, 0);
//            _hist[_mm_extract_epi32(recv._vi, 1)] = (uint64_t)_mm256_extract_epi64(reg._vi, 1);
//            _hist[_mm_extract_epi32(recv._vi, 2)] = (uint64_t)_mm256_extract_epi64(reg._vi, 2);
//            _hist[_mm_extract_epi32(recv._vi, 3)] = (uint64_t)_mm256_extract_epi64(reg._vi, 3);
//
//            dex_src += 4;
//        }
//    }
//}


_THREAD_FUNCTION_ void
decx::bp::CPUK::_histgen2D_u8_u64(const uint8_t* __restrict     src,
                              uint64_t* __restrict          _hist,
                              const uint2                   proc_dims,
                              const uint32_t                Wsrc,
                              const uint8_t                 _leagal_len)
{
    uint64_t dex_src = 0;
    uint8_t tmp[4];

    const uint32_t _procW_v4 = decx::utils::ceil<uint32_t>(proc_dims.x, 4);
    // Flush the histogram buffer
    for (int i = 0; i < 256 / 4; ++i) {
        _mm256_store_si256((__m256i*)_hist + i, _mm256_set1_epi64x(0));
    }

    for (uint32_t i = 0; i < proc_dims.y; ++i)
    {
        dex_src = i * Wsrc;
        for (uint32_t j = 0; j < _procW_v4 - 1; ++j)
        {
            *((uint32_t*)tmp) = *((uint32_t*)&src[dex_src]);
            ++_hist[tmp[0]];
            ++_hist[tmp[1]];
            ++_hist[tmp[2]];
            ++_hist[tmp[3]];

            dex_src += 4;
        }
        *((uint32_t*)tmp) = *((uint32_t*)&src[dex_src]);
        for (uint8_t k = 0; k < _leagal_len; ++k) {
            ++_hist[tmp[k]];
        }
    }
}