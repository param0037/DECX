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

#include "cvt_colors.h"
#include "../../../common/FMGR/fragment_arrangment.h"
#include "../../core/thread_management/thread_arrange.h"


_THREAD_FUNCTION_ void 
decx::vis::CPUK::_BGR2Gray_UC42UC(const float* __restrict    src, 
                              float* __restrict          dst, 
                              const int2                 dims,
                              const uint32_t             pitchsrc, 
                              const uint32_t             pitchdst)
{
    uint64_t glo_dex_src = 0, glo_dex_dst = 0;
    const __m128i _shuffle_var = _mm_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
    __m128i __recv;
    __m128i __res, __buf;

    for (int i = 0; i < dims.y; ++i) {
        glo_dex_src = i * pitchsrc;
        glo_dex_dst = i * pitchdst / 4;
        for (int j = 0; j < dims.x; ++j) {
            __recv = _mm_castps_si128(_mm_load_ps(src + glo_dex_src));
            glo_dex_src += 4;

            __recv = _mm_shuffle_epi8(__recv, _shuffle_var);

            __buf = _mm_cvtepu8_epi32(__recv);
            __res = _mm_mullo_epi32(__buf, _mm_set1_epi32(19595));

            __buf = _mm_cvtepu8_epi32(_mm_shuffle_epi32(__recv, 0b01010101));
            __buf = _mm_mullo_epi32(__buf, _mm_set1_epi32(38469));
            __res = _mm_add_epi32(__buf, __res);

            __buf = _mm_cvtepu8_epi32(_mm_shuffle_epi32(__recv, 0b10101010));
            __buf = _mm_mullo_epi32(__buf, _mm_set1_epi32(7472));
            __res = _mm_add_epi32(__buf, __res);

            __buf = _mm_srli_epi32(__res, 16);
            __res = _mm_shuffle_epi8(__buf, _shuffle_var);
            dst[glo_dex_dst] = *((float*)&__res);
            
            ++glo_dex_dst;
        }
    }
}


_THREAD_FUNCTION_ void 
decx::vis::CPUK::_BGR2Mean_UC42UC(const float* __restrict   src, 
                              float* __restrict         dst, 
                              const int2                dims, 
                              const uint32_t            pitchsrc, 
                              const uint32_t            pitchdst)
{
    uint64_t glo_dex_src = 0, glo_dex_dst = 0;
    const __m128i _shuffle_var = _mm_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
    __m128i __recv;
    __m128i __res, __buf;

    for (int i = 0; i < dims.y; ++i) {
        glo_dex_src = i * pitchsrc;
        glo_dex_dst = i * pitchdst / 4;
        for (int j = 0; j < dims.x; ++j) {
            __recv = _mm_castps_si128(_mm_load_ps(src + glo_dex_src));
            glo_dex_src += 4;

            __recv = _mm_shuffle_epi8(__recv, _shuffle_var);

            __buf = _mm_cvtepu8_epi32(__recv);
            __res = _mm_mullo_epi32(__buf, _mm_set1_epi32(21846));

            __buf = _mm_cvtepu8_epi32(_mm_shuffle_epi32(__recv, 0b01010101));
            __buf = _mm_mullo_epi32(__buf, _mm_set1_epi32(21846));
            __res = _mm_add_epi32(__buf, __res);

            __buf = _mm_cvtepu8_epi32(_mm_shuffle_epi32(__recv, 0b10101010));
            __buf = _mm_mullo_epi32(__buf, _mm_set1_epi32(21846));
            __res = _mm_add_epi32(__buf, __res);

            __buf = _mm_srli_epi32(__res, 16);
            __res = _mm_shuffle_epi8(__buf, _shuffle_var);
            dst[glo_dex_dst] = *((float*)&__res);

            ++glo_dex_dst;
        }
    }
}



_THREAD_FUNCTION_ void 
decx::vis::CPUK::_Preserve_B_UC42UC(const float* __restrict     src, 
                                float* __restrict           dst, 
                                const int2                  dims, 
                                const uint32_t              pitchsrc, 
                                const uint32_t              pitchdst)
{
    uint64_t glo_dex_src = 0, glo_dex_dst = 0;
    const __m128i _shuffle_var = _mm_setr_epi8(2, 6, 10, 14, 3, 7, 11, 15, 0, 4, 8, 12, 1, 5, 9, 13);
    __m128i __recv;
    __m128i __res;

    for (int i = 0; i < dims.y; ++i) {
        glo_dex_src = i * pitchsrc;
        glo_dex_dst = i * pitchdst / 4;
        for (int j = 0; j < dims.x; ++j) {
            __recv = _mm_castps_si128(_mm_load_ps(src + glo_dex_src));
            glo_dex_src += 4;

            __res = _mm_shuffle_epi8(__recv, _shuffle_var);
            dst[glo_dex_dst] = *((float*)&__res);
            ++glo_dex_dst;
        }
    }
}



_THREAD_FUNCTION_ void 
decx::vis::CPUK::_Preserve_G_UC42UC(const float* __restrict     src, 
                                float* __restrict           dst, 
                                const int2                  dims, 
                                const uint32_t              pitchsrc, 
                                const uint32_t              pitchdst)
{
    uint64_t glo_dex_src = 0, glo_dex_dst = 0;
    const __m128i _shuffle_var = _mm_setr_epi8(1, 5, 9, 13, 3, 7, 11, 15, 0, 4, 8, 12, 2, 6, 10, 14);
    __m128i __recv;
    __m128i __res;

    for (int i = 0; i < dims.y; ++i) {
        glo_dex_src = i * pitchsrc;
        glo_dex_dst = i * pitchdst / 4;
        for (int j = 0; j < dims.x; ++j) {
            __recv = _mm_castps_si128(_mm_load_ps(src + glo_dex_src));
            glo_dex_src += 4;

            __res = _mm_shuffle_epi8(__recv, _shuffle_var);
            dst[glo_dex_dst] = *((float*)&__res);
            ++glo_dex_dst;
        }
    }
}



_THREAD_FUNCTION_ void 
decx::vis::CPUK::_Preserve_R_UC42UC(const float* __restrict     src, 
                                float* __restrict           dst, 
                                const int2                  dims, 
                                const uint32_t              pitchsrc, 
                                const uint32_t              pitchdst)
{
    uint64_t glo_dex_src = 0, glo_dex_dst = 0;
    const __m128i _shuffle_var = _mm_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
    __m128i __recv;
    __m128i __res;

    for (int i = 0; i < dims.y; ++i) {
        glo_dex_src = i * pitchsrc;
        glo_dex_dst = i * pitchdst / 4;
        for (int j = 0; j < dims.x; ++j) {
            __recv = _mm_castps_si128(_mm_load_ps(src + glo_dex_src));
            glo_dex_src += 4;

            __res = _mm_shuffle_epi8(__recv, _shuffle_var);
            dst[glo_dex_dst] = *((float*)&__res);
            ++glo_dex_dst;
        }
    }
}



_THREAD_FUNCTION_ void 
decx::vis::CPUK::_Preserve_A_UC42UC(const float* __restrict     src, 
                                float* __restrict           dst, 
                                const int2                  dims, 
                                const uint32_t              pitchsrc, 
                                const uint32_t              pitchdst)
{
    uint64_t glo_dex_src = 0, glo_dex_dst = 0;
    const __m128i _shuffle_var = _mm_setr_epi8(3, 7, 11, 15, 1, 5, 9, 13, 0, 4, 8, 12, 2, 6, 10, 14);
    __m128i __recv;
    __m128i __res;

    for (int i = 0; i < dims.y; ++i) {
        glo_dex_src = i * pitchsrc;
        glo_dex_dst = i * pitchdst / 4;
        for (int j = 0; j < dims.x; ++j) {
            __recv = _mm_castps_si128(_mm_load_ps(src + glo_dex_src));
            glo_dex_src += 4;

            __res = _mm_shuffle_epi8(__recv, _shuffle_var);
            dst[glo_dex_dst] = *((float*)&__res);
            ++glo_dex_dst;
        }
    }
}



_THREAD_FUNCTION_ void 
decx::vis::CPUK::_RGB2YUV_UC42UC4(const float* __restrict     src, 
                                     float* __restrict           dst, 
                                     const int2                  dims, 
                                     const uint32_t              pitchsrc, 
                                     const uint32_t              pitchdst)
{
    uint64_t glo_dex_src = 0, glo_dex_dst = 0;

    const __m256i shuffle_var2 = _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7);
    const __m256i cvtu8_u16_mask = _mm256_set1_epi64x(0x00FF00FF00FF00FF);
    const __m256i shuffle_cvti16_u8 = _mm256_setr_epi64x(0x00000a0200000800, 0x00000e0600000c04,
                                                         0x00000a0200000800, 0x00000e0600000c04);

    decx::utils::simd::xmm256_reg _IO, _reg256, _YU_res;
    decx::utils::simd::xmm128_reg _V_res;

    for (uint32_t i = 0; i < dims.y; ++i) {
        glo_dex_src = i * pitchsrc;
        glo_dex_dst = i * pitchdst;
        for (uint32_t j = 0; j < dims.x; ++j) 
        {
            // [[R0 G0 B0 X0], [R1 G1 B1 X1], ... [R7 G7 B7 X7]] <8-bit>
            _IO._vf = _mm256_load_ps(src + glo_dex_src);
            // R
            // [[R0 R1 R2 R3], [R4 R5 R6 R7], [R0 R1 R2 R3], [R4 R5 R6 R7]] <16-bit>
            _reg256._vi = _mm256_and_si256(_mm256_shuffle_epi8(_IO._vi, _mm256_set1_epi64x(0x000c000800040000)), cvtu8_u16_mask);
            _reg256._vi = _mm256_permutevar8x32_epi32(_reg256._vi, shuffle_var2);

            _YU_res._vi = _mm256_mullo_epi16(_reg256._vi, _mm256_setr_epi32(5046349, 5046349, 5046349, 5046349, 
                                                                            -2752555, -2752555, -2752555, -2752555));
            _V_res._vi = _mm_slli_epi16(_mm256_castsi256_si128(_reg256._vi), 7);

            // G
            // [[G0 G1 G2 G3], [G4 G5 G6 G7], [G0 G1 G2 G3], [G4 G5 G6 G7]] <16-bit>
            _reg256._vi = _mm256_and_si256(_mm256_shuffle_epi8(_IO._vi, _mm256_set1_epi64x(0x000d000900050001)), cvtu8_u16_mask);
            _reg256._vi = _mm256_permutevar8x32_epi32(_reg256._vi, shuffle_var2);

            _YU_res._vi = _mm256_add_epi16(_mm256_mullo_epi16(_reg256._vi, _mm256_setr_epi32(9830550, 9830550, 9830550, 9830550, 
                                                                                            -5505109, -5505109, -5505109, -5505109)), _YU_res._vi);
            _V_res._vi = _mm_sub_epi16(_V_res._vi, _mm_mullo_epi16(_mm256_castsi256_si128(_reg256._vi), _mm_set1_epi16(107)));

            // B
            // [[B0 B1 B2 B3], [B4 B5 B6 B7], [B0 B1 B2 B3], [B4 B5 B6 B7]] <16-bit>
            _reg256._vi = _mm256_and_si256(_mm256_shuffle_epi8(_IO._vi, _mm256_set1_epi64x(0x000e000a00060002)), cvtu8_u16_mask);
            _reg256._vi = _mm256_permutevar8x32_epi32(_reg256._vi, shuffle_var2);

            _YU_res._vi = _mm256_add_epi16(_mm256_mullo_epi16(_reg256._vi, _mm256_setr_epi32(1900573, 1900573, 1900573, 1900573, 
                                                                                            8388736, 8388736, 8388736, 8388736)), _YU_res._vi);
            _V_res._vi = _mm_sub_epi16(_V_res._vi, _mm_mullo_epi16(_mm256_castsi256_si128(_reg256._vi), _mm_set1_epi16(21)));

            _YU_res._vi = _mm256_add_epi16(_YU_res._vi, _mm256_setr_epi32(0, 0, 0, 0, -2147450880, -2147450880, -2147450880, -2147450880));
            _V_res._vi = _mm_add_epi16(_V_res._vi, _mm_set1_epi16(32768));

            _YU_res._vi = _mm256_srai_epi16(_YU_res._vi, 8);
            _V_res._vi = _mm_srai_epi16(_V_res._vi, 8);

            // CVT
            _YU_res._vi = _mm256_permutevar8x32_epi32(_YU_res._vi, _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7));
            _YU_res._vi = _mm256_shuffle_epi8(_YU_res._vi, shuffle_cvti16_u8);
            _YU_res._vi = _mm256_and_si256(_YU_res._vi, _mm256_set1_epi32(0xffff));

            // Merge V channel
            _reg256._vi = _mm256_castsi128_si256(_V_res._vi);
            _reg256._vi = _mm256_permutevar8x32_epi32(_reg256._vi, _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7));

            _reg256._vi = _mm256_shuffle_epi8(_reg256._vi, _mm256_setr_epi64x(0x0002000000000000, 0x0006000000040000,
                0x0002000000000000, 0x0006000000040000));

            _YU_res._vi = _mm256_xor_si256(_YU_res._vi, _mm256_and_si256(_reg256._vi, _mm256_set1_epi32(0x00ff0000)));

            // Merge Alpha channel
            _YU_res._vi = _mm256_xor_si256(_YU_res._vi, _mm256_and_si256(_IO._vi, _mm256_set1_epi32(0xff000000)));

            _mm256_store_ps(dst + glo_dex_dst, _YU_res._vf);

            glo_dex_src += 8;
            glo_dex_dst += 8;
        }
    }
}




_THREAD_FUNCTION_ void 
decx::vis::CPUK::_YUV2RGB_UC42UC4(const float* __restrict     src,
                                     float* __restrict           dst, 
                                     const int2                  dims, 
                                     const uint32_t              pitchsrc, 
                                     const uint32_t              pitchdst)
{
    uint64_t glo_dex_src = 0, glo_dex_dst = 0;

    const __m256i shuffle_var2 = _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7);
    const __m256i cvtu8_u16_mask = _mm256_set1_epi64x(0x00FF00FF00FF00FF);
    const __m256i shuffle_cvti16_u8 = _mm256_setr_epi64x(0x00000a0200000800, 0x00000e0600000c04,
                                                         0x00000a0200000800, 0x00000e0600000c04);

    decx::utils::simd::xmm256_reg _IO, _reg256, _RG_res;
    decx::utils::simd::xmm128_reg _B_res;

    for (uint32_t i = 0; i < dims.y; ++i) {
        glo_dex_src = i * pitchsrc;
        glo_dex_dst = i * pitchdst;
        for (uint32_t j = 0; j < dims.x; ++j) 
        {
            // [[R0 G0 B0 X0], [R1 G1 B1 X1], ... [R7 G7 B7 X7]] <8-bit>
            _IO._vf = _mm256_load_ps(src + glo_dex_src);
            // Y
            // [[R0 R1 R2 R3], [R4 R5 R6 R7], [R0 R1 R2 R3], [R4 R5 R6 R7]] <16-bit>
            _reg256._vi = _mm256_and_si256(_mm256_shuffle_epi8(_IO._vi, _mm256_set1_epi64x(0x000c000800040000)), cvtu8_u16_mask);
            _RG_res._vi = _mm256_permutevar8x32_epi32(_reg256._vi, shuffle_var2);

            _B_res._vi = _mm256_castsi256_si128(_RG_res._vi);

            // Cr (U)
            // [[G0 G1 G2 G3], [G4 G5 G6 G7], [G0 G1 G2 G3], [G4 G5 G6 G7]] <16-bit>
            _reg256._vi = _mm256_and_si256(_mm256_shuffle_epi8(_IO._vi, _mm256_set1_epi64x(0x000d000900050001)), cvtu8_u16_mask);
            _reg256._vi = _mm256_permutevar8x32_epi32(_reg256._vi, shuffle_var2);
            _reg256._vi = _mm256_sub_epi16(_reg256._vi, _mm256_set1_epi16(128));

            _RG_res._vi = _mm256_add_epi16(
                _mm256_srai_epi16(_mm256_mullo_epi16(_reg256._vi, _mm256_setr_epi32(0, 0, 0, 0, 
                                                                -5701720, -5701720, -5701720, -5701720)), 8), _RG_res._vi);
            _B_res._vi = _mm_add_epi16(_B_res._vi, _mm256_castsi256_si128(_reg256._vi));
            _B_res._vi = _mm_add_epi16(_B_res._vi,
                _mm_srai_epi16(_mm_mullo_epi16(_mm256_castsi256_si128(_reg256._vi), _mm_set1_epi16(198)), 8));
            
            // Cb (V)
            // [[B0 B1 B2 B3], [B4 B5 B6 B7], [B0 B1 B2 B3], [B4 B5 B6 B7]] <16-bit>
            _reg256._vi = _mm256_and_si256(_mm256_shuffle_epi8(_IO._vi, _mm256_set1_epi64x(0x000e000a00060002)), cvtu8_u16_mask);
            _reg256._vi = _mm256_permutevar8x32_epi32(_reg256._vi, shuffle_var2);
            _reg256._vi = _mm256_sub_epi16(_reg256._vi, _mm256_set1_epi16(128));

            _RG_res._vi = _mm256_add_epi16(_RG_res._vi, _mm256_and_si256(_reg256._vi, _mm256_setr_epi64x(0xffffffffffffffff, 0xffffffffffffffff, 0, 0)));
            _RG_res._vi = _mm256_add_epi16(
                _mm256_srai_epi16(_mm256_mullo_epi16(_reg256._vi, _mm256_setr_epi32(6750311, 6750311, 6750311, 6750311, 
                                                                        -11927735, -11927735, -11927735, -11927735)), 8), _RG_res._vi);
            
            // Clamp to range
            _RG_res._vi = _mm256_or_si256(_RG_res._vi, _mm256_cmpgt_epi16(_RG_res._vi, _mm256_set1_epi16(255)));
            _RG_res._vi = _mm256_and_si256(_RG_res._vi, _mm256_cmpgt_epi16(_RG_res._vi, _mm256_set1_epi16(0)));
            _B_res._vi = _mm_or_si128(_B_res._vi, _mm_cmpgt_epi16(_B_res._vi, _mm_set1_epi16(255)));
            _B_res._vi = _mm_and_si128(_B_res._vi, _mm_cmpgt_epi16(_B_res._vi, _mm_set1_epi16(0)));

            // CVT
            _RG_res._vi = _mm256_permutevar8x32_epi32(_RG_res._vi, _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7));
            _RG_res._vi = _mm256_shuffle_epi8(_RG_res._vi, shuffle_cvti16_u8);
            _RG_res._vi = _mm256_and_si256(_RG_res._vi, _mm256_set1_epi32(0xffff));

            // Merge B channel
            _reg256._vi = _mm256_castsi128_si256(_B_res._vi);
            _reg256._vi = _mm256_permutevar8x32_epi32(_reg256._vi, _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7));

            _reg256._vi = _mm256_shuffle_epi8(_reg256._vi, _mm256_setr_epi64x(0x0002000000000000, 0x0006000000040000,
                0x0002000000000000, 0x0006000000040000));

            _RG_res._vi = _mm256_xor_si256(_RG_res._vi, _mm256_and_si256(_reg256._vi, _mm256_set1_epi32(0x00ff0000)));

            // Merge Alpha channel
            _RG_res._vi = _mm256_xor_si256(_RG_res._vi, _mm256_and_si256(_IO._vi, _mm256_set1_epi32(0xff000000)));

            _mm256_store_ps(dst + glo_dex_dst, _RG_res._vf);

            glo_dex_src += 8;
            glo_dex_dst += 8;
        }
    }
}




_THREAD_FUNCTION_ void 
decx::vis::CPUK::_RGB2BGR_UC42UC4(const float* __restrict     src, 
                                  float* __restrict           dst, 
                                  const int2                  dims, 
                                  const uint32_t              pitchsrc, 
                                  const uint32_t              pitchdst)
{
    uint64_t glo_dex_src = 0, glo_dex_dst = 0;
    const __m256i shuffle_var = _mm256_setr_epi64x(0x0704050603000102, 0x0f0c0d0e0b08090a,
        0x0704050603000102, 0x0f0c0d0e0b08090a);
    
    decx::utils::simd::xmm256_reg _IO;

    for (uint32_t i = 0; i < dims.y; ++i) {
        glo_dex_src = i * pitchsrc;
        glo_dex_dst = i * pitchdst;
        for (uint32_t j = 0; j < dims.x; ++j) 
        {
            // [[R0 G0 B0 X0], [R1 G1 B1 X1], ... [R7 G7 B7 X7]] <8-bit>
            _IO._vf = _mm256_load_ps(src + glo_dex_src);
            _IO._vi = _mm256_shuffle_epi8(_IO._vi, shuffle_var);

            _mm256_store_ps(dst + glo_dex_dst, _IO._vf);

            glo_dex_src += 8;
            glo_dex_dst += 8;
        }
    }
}



// --------------------------------------- CALLERS --------------------------------------------------------


void decx::vis::_channel_ops_UC42UC_caller(decx::vis::channel_ops_kernel kernel, 
                                            const float* src, float* dst, const int2 dims, 
                                            const uint32_t pitchsrc, const uint32_t pitchdst)
{
    decx::utils::_thread_arrange_1D t1D(decx::cpu::_get_permitted_concurrency());
    decx::utils::frag_manager f_mgr;
    decx::utils::frag_manager_gen(&f_mgr, dims.y, t1D.total_thread);

    int2 sub_dims = make_int2(decx::utils::ceil<uint32_t>(dims.x, 4), f_mgr.frag_len);

    uint64_t fragment_src = pitchsrc * (uint64_t)sub_dims.y, 
             fragment_dst = (pitchdst / 4) * (uint64_t)sub_dims.y,
             offset_src = 0,
             offset_dst = 0;

    for (int i = 0; i < t1D.total_thread - 1; ++i) {
        t1D._async_thread[i] = decx::cpu::register_task_default(kernel, src + offset_src, dst + offset_dst, sub_dims,
            pitchsrc, pitchdst);
        offset_src += fragment_src;
        offset_dst += fragment_dst;
    }

    sub_dims.y = f_mgr.is_left ? f_mgr.frag_left_over : f_mgr.frag_len;
    t1D._async_thread[decx::cpu::_get_permitted_concurrency() - 1] =
        decx::cpu::register_task_default(kernel, src + offset_src, dst + offset_dst, sub_dims,
            pitchsrc, pitchdst);

    t1D.__sync_all_threads();
}



void decx::vis::_channel_ops_UC42UC4_caller(decx::vis::channel_ops_kernel kernel, 
                                            const float* src, float* dst, const int2 dims, 
                                            const uint32_t pitchsrc, const uint32_t pitchdst)
{
    decx::utils::_thread_arrange_1D t1D(decx::cpu::_get_permitted_concurrency());
    decx::utils::frag_manager f_mgr;
    decx::utils::frag_manager_gen(&f_mgr, dims.y, t1D.total_thread);

    int2 sub_dims = make_int2(decx::utils::ceil<uint32_t>(dims.x, 8), f_mgr.frag_len);

    uint64_t fragment_src = pitchsrc * (uint64_t)sub_dims.y, 
             fragment_dst = pitchdst * (uint64_t)sub_dims.y,
             offset_src = 0,
             offset_dst = 0;

    for (int i = 0; i < t1D.total_thread - 1; ++i) {
        t1D._async_thread[i] = decx::cpu::register_task_default(kernel, src + offset_src, dst + offset_dst, sub_dims,
            pitchsrc, pitchdst);
        offset_src += fragment_src;
        offset_dst += fragment_dst;
    }

    sub_dims.y = f_mgr.is_left ? f_mgr.frag_left_over : f_mgr.frag_len;
    t1D._async_thread[decx::cpu::_get_permitted_concurrency() - 1] =
        decx::cpu::register_task_default(kernel, src + offset_src, dst + offset_dst, sub_dims,
            pitchsrc, pitchdst);

    t1D.__sync_all_threads();
}