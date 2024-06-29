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


#ifndef _TRANSPOSE_EXEC_H_
#define _TRANSPOSE_EXEC_H_


#include "../../../../core/basic.h"
#include "transpose2D_config.h"


namespace decx
{
namespace blas
{
    namespace CPUK
    {
#ifdef __GNUC__
        _THREAD_CALL_ static inline void
        block8x8_transpose_u8(__m64 _regs0[8], __m64 _regs1[8]);
#endif
        //#ifdef _MSC_VER
        _THREAD_CALL_ static inline void
        block8x8_transpose_u8(uint64_t _regs0[8], uint64_t _regs1[8]);
        //#endif
    }
}
}



#ifdef __GNUC__
_THREAD_CALL_ static inline void
decx::bp::CPUK::block8x8_transpose_u8(__m64 _regs0[8], __m64 _regs1[8])
{
    const __m64 _mask_2x2_front = _mm_setr_pi16(0xFFFF, 0,      0xFFFF, 0);
    const __m64 _mask_2x2_back  = _mm_setr_pi16(0,      0xFFFF, 0,      0xFFFF);
    const __m64 _mask_1x1_even  = _mm_setr_pi8( 0xFF,   0,      0xFF,   0,      0xFF,   0,      0xFF,   0);
    const __m64 _mask_1x1_odd   = _mm_setr_pi8( 0,      0xFF,   0,      0xFF,   0,      0xFF,   0,      0xFF);

    // Transpose 4x4
    _regs1[0] = _mm_unpacklo_pi32(_regs0[0], _regs0[4]);
    _regs1[1] = _mm_unpacklo_pi32(_regs0[1], _regs0[5]);
    _regs1[2] = _mm_unpacklo_pi32(_regs0[2], _regs0[6]);
    _regs1[3] = _mm_unpacklo_pi32(_regs0[3], _regs0[7]);

    _regs1[4] = _mm_unpackhi_pi32(_regs0[0], _regs0[4]);
    _regs1[5] = _mm_unpackhi_pi32(_regs0[1], _regs0[5]);
    _regs1[6] = _mm_unpackhi_pi32(_regs0[2], _regs0[6]);
    _regs1[7] = _mm_unpackhi_pi32(_regs0[3], _regs0[7]);

    // Transpose 2x2
    _regs0[0] = _mm_xor_si64(_mm_and_si64(_regs1[0], _mask_2x2_front), 
                             _mm_and_si64(_mm_slli_si64(_regs1[2], 16), _mask_2x2_back));
    _regs0[1] = _mm_xor_si64(_mm_and_si64(_regs1[1], _mask_2x2_front), 
                             _mm_and_si64(_mm_slli_si64(_regs1[3], 16), _mask_2x2_back));

    _regs0[2] = _mm_xor_si64(_mm_and_si64(_regs1[2], _mask_2x2_back), 
                             _mm_and_si64(_mm_srli_si64(_regs1[0], 16), _mask_2x2_front));
    _regs0[3] = _mm_xor_si64(_mm_and_si64(_regs1[3], _mask_2x2_back), 
                             _mm_and_si64(_mm_srli_si64(_regs1[1], 16), _mask_2x2_front));

    _regs0[4] = _mm_xor_si64(_mm_and_si64(_regs1[4], _mask_2x2_front), 
                             _mm_and_si64(_mm_slli_si64(_regs1[6], 16), _mask_2x2_back));
    _regs0[5] = _mm_xor_si64(_mm_and_si64(_regs1[5], _mask_2x2_front), 
                             _mm_and_si64(_mm_slli_si64(_regs1[7], 16), _mask_2x2_back));

    _regs0[6] = _mm_xor_si64(_mm_and_si64(_regs1[6], _mask_2x2_back), 
                             _mm_and_si64(_mm_srli_si64(_regs1[4], 16), _mask_2x2_front));
    _regs0[7] = _mm_xor_si64(_mm_and_si64(_regs1[7], _mask_2x2_back), 
                             _mm_and_si64(_mm_srli_si64(_regs1[5], 16), _mask_2x2_front));

    // Transpose 1x1
    _regs1[0] = _mm_xor_si64(_mm_and_si64(_regs0[0], _mask_1x1_even), 
                             _mm_and_si64(_mm_slli_si64(_regs0[1], 8), _mask_1x1_odd));
    _regs1[1] = _mm_xor_si64(_mm_and_si64(_regs0[1], _mask_1x1_odd), 
                             _mm_and_si64(_mm_srli_si64(_regs0[0], 8), _mask_1x1_even));

    _regs1[2] = _mm_xor_si64(_mm_and_si64(_regs0[2], _mask_1x1_even), 
                             _mm_and_si64(_mm_slli_si64(_regs0[3], 8), _mask_1x1_odd));
    _regs1[3] = _mm_xor_si64(_mm_and_si64(_regs0[3], _mask_1x1_odd), 
                             _mm_and_si64(_mm_srli_si64(_regs0[2], 8), _mask_1x1_even));

    _regs1[4] = _mm_xor_si64(_mm_and_si64(_regs0[4], _mask_1x1_even), 
                             _mm_and_si64(_mm_slli_si64(_regs0[5], 8), _mask_1x1_odd));
    _regs1[5] = _mm_xor_si64(_mm_and_si64(_regs0[5], _mask_1x1_odd), 
                             _mm_and_si64(_mm_srli_si64(_regs0[4], 8), _mask_1x1_even));

    _regs1[6] = _mm_xor_si64(_mm_and_si64(_regs0[6], _mask_1x1_even), 
                             _mm_and_si64(_mm_slli_si64(_regs0[7], 8), _mask_1x1_odd));
    _regs1[7] = _mm_xor_si64(_mm_and_si64(_regs0[7], _mask_1x1_odd), 
                             _mm_and_si64(_mm_srli_si64(_regs0[6], 8), _mask_1x1_even));
}
#endif

//#ifdef _MSC_VER
_THREAD_CALL_ static inline void
decx::blas::CPUK::block8x8_transpose_u8(uint64_t _regs0[8], uint64_t _regs1[8])
{
    const uint64_t _mask_2x2_front = 0xFFFF0000FFFF0000;
    const uint64_t _mask_2x2_back = 0x0000FFFF0000FFFF;
    const uint64_t _mask_1x1_even = 0xFF00FF00FF00FF00;
    const uint64_t _mask_1x1_odd = 0x00FF00FF00FF00FF;

    // Transpose 4x4
    _regs1[0] = (_regs0[0] & 0x00000000FFFFFFFF) ^ ((_regs0[4] << 32) & 0xFFFFFFFF00000000);
    _regs1[1] = (_regs0[1] & 0x00000000FFFFFFFF) ^ ((_regs0[5] << 32) & 0xFFFFFFFF00000000);
    _regs1[2] = (_regs0[2] & 0x00000000FFFFFFFF) ^ ((_regs0[6] << 32) & 0xFFFFFFFF00000000);
    _regs1[3] = (_regs0[3] & 0x00000000FFFFFFFF) ^ ((_regs0[7] << 32) & 0xFFFFFFFF00000000);

    _regs1[4] = (_regs0[4] & 0xFFFFFFFF00000000) ^ ((_regs0[0] >> 32) & 0x00000000FFFFFFFF);
    _regs1[5] = (_regs0[5] & 0xFFFFFFFF00000000) ^ ((_regs0[1] >> 32) & 0x00000000FFFFFFFF);
    _regs1[6] = (_regs0[6] & 0xFFFFFFFF00000000) ^ ((_regs0[2] >> 32) & 0x00000000FFFFFFFF);
    _regs1[7] = (_regs0[7] & 0xFFFFFFFF00000000) ^ ((_regs0[3] >> 32) & 0x00000000FFFFFFFF);

    // Transpose 2x2
    _regs0[0] = (_regs1[0] & 0x0000FFFF0000FFFF) ^ ((_regs1[2] << 16) & 0xFFFF0000FFFF0000);
    _regs0[1] = (_regs1[1] & 0x0000FFFF0000FFFF) ^ ((_regs1[3] << 16) & 0xFFFF0000FFFF0000);
    _regs0[2] = (_regs1[2] & 0xFFFF0000FFFF0000) ^ ((_regs1[0] >> 16) & 0x0000FFFF0000FFFF);
    _regs0[3] = (_regs1[3] & 0xFFFF0000FFFF0000) ^ ((_regs1[1] >> 16) & 0x0000FFFF0000FFFF);

    _regs0[4] = (_regs1[4] & 0x0000FFFF0000FFFF) ^ ((_regs1[6] << 16) & 0xFFFF0000FFFF0000);
    _regs0[5] = (_regs1[5] & 0x0000FFFF0000FFFF) ^ ((_regs1[7] << 16) & 0xFFFF0000FFFF0000);
    _regs0[6] = (_regs1[6] & 0xFFFF0000FFFF0000) ^ ((_regs1[4] >> 16) & 0x0000FFFF0000FFFF);
    _regs0[7] = (_regs1[7] & 0xFFFF0000FFFF0000) ^ ((_regs1[5] >> 16) & 0x0000FFFF0000FFFF);

    // Transpose 1x1
    _regs1[0] = (_regs0[0] & 0x00FF00FF00FF00FF) ^ ((_regs0[1] << 8) & 0xFF00FF00FF00FF00);
    _regs1[1] = (_regs0[1] & 0xFF00FF00FF00FF00) ^ ((_regs0[0] >> 8) & 0x00FF00FF00FF00FF);

    _regs1[2] = (_regs0[2] & 0x00FF00FF00FF00FF) ^ ((_regs0[3] << 8) & 0xFF00FF00FF00FF00);
    _regs1[3] = (_regs0[3] & 0xFF00FF00FF00FF00) ^ ((_regs0[2] >> 8) & 0x00FF00FF00FF00FF);

    _regs1[4] = (_regs0[4] & 0x00FF00FF00FF00FF) ^ ((_regs0[5] << 8) & 0xFF00FF00FF00FF00);
    _regs1[5] = (_regs0[5] & 0xFF00FF00FF00FF00) ^ ((_regs0[4] >> 8) & 0x00FF00FF00FF00FF);

    _regs1[6] = (_regs0[6] & 0x00FF00FF00FF00FF) ^ ((_regs0[7] << 8) & 0xFF00FF00FF00FF00);
    _regs1[7] = (_regs0[7] & 0xFF00FF00FF00FF00) ^ ((_regs0[6] >> 8) & 0x00FF00FF00FF00FF);

}
//#endif


#define _AVX_MM128_TRANSPOSE_4X4_(src4, reg4) {             \
    reg4[0] = _mm_shuffle_ps(src4[0], src4[1], 0x44);       \
    reg4[2] = _mm_shuffle_ps(src4[0], src4[1], 0xEE);       \
    reg4[1] = _mm_shuffle_ps(src4[2], src4[3], 0x44);       \
    reg4[3] = _mm_shuffle_ps(src4[2], src4[3], 0xEE);       \
                                                            \
    src4[0] = _mm_shuffle_ps(reg4[0], reg4[1], 0x88);       \
    src4[1] = _mm_shuffle_ps(reg4[0], reg4[1], 0xDD);       \
    src4[2] = _mm_shuffle_ps(reg4[2], reg4[3], 0x88);       \
    src4[3] = _mm_shuffle_ps(reg4[2], reg4[3], 0xDD);       \
}



#define _AVX_MM128_TRANSPOSE_2X2_(src2, dst2) {             \
    dst2[0] = _mm_shuffle_pd(src2[0], src2[1], 0);          \
    dst2[1] = _mm_shuffle_pd(src2[0], src2[1], 3);          \
}



#define _AVX_MM256_TRANSPOSE_2X2_(src2, dst2) {                 \
    dst2[0] = _mm256_permute2f128_pd(src2[0], src2[1], 0x20);   \
    dst2[1] = _mm256_permute2f128_pd(src2[0], src2[1], 0x31);   \
}                                                               \


#endif
