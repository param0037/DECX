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

#include "../../../../common/Classes/Matrix.h"
#include "../../../../common/Classes/Vector.h"
#include "../../../../common/element_wise/common/cpu_element_wise_planner.h"


// namespace de
// {
//     namespace cpu{
//         _DECX_API_ void test_type_cast(de::Matrix& src, de::Matrix& dst);
//         _DECX_API_ void test_type_cast(de::Vector& src, de::Vector& dst);
//     }
// }



// _THREAD_FUNCTION_ static void
// _v256_cvtf32_ui8_saturated2D(const float* __restrict      src, 
//                              uint8_t* __restrict          dst, 
//                              const uint                   Wsrc,
//                              const uint                   Wdst,
//                              const uint2                  proc_dims)
// {
//     decx::utils::simd::xmm256_reg recv, _crit, store;

//     uint64_t dex_src = 0, dex_dst = 0;

//     for (int i = 0; i < proc_dims.y; ++i) {
//         dex_src = i * Wsrc;
//         dex_dst = i * Wdst;
//         for (int j = 0; j < proc_dims.x; ++j) {
//             recv._vf = _mm256_load_ps(src + dex_src);

//             _crit._vf = _mm256_cmp_ps(recv._vf, _mm256_set1_ps(0), _CMP_GT_OS);
//             store._vi = _mm256_and_si256(recv._vi, _crit._vi);

//             store._vi = _mm256_cvtps_epi32(store._vf);
//             _crit._vf = _mm256_cmp_ps(recv._vf, _mm256_set1_ps(255), _CMP_GT_OS);     // saturated_cast
//             store._vi = _mm256_or_si256(store._vi, _crit._vi);       // saturated_cast
//             store._vi = _mm256_shuffle_epi8(_mm256_and_si256(store._vi, _mm256_set1_epi32(0x000000ff)), _mm256_set1_epi32(201851904));

// #ifdef _MSC_VER
//             dst[dex_dst] = store._vi.m256i_i32[0];
//             dst[dex_dst + 1] = store._vi.m256i_i32[4];
// #endif
// #ifdef __GNUC__
//             *((int*)(dst + dex_dst)) = ((int*)&store._vi)[0];
//             *((int*)(dst + dex_dst + 4)) = ((int*)&store._vi)[4];
// #endif
//             dex_src += 8;
//             dex_dst += 8;
//         }
//     }
// }



// _THREAD_FUNCTION_ void
// _v256_cvtf32_ui8_saturated1D(const float* __restrict      src, 
//                              uint8_t* __restrict          dst, 
//                              const uint64_t               proc_len_v)
// {
//     decx::utils::simd::xmm256_reg recv, _crit, store;

//     for (int i = 0; i < proc_len_v; ++i) {
//         recv._vf = _mm256_load_ps(src + i * 8);

//         _crit._vf = _mm256_cmp_ps(recv._vf, _mm256_set1_ps(0), _CMP_GT_OS);
//         store._vi = _mm256_and_si256(recv._vi, _crit._vi);

//         store._vi = _mm256_cvtps_epi32(store._vf);
//         _crit._vf = _mm256_cmp_ps(recv._vf, _mm256_set1_ps(255), _CMP_GT_OS);     // saturated_cast
//         store._vi = _mm256_or_si256(store._vi, _crit._vi);       // saturated_cast
//         store._vi = _mm256_shuffle_epi8(_mm256_and_si256(store._vi, _mm256_set1_epi32(0x000000ff)), _mm256_set1_epi32(201851904));

// #ifdef _MSC_VER
//         dst[i * 2] = store._vi.m256i_i32[0];
//         dst[i * 2 + 1] = store._vi.m256i_i32[4];
// #endif

// #ifdef __GNUC__
//         *((int32_t*)(dst + i * 8)) = ((int*)&store._vi)[0];
//         *((int32_t*)(dst + i * 8 + 4)) = ((int*)&store._vi)[4];
// #endif
//     }
// }


// _DECX_API_ void de::cpu::test_type_cast(de::Matrix& src, de::Matrix& dst)
// {
//     decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);
//     decx::_Matrix* _dst = dynamic_cast<decx::_Matrix*>(&dst);

//     decx::utils::_thr_1D t1D(12);
//     decx::cpu_ElementWise2D_planner _planner;
//     _planner.plan(decx::cpu::_get_permitted_concurrency(), make_uint2(_src->Width(), _src->Height()), sizeof(float), sizeof(uint8_t));

//     _planner.caller(_v256_cvtf32_ui8_saturated2D, (float*)_src->Mat.ptr, (uint8_t*)_dst->Mat.ptr, _src->Pitch(), _dst->Pitch(), &t1D);


// }



// _DECX_API_ void de::cpu::test_type_cast(de::Vector& src, de::Vector& dst)
// {
//     decx::_Vector* _src = dynamic_cast<decx::_Vector*>(&src);
//     decx::_Vector* _dst = dynamic_cast<decx::_Vector*>(&dst);

//     decx::utils::_thr_1D t1D(12);
//     decx::cpu_ElementWise1D_planner _planner;
//     _planner.plan(decx::cpu::_get_permitted_concurrency(), _src->Len(), sizeof(float), sizeof(uint8_t));

//     _planner.caller(_v256_cvtf32_ui8_saturated1D, (float*)_src->Vec.ptr, (uint8_t*)_dst->Vec.ptr, &t1D);


// }