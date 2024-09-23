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

#include "gather_kernels.h"

#if 0
_THREAD_FUNCTION_ void decx::CPUK::
gather2D_uint8_exec_bilinear(const uint8_t* src_head_ptr,  const float2* __restrict map,
                            uint8_t* __restrict dst,      const uint2 proc_dims_v, 
                            const uint32_t Wsrc_v1,     const uint32_t Wmap_v1, 
                            const uint32_t Wdst_v1,     decx::CPUK::VGT_addr_mgr* _addr_info)
{
    uint64_t dex_map = 0, dex_dst = 0;
    
    for (int32_t i = 0; i < proc_dims_v.y; ++i)
    {
        dex_map = i * Wmap_v1;
        dex_dst = i * Wdst_v1;
        for (int32_t j = 0; j < proc_dims_v.x; ++j)
        {
            __m256 map_lane1 = _mm256_load_ps((float*)(map + dex_map));
            __m256 map_lane2 = _mm256_load_ps((float*)(map + dex_map + 4));

            _addr_info->plan(map_lane1, map_lane2);

            decx::utils::simd::xmm256_reg row, row_R, res;
            __m256 T, B;
            __m256 dist_down_x = _addr_info->get_dist_down_X();

            row._vi = _mm256_i32gather_epi32((int32_t*)src_head_ptr, _addr_info->get_addr0(), 1);
            row_R._vi = _mm256_and_si256(row._vi, _mm256_set1_epi32(0xFF00));
            row_R._vi = _mm256_srli_epi32(row_R._vi, 8);
            row._vi = _mm256_and_si256(row._vi, _mm256_set1_epi32(0xFF));

            row._vf = _mm256_cvtepi32_ps(row._vi);
            row_R._vf = _mm256_cvtepi32_ps(row_R._vi);

            T = _mm256_mul_ps(row._vf, _addr_info->get_dist_up_X());
            T = _mm256_fmadd_ps(row_R._vf, dist_down_x, T);
            
            // Right-Top
            row._vi = _mm256_i32gather_epi32((int32_t*)src_head_ptr, _addr_info->get_addr2(), 1);
            row_R._vi = _mm256_and_si256(row._vi, _mm256_set1_epi32(0xFF00));
            row_R._vi = _mm256_srli_epi32(row_R._vi, 8);
            row._vi = _mm256_and_si256(row._vi, _mm256_set1_epi32(0xFF));

            row._vf = _mm256_cvtepi32_ps(row._vi);
            row_R._vf = _mm256_cvtepi32_ps(row_R._vi);

            B = _mm256_mul_ps(row._vf, _addr_info->get_dist_up_X());
            B = _mm256_fmadd_ps(row_R._vf, dist_down_x, B);

            res._vf = _mm256_mul_ps(T, _addr_info->get_dist_up_Y());
            res._vf = _mm256_fmadd_ps(B, _addr_info->get_dist_down_Y(), res._vf);
            
            res._vf = _mm256_andnot_ps(_addr_info->get_current_inbound_fp32(), res._vf);

            res._vi = _mm256_cvtps_epi32(res._vf);
            res._vi = _mm256_shuffle_epi8(_mm256_and_si256(res._vi, _mm256_set1_epi32(0x000000ff)), _mm256_set1_epi32(201851904));

            *((uint32_t*)(dst + dex_dst)) = res._arrui[0];
            *((uint32_t*)(dst + dex_dst + 4)) = res._arrui[4];

            dex_map += 8;
            dex_dst += 8;
        }
    }
}


#else

_THREAD_FUNCTION_ void decx::CPUK::
gather2D_uint8_exec_bilinear(const uint8_t* src_head_ptr,  const float2* __restrict map,
                            uint8_t* __restrict dst,      const uint2 proc_dims_v, 
                            const uint32_t Wsrc_v1,     const uint32_t Wmap_v1, 
                            const uint32_t Wdst_v1,     decx::CPUK::VGT_addr_mgr* _addr_info)
{
    uint64_t dex_map = 0, dex_dst = 0;
    
    for (int32_t i = 0; i < proc_dims_v.y; ++i)
    {
        dex_map = i * Wmap_v1;
        dex_dst = i * Wdst_v1;
        for (int32_t j = 0; j < proc_dims_v.x; ++j)
        {
            __m256 map_lane1 = _mm256_load_ps((float*)(map + dex_map));
            __m256 map_lane2 = _mm256_load_ps((float*)(map + dex_map + 4));

            _addr_info->plan(map_lane1, map_lane2);

            decx::utils::simd::xmm256_reg row, row_R, res;
            __m256i T, B;
            decx::utils::simd::xmm256_reg dist_up, dist_down;
            dist_up._vf = _addr_info->get_dist_up_X();
            dist_up._vi = _mm256_add_epi32(dist_up._vi, _mm256_set1_epi32(0x4000000U));
            dist_up._vi = _mm256_cvtps_epi32(dist_up._vf);
            dist_down._vi = _mm256_sub_epi32(_mm256_set1_epi32(256), dist_up._vi);

            row._vi = _mm256_i32gather_epi32((int32_t*)src_head_ptr, _addr_info->get_addr0(), 1);
            row_R._vi = _mm256_and_si256(row._vi, _mm256_set1_epi32(0xFF00));
            row_R._vi = _mm256_srli_epi32(row_R._vi, 8);
            row._vi = _mm256_and_si256(row._vi, _mm256_set1_epi32(0xFF));

            T = _mm256_mullo_epi16(row._vi, dist_up._vi);
            T = _mm256_add_epi32(_mm256_mullo_epi16(row_R._vi, dist_down._vi), T);
            
            // Right-Top
            row._vi = _mm256_i32gather_epi32((int32_t*)src_head_ptr, _addr_info->get_addr2(), 1);
            row_R._vi = _mm256_and_si256(row._vi, _mm256_set1_epi32(0xFF00));
            row_R._vi = _mm256_srli_epi32(row_R._vi, 8);
            row._vi = _mm256_and_si256(row._vi, _mm256_set1_epi32(0xFF));

            B = _mm256_mullo_epi16(row._vi, dist_up._vi);
            B = _mm256_add_epi32(_mm256_mullo_epi16(row_R._vi, dist_down._vi), B);

            dist_up._vf = _addr_info->get_dist_up_Y();
            dist_up._vi = _mm256_add_epi32(dist_up._vi, _mm256_set1_epi32(0x4000000U));
            dist_up._vi = _mm256_cvtps_epi32(dist_up._vf);
            dist_down._vi = _mm256_sub_epi32(_mm256_set1_epi32(256), dist_up._vi);

            T = _mm256_srli_epi32(T, 8);
            B = _mm256_srli_epi32(B, 8);

            res._vi = _mm256_mullo_epi16(T, dist_up._vi);
            res._vi = _mm256_add_epi32(_mm256_mullo_epi16(B, dist_down._vi), res._vi);
            
            res._vf = _mm256_andnot_ps(_addr_info->get_current_inbound_fp32(), res._vf);

            res._vi = _mm256_shuffle_epi8(_mm256_and_si256(res._vi, _mm256_set1_epi32(0x0000ff00)), _mm256_set1_epi32(0x0D090501));

            *((uint32_t*)(dst + dex_dst)) = res._arrui[0];
            *((uint32_t*)(dst + dex_dst + 4)) = res._arrui[4];

            dex_map += 8;
            dex_dst += 8;
        }
    }
}
#endif