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


_THREAD_FUNCTION_ void decx::CPUK::
gather2D_fp32_exec_bilinear(const float* src_head_ptr,  const float2* __restrict map,
                            float* __restrict dst,      const uint2 proc_dims_v, 
                            const uint32_t Wsrc_v1,     const uint32_t Wmap_v1, 
                            const uint32_t Wdst_v1,     void* _addr_info)
{
    uint64_t dex_map = 0, dex_dst = 0;

    auto* p_addr_mgr = (decx::CPUK::VGT_bilinear_addr_mgr*)_addr_info;
    
    for (int32_t i = 0; i < proc_dims_v.y; ++i)
    {
        dex_map = i * Wmap_v1;
        dex_dst = i * Wdst_v1;
        for (int32_t j = 0; j < proc_dims_v.x; ++j)
        {
            __m256 map_lane1 = _mm256_load_ps((float*)(map + dex_map));
            __m256 map_lane2 = _mm256_load_ps((float*)(map + dex_map + 4));
            p_addr_mgr->plan(map_lane1, map_lane2);

            // Left-Top
            __m256 T = _mm256_i32gather_ps(src_head_ptr, p_addr_mgr->get_addr0(), 4);
            // Right-Top
            __m256 tmp = _mm256_i32gather_ps(src_head_ptr, p_addr_mgr->get_addr1(), 4);
            __m256 dist_down_x = p_addr_mgr->get_dist_down_X();
            // Interpolation along X axis
            T = _mm256_mul_ps(T, p_addr_mgr->get_dist_up_X());
            T = _mm256_fmadd_ps(tmp, dist_down_x, T);

            __m256 B = _mm256_i32gather_ps(src_head_ptr, p_addr_mgr->get_addr2(), 4);
            // Right-Top
            tmp = _mm256_i32gather_ps(src_head_ptr, p_addr_mgr->get_addr3(), 4);
            // Interpolation along X axis
            B = _mm256_mul_ps(B, p_addr_mgr->get_dist_up_X());
            B = _mm256_fmadd_ps(tmp, dist_down_x, B);

            __m256 res = _mm256_mul_ps(T, p_addr_mgr->get_dist_up_Y());
            res = _mm256_fmadd_ps(B, p_addr_mgr->get_dist_down_Y(), res);
            
            res = _mm256_andnot_ps(p_addr_mgr->get_current_inbound_fp32(), res);
            _mm256_store_ps(dst + dex_dst, res);

            dex_map += 8;
            dex_dst += 8;
        }
    }
}



_THREAD_FUNCTION_ void decx::CPUK::
gather2D_fp32_exec_nearest(const float* src_head_ptr,  const float2* __restrict map,
                           float* __restrict dst,      const uint2 proc_dims_v, 
                           const uint32_t Wsrc_v1,     const uint32_t Wmap_v1, 
                           const uint32_t Wdst_v1,     void* _addr_info)
{
    uint64_t dex_map = 0, dex_dst = 0;
    
    auto* p_addr_mgr = (decx::CPUK::VGT_nearest_addr_mgr*)_addr_info;

    for (int32_t i = 0; i < proc_dims_v.y; ++i)
    {
        dex_map = i * Wmap_v1;
        dex_dst = i * Wdst_v1;
        for (int32_t j = 0; j < proc_dims_v.x; ++j)
        {
            __m256 map_lane1 = _mm256_load_ps((float*)(map + dex_map));
            __m256 map_lane2 = _mm256_load_ps((float*)(map + dex_map + 4));
            p_addr_mgr->plan(map_lane1, map_lane2);

            __m256 res = _mm256_i32gather_ps(src_head_ptr, p_addr_mgr->get_addr(), 4);

            res = _mm256_andnot_ps(p_addr_mgr->get_current_inbound_fp32(), res);
            _mm256_store_ps(dst + dex_dst, res);

            dex_map += 8;
            dex_dst += 8;
        }
    }
}
