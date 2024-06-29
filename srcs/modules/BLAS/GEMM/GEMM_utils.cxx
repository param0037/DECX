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


#include "GEMM_utils.h"


void decx::gemm::CPUK::GEMM_fp32_cpy_L8(const float* src, float* dst, const uint Wsrc, const uint Wdst, const uint2 cpy_dim)
{
    size_t dex_src = 0, dex_dst = 0, tmp_dex_src = 0, tmp_dex_dst = 0;
    for (int i = 0; i < cpy_dim.y; ++i) {
        tmp_dex_src = dex_src;
        tmp_dex_dst = dex_dst;
        for (int j = 0; j < cpy_dim.x; ++j) {
            _mm256_store_ps(dst + tmp_dex_dst, _mm256_load_ps(src + dex_src));
            tmp_dex_src += 8;
            tmp_dex_dst += 8;
        }
        dex_src += Wsrc;
        dex_dst += Wdst;
    }
}


void decx::gemm::CPUK::GEMM_fp64_cpy_L8(const double* src, double* dst, const uint Wsrc, const uint Wdst, const uint2 cpy_dim)
{
    size_t dex_src = 0, dex_dst = 0, tmp_dex_src = 0, tmp_dex_dst = 0;
    for (int i = 0; i < cpy_dim.y; ++i) {
        tmp_dex_src = dex_src;
        tmp_dex_dst = dex_dst;
        for (int j = 0; j < cpy_dim.x; ++j) {
            _mm256_store_pd(dst + tmp_dex_dst, _mm256_load_pd(src + dex_src));
            tmp_dex_src += 4;
            tmp_dex_dst += 4;
        }
        dex_src += Wsrc;
        dex_dst += Wdst;
    }
}