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


#include "cpl32_extract_exec.h"


_THREAD_FUNCTION_ void
decx::dsp::CPUK::_module_fp32_ST2D(const double* __restrict    src, 
                                    float* __restrict           dst, 
                                    const uint2                 _proc_dims, 
                                    const uint64_t              Wsrc, 
                                    const uint64_t              Wdst)       // 4x
{
    __m256 _recv, tmp1;
    __m256 tmp2;
    const __m256i _shufflevar = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);

    uint64_t dex_src = 0, dex_dst = 0;

    for (uint32_t i = 0; i < _proc_dims.y; ++i) 
    {
        dex_src = i * Wsrc;
        dex_dst = i * Wdst;
        for (uint32_t j = 0; j < _proc_dims.x; ++j) {
            _recv = _mm256_castpd_ps(_mm256_load_pd(src + dex_src));
            tmp1 = _mm256_mul_ps(_recv, _recv);
            tmp1 = _mm256_sqrt_ps(_mm256_hadd_ps(tmp1, tmp1));
            tmp2 = _mm256_permutevar8x32_ps(tmp1, _shufflevar);
            _mm_store_ps(dst + dex_dst, _mm256_castps256_ps128(tmp2));

            dex_src += 4;
            dex_dst += 4;
        }
    }
}



_THREAD_FUNCTION_ void
decx::dsp::CPUK::_angle_fp32_ST2D(const double* __restrict    src, 
                                    float* __restrict           dst, 
                                    const uint2                 _proc_dims, 
                                    const uint64_t              Wsrc, 
                                    const uint64_t              Wdst)       // 4x
{
    __m256 _recv, tmp1;
    __m256 tmp2;
    const __m256i _shufflevar = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);

    uint64_t dex_src = 0, dex_dst = 0;

    for (uint32_t i = 0; i < _proc_dims.y; ++i)
    {
        dex_src = i * Wsrc;
        dex_dst = i * Wdst;
        for (uint32_t j = 0; j < _proc_dims.x; ++j) {
            _recv = _mm256_castpd_ps(_mm256_load_pd(src + dex_src));
            tmp1 = _mm256_permute_ps(_recv, 0b10110001);
#ifdef _MSC_VER
            tmp1 = _mm256_atan2_ps(tmp1, _recv);
#endif
#ifdef __GNUC__
            tmp1 = decx::utils::simd::_mm256_atan2_ps(tmp1, _recv);
#endif
            tmp2 = _mm256_permutevar8x32_ps(tmp1, _shufflevar);
            _mm_store_ps(dst + dex_dst, _mm256_castps256_ps128(tmp2));

            dex_src += 4;
            dex_dst += 4;
        }
    }
}



void decx::dsp::_cpl32_extract_caller(const de::CPf* src, float* dst, const uint2 _proc_dims,
    const uint64_t Wsrc, const uint64_t Wdst, decx::dsp::CPUK::cpl32_extract_kernel2D kernel)
{
    if (kernel != NULL) 
    {
        uint32_t conc_thr = decx::cpu::_get_permitted_concurrency();
        decx::utils::frag_manager f_mgr;
        decx::utils::frag_manager_gen(&f_mgr, _proc_dims.y, conc_thr);

        decx::utils::_thread_arrange_1D t1D(conc_thr);

        const double* loc_src = (double*)src;
        float* loc_dst = dst;

        uint2 _frag_proc_dims = make_uint2(_proc_dims.x, f_mgr.frag_len);
        const uint64_t frag_src = Wsrc * f_mgr.frag_len;
        const uint64_t frag_dst = Wdst * f_mgr.frag_len;

        for (int i = 0; i < conc_thr - 1; ++i) {
            t1D._async_thread[i] = decx::cpu::register_task_default(
                kernel, loc_src, loc_dst, _frag_proc_dims, Wsrc, Wdst);

            loc_src += frag_src;
            loc_dst += frag_dst;
        }
        const size_t L_proc = f_mgr.is_left ? f_mgr.frag_left_over : f_mgr.frag_len;
        _frag_proc_dims.y = L_proc;
        t1D._async_thread[conc_thr - 1] = decx::cpu::register_task_default(
            kernel, loc_src, loc_dst, _frag_proc_dims, Wsrc, Wdst);

        t1D.__sync_all_threads();
    }
}