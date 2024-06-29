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


#include "integral.h"



_THREAD_FUNCTION_ void
decx::calc::CPUK::_integral_V_uint8_2D_ST(float* __restrict src, 
                                         const uint2 proc_dims,     // ~.x -> in __m256
                                         const uint pitch)          // in float
{
    size_t dex;
    __m256i buffer = _mm256_set1_epi32(0), reg = _mm256_set1_epi32(0);
    for (int i = 0; i < proc_dims.x; ++i) {
        dex = ((size_t)i << 3);
        buffer = _mm256_castps_si256(_mm256_load_ps(src + dex));
        dex += (size_t)pitch;
        for (int j = 1; j < proc_dims.y; ++j) {
            reg = _mm256_castps_si256(_mm256_load_ps(src + dex));
            buffer = _mm256_add_epi32(reg, buffer);
            _mm256_store_ps(src + dex, _mm256_castsi256_ps(buffer));
            dex += (size_t)pitch;
        }
    }
}


_THREAD_FUNCTION_ void
decx::calc::CPUK::_integral_V_uint8_f32_2D_ST(float* __restrict src, 
                                         const uint2 proc_dims,     // ~.x -> in __m256
                                         const uint pitch)          // in float
{
    size_t dex;
    decx::utils::simd::_mmv256 buffer, reg;
    buffer._vi = _mm256_set1_epi32(0);
    reg._vi = _mm256_set1_epi32(0);
    for (int i = 0; i < proc_dims.x; ++i) {
        dex = ((size_t)i << 3);
        buffer._vi = _mm256_castps_si256(_mm256_load_ps(src + dex));
        reg._vf = _mm256_cvtepi32_ps(buffer._vi);
        _mm256_store_ps(src + dex, reg._vf);

        dex += (size_t)pitch;
        for (int j = 1; j < proc_dims.y; ++j) {
            reg._vi = _mm256_castps_si256(_mm256_load_ps(src + dex));
            buffer._vi = _mm256_add_epi32(reg._vi, buffer._vi);
            _mm256_store_ps(src + dex, _mm256_cvtepi32_ps(buffer._vi));
            dex += (size_t)pitch;
        }
    }
}



_THREAD_FUNCTION_ void
decx::calc::CPUK::_integral_H_uint8_2D_ST(const uint64_t* __restrict src,
                                          float* __restrict dst, 
                                          const uint2 proc_dims,        // ~.x -> 8x
                                          const uint pitchsrc,          // 8x
                                          const uint pitchdst)          // 1x
{
    size_t dex_src, dex_dst;
    uint64_t recv;
    __m256i reg1, reg2, local_sums,
        prev_sum = _mm256_set1_epi32(0);

    const __m256i maskL = _mm256_setr_epi64x(0x00000000000000FF, 0x000000000000FFFF, 
                                             0x0000000000FFFFFF, 0x00000000FFFFFFFF),
                  maskR = _mm256_setr_epi64x(0x000000FFFFFFFFFF, 0x0000FFFFFFFFFFFF, 
                                             0x00FFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF);
    const __m256i zeros = _mm256_set1_epi32(0);

    for (int i = 0; i < proc_dims.y; ++i) 
    {
        dex_src = (size_t)i * (size_t)pitchsrc;
        dex_dst = (size_t)i * (size_t)pitchdst;
        
        recv = src[dex_src];
        reg1 = _mm256_set1_epi64x(recv);
        // integrating within (addr 0 ~ 3)
        reg2 = _mm256_and_si256(reg1, maskL);
        reg2 = _mm256_sad_epu8(reg2, zeros);
        local_sums = _mm256_permutevar8x32_epi32(reg2, _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7));

        // integrating within (addr 4 ~ 7)
        reg2 = _mm256_and_si256(reg1, maskR);
        reg2 = _mm256_sad_epu8(reg2, zeros);
        reg2 = _mm256_permutevar8x32_epi32(reg2, _mm256_setr_epi32(1, 3, 5, 7, 0, 2, 4, 6));
        local_sums = _mm256_blend_epi32(local_sums, reg2, 0b11110000);

        _mm256_store_ps(dst + dex_dst, _mm256_castsi256_ps(local_sums));

        ++dex_src;
        dex_dst += 8;

        prev_sum = _mm256_permutevar8x32_epi32(local_sums, _mm256_setr_epi32(7, 7, 7, 7, 7, 7, 7, 7));

        for (int j = 1; j < proc_dims.x; ++j) {
            recv = src[dex_src];
            reg1 = _mm256_set1_epi32(recv);
            // integrating within (addr 0 ~ 3)
            reg2 = _mm256_and_si256(reg1, maskL);
            reg2 = _mm256_sad_epu8(reg2, zeros);
            local_sums = _mm256_permutevar8x32_epi32(reg2, _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7));

            // integrating within (addr 4 ~ 7)
            reg2 = _mm256_and_si256(reg1, maskR);
            reg2 = _mm256_sad_epu8(reg2, zeros);
            reg2 = _mm256_permutevar8x32_epi32(reg2, _mm256_setr_epi32(1, 3, 5, 7, 0, 2, 4, 6));
            local_sums = _mm256_blend_epi32(local_sums, reg2, 0b11110000);
            local_sums = _mm256_add_epi32(local_sums, prev_sum);

            _mm256_store_ps(dst + dex_dst, _mm256_castsi256_ps(local_sums));

            prev_sum = _mm256_permutevar8x32_epi32(local_sums, _mm256_setr_epi32(7, 7, 7, 7, 7, 7, 7, 7));

            ++dex_src;
            dex_dst += 8;
        }
    }
}



_DECX_API_
void decx::calc::_integral_caller2D_uint8(const uint8_t* src, int32_t* dst, const uint2 proc_dims,
    const uint pitchsrc, const uint pitchdst)
{
    decx::utils::_thread_arrange_1D t1D(decx::cpu::_get_permitted_concurrency());
    decx::utils::frag_manager f_mgrH, f_mgrV;
    decx::utils::frag_manager_gen(&f_mgrH, proc_dims.y, t1D.total_thread);
    decx::utils::frag_manager_gen(&f_mgrV, pitchdst / 8, t1D.total_thread);

    const size_t frag_size_src = (size_t)pitchsrc / 8 * (size_t)f_mgrH.frag_len;
    const size_t frag_size_dst = (size_t)pitchdst * (size_t)f_mgrH.frag_len;

    uint2 frag_proc_dims = make_uint2(decx::utils::ceil<uint>(proc_dims.x, 8), proc_dims.y / t1D.total_thread);

    const uint64_t* src_ptr = (uint64_t*)src;
    float* dst_ptr = (float*)dst;
    // integral horizentally
    for (int i = 0; i < t1D.total_thread - 1; ++i) {
        t1D._async_thread[i] = decx::cpu::register_task_default( decx::calc::CPUK::_integral_H_uint8_2D_ST,
            src_ptr, dst_ptr, frag_proc_dims, pitchsrc / 8, pitchdst);
        src_ptr += frag_size_src;
        dst_ptr += frag_size_dst;
    }
    frag_proc_dims.y = proc_dims.y - (t1D.total_thread - 1) * frag_proc_dims.y;

    t1D._async_thread[t1D.total_thread - 1] = decx::cpu::register_task_default(
        decx::calc::CPUK::_integral_H_uint8_2D_ST, src_ptr, dst_ptr,
        frag_proc_dims, pitchsrc / 8, pitchdst);

    t1D.__sync_all_threads();

    // integral vertically
    dst_ptr = (float*)dst;
    for (int i = 0; i < t1D.total_thread - 1; ++i) {
        t1D._async_thread[i] = decx::cpu::register_task_default( decx::calc::CPUK::_integral_V_uint8_2D_ST,
            dst_ptr, make_uint2(f_mgrV.frag_len, proc_dims.y), pitchdst);
        dst_ptr += (size_t)f_mgrV.frag_len << 3;
    }
    frag_proc_dims = make_uint2(f_mgrV.is_left ? f_mgrV.frag_left_over : f_mgrV.frag_len, proc_dims.y);
    t1D._async_thread[t1D.total_thread - 1] = decx::cpu::register_task_default( decx::calc::CPUK::_integral_V_uint8_2D_ST,
        dst_ptr, frag_proc_dims, pitchdst);

    t1D.__sync_all_threads();
}



_DECX_API_
void decx::calc::_integral_caller2D_uint8_f32(const uint8_t* src, float* dst, const uint2 proc_dims,
    const uint pitchsrc, const uint pitchdst)
{
    decx::utils::_thread_arrange_1D t1D(decx::cpu::_get_permitted_concurrency());
    decx::utils::frag_manager f_mgrH, f_mgrV;
    decx::utils::frag_manager_gen(&f_mgrH, proc_dims.y, t1D.total_thread);
    decx::utils::frag_manager_gen(&f_mgrV, pitchdst / 8, t1D.total_thread);

    const size_t frag_size_src = (size_t)pitchsrc / 8 * (size_t)f_mgrH.frag_len;
    const size_t frag_size_dst = (size_t)pitchdst * (size_t)f_mgrH.frag_len;

    uint2 frag_proc_dims = make_uint2(decx::utils::ceil<uint>(proc_dims.x, 8), proc_dims.y / t1D.total_thread);

    const uint64_t* src_ptr = (uint64_t*)src;
    float* dst_ptr = (float*)dst;
    // integral horizentally
    for (int i = 0; i < t1D.total_thread - 1; ++i) {
        t1D._async_thread[i] = decx::cpu::register_task_default( decx::calc::CPUK::_integral_H_uint8_2D_ST,
            src_ptr, dst_ptr, frag_proc_dims, pitchsrc / 8, pitchdst);
        src_ptr += frag_size_src;
        dst_ptr += frag_size_dst;
    }
    frag_proc_dims.y = proc_dims.y - (t1D.total_thread - 1) * frag_proc_dims.y;

    t1D._async_thread[t1D.total_thread - 1] = decx::cpu::register_task_default(
        decx::calc::CPUK::_integral_H_uint8_2D_ST, src_ptr, dst_ptr,
        frag_proc_dims, pitchsrc / 8, pitchdst);

    t1D.__sync_all_threads();

    // integral vertically
    dst_ptr = (float*)dst;
    for (int i = 0; i < t1D.total_thread - 1; ++i) {
        t1D._async_thread[i] = decx::cpu::register_task_default( decx::calc::CPUK::_integral_V_uint8_f32_2D_ST,
            dst_ptr, make_uint2(f_mgrV.frag_len, proc_dims.y), pitchdst);
        dst_ptr += (size_t)f_mgrV.frag_len << 3;
    }
    frag_proc_dims = make_uint2(f_mgrV.is_left ? f_mgrV.frag_left_over : f_mgrV.frag_len, proc_dims.y);
    t1D._async_thread[t1D.total_thread - 1] = decx::cpu::register_task_default( decx::calc::CPUK::_integral_V_uint8_f32_2D_ST,
        dst_ptr, frag_proc_dims, pitchdst);

    t1D.__sync_all_threads();
}