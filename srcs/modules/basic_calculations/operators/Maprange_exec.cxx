/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#include "Maprange_exec.h"



_THREAD_FUNCTION_ void
decx::calc::CPUK::maprange_fvec8_ST(const float* __restrict src,
                                    float* __restrict dst, 
                                    uint64_t len, 
                                    const float2 _min_max, 
                                    const float2 _dst_range)
{
    __m256 _recv, _res;
    const __m256 _scale_fac = _mm256_set1_ps(fabs(_dst_range.y - _dst_range.x) / fabs(_min_max.y - _min_max.x));

    for (uint i = 0; i < len; ++i) {
        _recv = _mm256_load_ps(src + (i << 3));

        _recv = _mm256_sub_ps(_recv, _mm256_set1_ps(_min_max.x));
        _res = _mm256_mul_ps(_recv, _scale_fac);

        _mm256_store_ps(dst + (i << 3), _res);
    }
}




_THREAD_FUNCTION_ void
decx::calc::CPUK::maprange2D_cvtf32_u8vec8_ST(const float* __restrict   src, 
                                              double* __restrict        dst, 
                                              uint32_t                  pitchsrc_v1, 
                                              uint32_t                  pitchdst_v8, 
                                              const uint2               proc_dims_v1, 
                                              const float2              _min_max, 
                                              const float2              _dst_range)
{
    __m256 _recv;
    decx::utils::simd::xmm256_reg _reg;
    double _pixels_IO_v8;
    const __m256 _scale_fac = _mm256_set1_ps(fabs(_dst_range.y - _dst_range.x) / fabs(_min_max.y - _min_max.x));

    uint64_t dex_src = 0, dex_dst = 0;

    for (uint32_t i = 0; i < proc_dims_v1.y; ++i)
    {
        dex_src = i * pitchsrc_v1;
        dex_dst = i * pitchdst_v8;
        for (uint32_t j = 0; j < decx::utils::ceil<uint32_t>(proc_dims_v1.x, 8); ++j) {
            _recv = _mm256_load_ps(src + dex_src);

            _recv = _mm256_sub_ps(_recv, _mm256_set1_ps(_min_max.x));
            _reg._vf = _mm256_mul_ps(_recv, _scale_fac);

            _reg._vi = _mm256_cvtps_epi32(_reg._vf);
            _reg._vi = _mm256_shuffle_epi8(_reg._vi, _mm256_setr_epi32(0x0C080400, 0, 0, 0, 0x0C080400, 0, 0, 0));
            _reg._vi = _mm256_permutevar8x32_epi32(_reg._vi, _mm256_setr_epi32(0, 4, 0, 0, 0, 0, 0, 0));
            _pixels_IO_v8 = *((double*)&_reg._vi);

            dst[dex_dst] = _pixels_IO_v8;
            dex_src += 8;
            ++dex_dst;
        }
    }
}



void
decx::calc::maprange2D_cvtf32_u8_caller(const float* __restrict   src, 
                                        double* __restrict        dst, 
                                        uint32_t                  pitchsrc_v1, 
                                        uint32_t                  pitchdst_v8, 
                                        const uint2               proc_dims_v1, 
                                        const float2              _min_max, 
                                        const float2              _dst_range)
{
    decx::utils::_thread_arrange_1D t1D(decx::cpu::_get_permitted_concurrency());
    decx::utils::frag_manager f_mgr;
    decx::utils::frag_manager_gen(&f_mgr, proc_dims_v1.y, t1D.total_thread);

    const float* _loc_src = src;
    double* _loc_dst = dst;
    for (uint32_t i = 0; i < t1D.total_thread - 1; ++i) {
        t1D._async_thread[i] = decx::cpu::register_task_default(decx::calc::CPUK::maprange2D_cvtf32_u8vec8_ST,
            _loc_src, _loc_dst, pitchsrc_v1, pitchdst_v8, make_uint2(proc_dims_v1.x, f_mgr.frag_len), _min_max, _dst_range);

        _loc_src += f_mgr.frag_len * pitchsrc_v1;
        _loc_dst += f_mgr.frag_len * pitchdst_v8;
    }
    const uint32_t _L = f_mgr.is_left ? f_mgr.frag_left_over : f_mgr.frag_len;
    t1D._async_thread[t1D.total_thread - 1] = decx::cpu::register_task_default(decx::calc::CPUK::maprange2D_cvtf32_u8vec8_ST,
        _loc_src, _loc_dst, pitchsrc_v1, pitchdst_v8, make_uint2(proc_dims_v1.x, _L), _min_max, _dst_range);

    t1D.__sync_all_threads();
}