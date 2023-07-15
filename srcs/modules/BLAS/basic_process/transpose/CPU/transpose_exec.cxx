/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#include "transpose_exec.h"



_THREAD_FUNCTION_ void
decx::bp::CPUK::transpose_4x4_b32(const float* __restrict src, float* __restrict dst,
    const uint2 proc_dims_src, const uint Wsrc, const uint Wdst, const uint _LW)
{
    __m128 recv[4], reg[4];

    size_t dex_src = 0, dex_dst = 0;

    for (int i = 0; i < proc_dims_src.y; ++i) {
        dex_src = i * 4 * Wsrc;
        dex_dst = i * 4;
        for (int j = 0; j < proc_dims_src.x; ++j) {
            recv[0] = _mm_load_ps(src + dex_src);
            recv[1] = _mm_load_ps(src + dex_src + Wsrc);
            recv[2] = _mm_load_ps(src + dex_src + (Wsrc << 1));
            recv[3] = _mm_load_ps(src + dex_src + (Wsrc * 3));

            dex_src += 4;

            _AVX_MM128_TRANSPOSE_4X4_(recv, reg);

            _mm_store_ps(dst + dex_dst, recv[0]);   dex_dst += Wdst;
            _mm_store_ps(dst + dex_dst, recv[1]);   dex_dst += Wdst;
            _mm_store_ps(dst + dex_dst, recv[2]);   dex_dst += Wdst;
            _mm_store_ps(dst + dex_dst, recv[3]);   dex_dst += Wdst;
        }

        recv[0] = _mm_set1_ps(0);       recv[1] = _mm_set1_ps(0);
        recv[2] = _mm_set1_ps(0);       recv[3] = _mm_set1_ps(0);

        if (_LW > 0) {
            recv[0] = _mm_load_ps(src + dex_src);
            recv[1] = _mm_load_ps(src + dex_src + Wsrc);
            recv[2] = _mm_load_ps(src + dex_src + (Wsrc << 1));
            recv[3] = _mm_load_ps(src + dex_src + (Wsrc * 3));

            dex_src += 4;

            _AVX_MM128_TRANSPOSE_4X4_(recv, reg);

            for (int k = 0; k < _LW; ++k) {
                _mm_store_ps(dst + dex_dst, recv[k]);
                dex_dst += Wdst;
            }
        }
    }
}



_THREAD_FUNCTION_ void
decx::bp::CPUK::transpose_4x4_b32_LH(const float* __restrict src, float* __restrict dst,
    const uint2 proc_dims_src, const uint Wsrc, const uint Wdst, const uint _LW, const uint _LH)
{
    __m128 recv[4], reg[4];

    size_t dex_src = 0, dex_dst = 0;

    for (int i = 0; i < proc_dims_src.y; ++i) {
        dex_src = i * 4 * Wsrc;
        dex_dst = i * 4;
        for (int j = 0; j < proc_dims_src.x; ++j) {
            recv[0] = _mm_load_ps(src + dex_src);
            recv[1] = _mm_load_ps(src + dex_src + Wsrc);
            recv[2] = _mm_load_ps(src + dex_src + (Wsrc << 1));
            recv[3] = _mm_load_ps(src + dex_src + (Wsrc * 3));

            dex_src += 4;

            _AVX_MM128_TRANSPOSE_4X4_(recv, reg);

            _mm_store_ps(dst + dex_dst, recv[0]);   dex_dst += Wdst;
            _mm_store_ps(dst + dex_dst, recv[1]);   dex_dst += Wdst;
            _mm_store_ps(dst + dex_dst, recv[2]);   dex_dst += Wdst;
            _mm_store_ps(dst + dex_dst, recv[3]);   dex_dst += Wdst;
        }
        recv[0] = _mm_load_ps(src + dex_src);
        recv[1] = _mm_load_ps(src + dex_src + Wsrc);
        recv[2] = _mm_load_ps(src + dex_src + (Wsrc << 1));
        recv[3] = _mm_load_ps(src + dex_src + (Wsrc * 3));

        dex_src += 4;

        _AVX_MM128_TRANSPOSE_4X4_(recv, reg);

        for (int k = 0; k < _LW; ++k) {
            _mm_store_ps(dst + dex_dst, recv[k]);
            dex_dst += Wdst;
        }
    }

    // _leftovers
    dex_src = proc_dims_src.y * 4 * Wsrc;
    dex_dst = proc_dims_src.y * 4;
    if (_LH > 0) {
        for (int j = 0; j < proc_dims_src.x; ++j) {

            recv[0] = _mm_set1_ps(0);       recv[1] = _mm_set1_ps(0);
            recv[2] = _mm_set1_ps(0);       recv[3] = _mm_set1_ps(0);

            for (int k = 0; k < _LH; ++k) {
                recv[k] = _mm_load_ps(src + dex_src + Wsrc * k);
            }

            dex_src += 4;

            _AVX_MM128_TRANSPOSE_4X4_(recv, reg);

            _mm_store_ps(dst + dex_dst, recv[0]);   dex_dst += Wdst;
            _mm_store_ps(dst + dex_dst, recv[1]);   dex_dst += Wdst;
            _mm_store_ps(dst + dex_dst, recv[2]);   dex_dst += Wdst;
            _mm_store_ps(dst + dex_dst, recv[3]);   dex_dst += Wdst;
        }

        recv[0] = _mm_set1_ps(0);       recv[1] = _mm_set1_ps(0);
        recv[2] = _mm_set1_ps(0);       recv[3] = _mm_set1_ps(0);

        for (int k = 0; k < _LH; ++k) {
            recv[k] = _mm_load_ps(src + dex_src + Wsrc * k);
        }

        dex_src += 4;

        _AVX_MM128_TRANSPOSE_4X4_(recv, reg);

        for (int k = 0; k < _LW; ++k) {
            _mm_store_ps(dst + dex_dst, recv[k]);
            dex_dst += Wdst;
        }
    }
}




_THREAD_FUNCTION_ void
decx::bp::CPUK::transpose_2x2_b64(const double* __restrict src, double* __restrict dst,
    const uint2 proc_dims_src, const uint Wsrc, const uint Wdst, const bool _LW)
{
    __m128d recv[2], store[2];

    size_t dex_src = 0, dex_dst = 0;

    for (int i = 0; i < proc_dims_src.y; ++i) {
        dex_src = i * 2 * Wsrc;
        dex_dst = i * 2;
        for (int j = 0; j < proc_dims_src.x; ++j) {
            recv[0] = _mm_load_pd(src + dex_src);
            recv[1] = _mm_load_pd(src + dex_src + Wsrc);

            dex_src += 2;

            _AVX_MM128_TRANSPOSE_2X2_(recv, store);

            _mm_store_pd(dst + dex_dst, store[0]);   dex_dst += Wdst;
            _mm_store_pd(dst + dex_dst, store[1]);   dex_dst += Wdst;
        }

        recv[1] = _mm_set1_pd(0);

        if (_LW) {
            recv[0] = _mm_load_pd(src + dex_src);
            recv[1] = _mm_load_pd(src + dex_src + Wsrc);

            dex_src += 2;

            _AVX_MM128_TRANSPOSE_2X2_(recv, store);

            _mm_store_pd(dst + dex_dst, store[0]);
            dex_dst += Wdst;
        }
    }
}



_THREAD_FUNCTION_ void
decx::bp::CPUK::transpose_2x2_b64_LH(const double* __restrict src, double* __restrict dst,
    const uint2 proc_dims_src, const uint Wsrc, const uint Wdst, const bool is_LW, const bool is_LH)
{
    __m128d recv[2], store[2];

    size_t dex_src = 0, dex_dst = 0;

    for (int i = 0; i < proc_dims_src.y; ++i) {
        dex_src = i * 2 * Wsrc;
        dex_dst = i * 2;
        for (int j = 0; j < proc_dims_src.x; ++j) {
            recv[0] = _mm_load_pd(src + dex_src);
            recv[1] = _mm_load_pd(src + dex_src + Wsrc);

            dex_src += 2;

            _AVX_MM128_TRANSPOSE_2X2_(recv, store);

            _mm_store_pd(dst + dex_dst, store[0]);   dex_dst += Wdst;
            _mm_store_pd(dst + dex_dst, store[1]);   dex_dst += Wdst;
        }
        recv[0] = _mm_load_pd(src + dex_src);
        recv[1] = _mm_load_pd(src + dex_src + Wsrc);

        dex_src += 2;

        _AVX_MM128_TRANSPOSE_2X2_(recv, store);

        _mm_store_pd(dst + dex_dst, store[0]);
        dex_dst += Wdst;
    }

    // _leftovers
    dex_src = proc_dims_src.y * 2 * Wsrc;
    dex_dst = proc_dims_src.y * 2;
    if (is_LH) {
        for (int j = 0; j < proc_dims_src.x; ++j) {

            recv[1] = _mm_set1_pd(0);
            recv[0] = _mm_load_pd(src + dex_src);

            dex_src += 2;

            _AVX_MM128_TRANSPOSE_2X2_(recv, store);

            _mm_store_pd(dst + dex_dst, store[0]);   dex_dst += Wdst;
            _mm_store_pd(dst + dex_dst, store[1]);   dex_dst += Wdst;
        }

        recv[1] = _mm_set1_pd(0);
        recv[0] = _mm_load_pd(src + dex_src);

        dex_src += 2;

        _AVX_MM128_TRANSPOSE_2X2_(recv, store);

        _mm_store_pd(dst + dex_dst, store[0]);
    }
}




void decx::bp::transpose_4x4_caller(const float* src, float* dst, const uint2 proc_dim_src, const uint Wsrc, const uint Wdst)
{
    decx::utils::_thread_arrange_1D t1D(decx::cpu::_get_permitted_concurrency());
    decx::utils::frag_manager f_mgr;
    decx::utils::frag_manager_gen(&f_mgr, proc_dim_src.y / 4, t1D.total_thread);

    const float* loc_src = src;
    float* loc_dst = dst;

    const size_t frag_src = Wsrc * f_mgr.frag_len * 4,
        frag_dst = f_mgr.frag_len * 4;

    for (int i = 0; i < t1D.total_thread - 1; ++i) {
        t1D._async_thread[i] = decx::cpu::register_task_default( decx::bp::CPUK::transpose_4x4_b32,
            loc_src, loc_dst, make_uint2(proc_dim_src.x / 4, f_mgr.frag_len), Wsrc, Wdst, proc_dim_src.x % 4);

        loc_src += frag_src;
        loc_dst += frag_dst;
    }
    const uint _L = f_mgr.is_left ? f_mgr.frag_left_over : f_mgr.frag_len;
    t1D._async_thread[t1D.total_thread - 1] = decx::cpu::register_task_default( decx::bp::CPUK::transpose_4x4_b32_LH,
        loc_src, loc_dst, make_uint2(proc_dim_src.x / 4, _L), Wsrc, Wdst, proc_dim_src.x % 4, proc_dim_src.y % 4);

    t1D.__sync_all_threads();
}




void decx::bp::transpose_2x2_caller(const double* src, double* dst, const uint2 proc_dim_src, const uint Wsrc, const uint Wdst)
{
    decx::utils::_thread_arrange_1D t1D(decx::cpu::_get_permitted_concurrency());
    decx::utils::frag_manager f_mgr;
    decx::utils::frag_manager_gen(&f_mgr, proc_dim_src.y / 2, t1D.total_thread);

    const double* loc_src = src;
    double* loc_dst = dst;

    const size_t frag_src = Wsrc * f_mgr.frag_len * 2,
        frag_dst = f_mgr.frag_len * 2;

    for (int i = 0; i < t1D.total_thread - 1; ++i) {
        t1D._async_thread[i] = decx::cpu::register_task_default( decx::bp::CPUK::transpose_2x2_b64,
            loc_src, loc_dst, make_uint2(proc_dim_src.x / 2, f_mgr.frag_len), Wsrc, Wdst, proc_dim_src.x % 2);

        loc_src += frag_src;
        loc_dst += frag_dst;
    }
    const uint _L = f_mgr.is_left ? f_mgr.frag_left_over : f_mgr.frag_len;
    t1D._async_thread[t1D.total_thread - 1] = decx::cpu::register_task_default( decx::bp::CPUK::transpose_2x2_b64_LH,
        loc_src, loc_dst, make_uint2(proc_dim_src.x / 2, _L), Wsrc, Wdst, proc_dim_src.x % 2, proc_dim_src.y % 2);

    t1D.__sync_all_threads();
}