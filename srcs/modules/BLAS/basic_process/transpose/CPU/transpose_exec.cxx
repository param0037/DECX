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


namespace decx
{
namespace bp {
    namespace CPUK {
        _THREAD_CALL_ static void transpose_Nx8_b8(const double* src, double* dst, const uint32_t proc_dims_Wsrc, const uint32_t Wsrc_v8,
            const uint32_t Wdst_v8, const uint8_t N);


        _THREAD_CALL_ static void transpose_Nx2_b64(const double* src, double* dst, const uint32_t proc_dims_Wsrc, const uint32_t Wsrc_v1,
            const uint32_t Wdst_v1);


        _THREAD_CALL_ static void transpose_Nx4_b32(const float* src, float* dst, const uint32_t proc_dims_Wsrc, const uint32_t Wsrc_v8,
            const uint32_t Wdst_v1, const uint8_t N);
    }
}
}

// -------------------------------------------------- 32-bit --------------------------------------------------


_THREAD_CALL_ static void 
decx::bp::CPUK::transpose_Nx4_b32(const float* __restrict   src, 
                                  float* __restrict         dst, 
                                  const uint32_t            proc_dims_Wsrc, 
                                  const uint32_t            Wsrc_v1,
                                  const uint32_t            Wdst_v1, 
                                  const uint8_t             N)
{
    __m128 recv[4], reg[4];

    uint64_t dex_src = 0, dex_dst = 0;

    const uint32_t _integral_x_v8 = (proc_dims_Wsrc >> 2);
    const uint8_t _LX = (proc_dims_Wsrc % 4);

    for (int j = 0; j < _integral_x_v8; ++j) {

        recv[0] = _mm_set1_ps(0);       recv[1] = _mm_set1_ps(0);
        recv[2] = _mm_set1_ps(0);       recv[3] = _mm_set1_ps(0);

        for (int k = 0; k < N; ++k) {
            recv[k] = _mm_load_ps(src + dex_src + Wsrc_v1 * k);
        }

        dex_src += 4;

        _AVX_MM128_TRANSPOSE_4X4_(recv, reg);

        _mm_store_ps(dst + dex_dst, recv[0]);   dex_dst += Wdst_v1;
        _mm_store_ps(dst + dex_dst, recv[1]);   dex_dst += Wdst_v1;
        _mm_store_ps(dst + dex_dst, recv[2]);   dex_dst += Wdst_v1;
        _mm_store_ps(dst + dex_dst, recv[3]);   dex_dst += Wdst_v1;
    }

    recv[0] = _mm_set1_ps(0);       recv[1] = _mm_set1_ps(0);
    recv[2] = _mm_set1_ps(0);       recv[3] = _mm_set1_ps(0);

    for (int k = 0; k < N; ++k) {
        recv[k] = _mm_load_ps(src + dex_src + Wsrc_v1 * k);
    }

    dex_src += 4;

    _AVX_MM128_TRANSPOSE_4X4_(recv, reg);

    for (int k = 0; k < _LX; ++k) {
        _mm_store_ps(dst + dex_dst, recv[k]);
        dex_dst += Wdst_v1;
    }
}


_THREAD_FUNCTION_ void
decx::bp::CPUK::transpose_4x4_b32(const float* __restrict   src, 
                                  float* __restrict         dst,
                                  const uint2               proc_dims_src, 
                                  const uint32_t            Wsrc_v1, 
                                  const uint32_t            Wdst_v1)
{
    __m128 recv[4], reg[4];

    uint64_t dex_src = 0, dex_dst = 0;

    const uint32_t _integral_x_v4 = (proc_dims_src.x >> 2);
    const uint8_t _LX = (proc_dims_src.x % 4);

    for (int i = 0; i < (proc_dims_src.y >> 2); ++i) {
        dex_src = (i << 2) * Wsrc_v1;
        dex_dst = (i << 2);
        for (int j = 0; j < _integral_x_v4; ++j) {
            recv[0] = _mm_load_ps(src + dex_src);
            recv[1] = _mm_load_ps(src + dex_src + Wsrc_v1);
            recv[2] = _mm_load_ps(src + dex_src + (Wsrc_v1 << 1));
            recv[3] = _mm_load_ps(src + dex_src + (Wsrc_v1 * 3));

            dex_src += 4;

            _AVX_MM128_TRANSPOSE_4X4_(recv, reg);

            _mm_store_ps(dst + dex_dst, recv[0]);   dex_dst += Wdst_v1;
            _mm_store_ps(dst + dex_dst, recv[1]);   dex_dst += Wdst_v1;
            _mm_store_ps(dst + dex_dst, recv[2]);   dex_dst += Wdst_v1;
            _mm_store_ps(dst + dex_dst, recv[3]);   dex_dst += Wdst_v1;
        }

        recv[0] = _mm_set1_ps(0);       recv[1] = _mm_set1_ps(0);
        recv[2] = _mm_set1_ps(0);       recv[3] = _mm_set1_ps(0);

        if (_LX > 0) {
            recv[0] = _mm_load_ps(src + dex_src);
            recv[1] = _mm_load_ps(src + dex_src + Wsrc_v1);
            recv[2] = _mm_load_ps(src + dex_src + (Wsrc_v1 << 1));
            recv[3] = _mm_load_ps(src + dex_src + (Wsrc_v1 * 3));

            dex_src += 4;

            _AVX_MM128_TRANSPOSE_4X4_(recv, reg);

            for (int k = 0; k < _LX; ++k) {
                _mm_store_ps(dst + dex_dst, recv[k]);
                dex_dst += Wdst_v1;
            }
        }
    }
}



_THREAD_FUNCTION_ void
decx::bp::CPUK::transpose_4x4_b32_LH(const float* __restrict    src, 
                                     float* __restrict          dst,
                                     const uint2                proc_dims_src, 
                                     const uint32_t             Wsrc_v1, 
                                     const uint32_t             Wdst_v1)
{
    __m128 recv[4], reg[4];

    const uint32_t _integral_x_v4 = (proc_dims_src.x >> 2);
    const uint8_t _LX = (proc_dims_src.x % 4);
    const uint32_t _integral_y_v4 = (proc_dims_src.y >> 2);
    const uint8_t _LY = (proc_dims_src.y % 4);

    uint64_t dex_src = 0, dex_dst = 0;

    decx::bp::CPUK::transpose_4x4_b32(src, dst, make_uint2(proc_dims_src.x, _integral_y_v4 << 2), Wsrc_v1, Wdst_v1);

    if (_LY) {
        decx::bp::CPUK::transpose_Nx4_b32(src + (_integral_y_v4 << 2) * Wsrc_v1, dst + (_integral_y_v4 << 2), proc_dims_src.x, Wsrc_v1, Wdst_v1, _LY);
    }
}


// --------------------------------------------------- 8-bit ---------------------------------------------------

_THREAD_GENERAL_ void
decx::bp::CPUK::transpose_8x8_b8(const double* __restrict   src, 
                                  double* __restrict        dst,
                                  const uint2               proc_dims_src, 
                                  const uint32_t            Wsrc_v8, 
                                  const uint32_t            Wdst_v8)
{
    decx::utils::simd::xmm64_reg _regs0[8], _regs1[8];

    uint64_t dex_src = 0, dex_dst = 0;

    const uint32_t _integral_x_v8 = (proc_dims_src.x >> 3);
    const uint8_t _LX = (proc_dims_src.x % 8);

    for (int i = 0; i < (proc_dims_src.y >> 3); ++i) {
        dex_src = i * 8 * Wsrc_v8;
        dex_dst = i;
        for (int j = 0; j < _integral_x_v8; ++j) {
            _regs0[0]._fp64 = src[dex_src];                         _regs0[1]._fp64 = src[dex_src + Wsrc_v8];
            _regs0[2]._fp64 = src[dex_src + (Wsrc_v8 << 1)];           _regs0[3]._fp64 = src[dex_src + (Wsrc_v8 * 3)];
            _regs0[4]._fp64 = src[dex_src + (Wsrc_v8 << 2)];           _regs0[5]._fp64 = src[dex_src + Wsrc_v8 * 5];
            _regs0[6]._fp64 = src[dex_src + Wsrc_v8 * 6];              _regs0[7]._fp64 = src[dex_src + (Wsrc_v8 * 7)];

            ++dex_src;

#ifdef __GNUC__
            decx::bp::CPUK::block8x8_transpose_u8((__m64*)_regs0, (__m64*)_regs1);
#endif
#ifdef _MSC_VER
            decx::bp::CPUK::block8x8_transpose_u8((uint64_t*)_regs0, (uint64_t*)_regs1);
#endif

            dst[dex_dst] = _regs1[0]._fp64;   dex_dst += Wdst_v8;
            dst[dex_dst] = _regs1[1]._fp64;   dex_dst += Wdst_v8;
            dst[dex_dst] = _regs1[2]._fp64;   dex_dst += Wdst_v8;
            dst[dex_dst] = _regs1[3]._fp64;   dex_dst += Wdst_v8;
            dst[dex_dst] = _regs1[4]._fp64;   dex_dst += Wdst_v8;
            dst[dex_dst] = _regs1[5]._fp64;   dex_dst += Wdst_v8;
            dst[dex_dst] = _regs1[6]._fp64;   dex_dst += Wdst_v8;
            dst[dex_dst] = _regs1[7]._fp64;   dex_dst += Wdst_v8;
        }

        _regs0[0]._ull = 0;       _regs0[1]._ull = 0;
        _regs0[2]._ull = 0;       _regs0[3]._ull = 0;
        _regs0[4]._ull = 0;       _regs0[5]._ull = 0;
        _regs0[6]._ull = 0;       _regs0[7]._ull = 0;

        if (_LX > 0) {
            _regs0[0]._fp64 = src[dex_src];                         _regs0[1]._fp64 = src[dex_src + Wsrc_v8];
            _regs0[2]._fp64 = src[dex_src + (Wsrc_v8 << 1)];           _regs0[3]._fp64 = src[dex_src + (Wsrc_v8 * 3)];
            _regs0[4]._fp64 = src[dex_src + (Wsrc_v8 << 2)];           _regs0[5]._fp64 = src[dex_src + Wsrc_v8 * 5];
            _regs0[6]._fp64 = src[dex_src + Wsrc_v8 * 6];              _regs0[7]._fp64 = src[dex_src + (Wsrc_v8 * 7)];

            ++dex_src;

#ifdef __GNUC__
            decx::bp::CPUK::block8x8_transpose_u8((__m64*)_regs0, (__m64*)_regs1);
#endif
#ifdef _MSC_VER
            decx::bp::CPUK::block8x8_transpose_u8((uint64_t*)_regs0, (uint64_t*)_regs1);
#endif

            for (int k = 0; k < _LX; ++k) {
                dst[dex_dst] = _regs1[k]._fp64;
                dex_dst += Wdst_v8;
            }
        }
    }
}



_THREAD_CALL_ static void 
decx::bp::CPUK::transpose_Nx8_b8(const double* __restrict   src, 
                                 double* __restrict         dst, 
                                 const uint32_t             proc_dims_Wsrc, 
                                 const uint32_t             Wsrc_v8,
                                 const uint32_t             Wdst_v8, 
                                 const uint8_t              N)
{
    decx::utils::simd::xmm64_reg _regs0[8], _regs1[8];

    uint64_t dex_src = 0, dex_dst = 0;

    const uint32_t _integral_x_v8 = (proc_dims_Wsrc >> 3);
    const uint8_t _LX = (proc_dims_Wsrc % 8);

    for (int j = 0; j < _integral_x_v8; ++j) {
        _regs0[0]._ull = 0;       _regs0[1]._ull = 0;
        _regs0[2]._ull = 0;       _regs0[3]._ull = 0;
        _regs0[4]._ull = 0;       _regs0[5]._ull = 0;
        _regs0[6]._ull = 0;       _regs0[7]._ull = 0;

        for (int k = 0; k < N; ++k) {
            _regs0[k]._fp64 = src[dex_src + Wsrc_v8 * k];
        }

        ++dex_src;

#ifdef __GNUC__
        decx::bp::CPUK::block8x8_transpose_u8((__m64*)_regs0, (__m64*)_regs1);
#endif
#ifdef _MSC_VER
        decx::bp::CPUK::block8x8_transpose_u8((uint64_t*)_regs0, (uint64_t*)_regs1);
#endif

        dst[dex_dst] = _regs1[0]._fp64;   dex_dst += Wdst_v8;
        dst[dex_dst] = _regs1[1]._fp64;   dex_dst += Wdst_v8;
        dst[dex_dst] = _regs1[2]._fp64;   dex_dst += Wdst_v8;
        dst[dex_dst] = _regs1[3]._fp64;   dex_dst += Wdst_v8;
        dst[dex_dst] = _regs1[4]._fp64;   dex_dst += Wdst_v8;
        dst[dex_dst] = _regs1[5]._fp64;   dex_dst += Wdst_v8;
        dst[dex_dst] = _regs1[6]._fp64;   dex_dst += Wdst_v8;
        dst[dex_dst] = _regs1[7]._fp64;   dex_dst += Wdst_v8;
    }

    _regs0[0]._ull = 0;       _regs0[1]._ull = 0;
    _regs0[2]._ull = 0;       _regs0[3]._ull = 0;
    _regs0[4]._ull = 0;       _regs0[5]._ull = 0;
    _regs0[6]._ull = 0;       _regs0[7]._ull = 0;

    for (int k = 0; k < N; ++k) {
        _regs0[k]._fp64 = src[dex_src + Wsrc_v8 * k];
    }

    ++dex_src;

#ifdef __GNUC__
    decx::bp::CPUK::block8x8_transpose_u8((__m64*)_regs0, (__m64*)_regs1);
#endif
#ifdef _MSC_VER
    decx::bp::CPUK::block8x8_transpose_u8((uint64_t*)_regs0, (uint64_t*)_regs1);
#endif

    for (int k = 0; k < _LX; ++k) {
        dst[dex_dst] = _regs1[k]._fp64;
        dex_dst += Wdst_v8;
    }
}



_THREAD_FUNCTION_ void
decx::bp::CPUK::transpose_8x8_b8_LH(const double* __restrict        src, 
                                    double* __restrict              dst,
                                    const uint2                     proc_dims_src, 
                                    const uint32_t                  Wsrc_v8, 
                                    const uint32_t                  Wdst_v8)
{
    decx::utils::simd::xmm64_reg _regs0[8], _regs1[8];

    uint64_t dex_src = 0, dex_dst = 0;

    const uint32_t _integral_x_v8 = (proc_dims_src.x >> 3);

    const uint32_t _integral_y_v8 = (proc_dims_src.y >> 3);
    const uint8_t _LY = (proc_dims_src.y % 8);

    decx::bp::CPUK::transpose_8x8_b8(src, dst, make_uint2(proc_dims_src.x, _integral_y_v8 << 3), Wsrc_v8, Wdst_v8);

    // _leftovers
    dex_src = _integral_y_v8 * 8 * Wsrc_v8;
    dex_dst = _integral_y_v8;
    if (_LY) {
        decx::bp::CPUK::transpose_Nx8_b8(src + (_integral_y_v8 << 3) * Wsrc_v8, dst + _integral_y_v8, proc_dims_src.x, Wsrc_v8, Wdst_v8, _LY);
    }
}

// --------------------------------------------------- 64-bit ---------------------------------------------------


_THREAD_CALL_ static void 
decx::bp::CPUK::transpose_Nx2_b64(const double* __restrict src, 
                                  double* __restrict dst, 
                                  const uint32_t proc_dims_Wsrc, 
                                  const uint32_t Wsrc_v1,
                                  const uint32_t Wdst_v1)
{
    __m128d recv[2], store[2];

    uint64_t dex_src = 0, dex_dst = 0;

    const uint32_t _integral_x_v2 = (proc_dims_Wsrc >> 1);
    const uint8_t _LX = (proc_dims_Wsrc % 2);

    for (int j = 0; j < _integral_x_v2; ++j) {

        recv[1] = _mm_set1_pd(0);
        recv[0] = _mm_load_pd(src + dex_src);

        dex_src += 2;

        _AVX_MM128_TRANSPOSE_2X2_(recv, store);

        _mm_store_pd(dst + dex_dst, store[0]);   dex_dst += Wdst_v1;
        _mm_store_pd(dst + dex_dst, store[1]);   dex_dst += Wdst_v1;
    }

    recv[1] = _mm_set1_pd(0);
    recv[0] = _mm_load_pd(src + dex_src);

    dex_src += 2;

    _AVX_MM128_TRANSPOSE_2X2_(recv, store);

    _mm_store_pd(dst + dex_dst, store[0]);
}


_THREAD_FUNCTION_ void
decx::bp::CPUK::transpose_2x2_b64(const double* __restrict src, 
                                  double* __restrict dst,
                                  const uint2 proc_dims_src, 
                                  const uint32_t Wsrc_v1, 
                                  const uint32_t Wdst_v1)
{
    __m128d recv[2], store[2];

    uint64_t dex_src = 0, dex_dst = 0;

    const uint32_t _integral_x_v2 = (proc_dims_src.x >> 1);
    const uint8_t _LX = (proc_dims_src.x % 2);

    for (int i = 0; i < (proc_dims_src.y >> 1); ++i) {
        dex_src = i * 2 * Wsrc_v1;
        dex_dst = i * 2;
        for (int j = 0; j < _integral_x_v2; ++j) {
            recv[0] = _mm_load_pd(src + dex_src);
            recv[1] = _mm_load_pd(src + dex_src + Wsrc_v1);

            dex_src += 2;

            _AVX_MM128_TRANSPOSE_2X2_(recv, store);

            _mm_store_pd(dst + dex_dst, store[0]);   dex_dst += Wdst_v1;
            _mm_store_pd(dst + dex_dst, store[1]);   dex_dst += Wdst_v1;
        }

        recv[1] = _mm_set1_pd(0);

        if (_LX) {
            recv[0] = _mm_load_pd(src + dex_src);
            recv[1] = _mm_load_pd(src + dex_src + Wsrc_v1);

            dex_src += 2;

            _AVX_MM128_TRANSPOSE_2X2_(recv, store);

            _mm_store_pd(dst + dex_dst, store[0]);
            dex_dst += Wdst_v1;
        }
    }
}



_THREAD_FUNCTION_ void
decx::bp::CPUK::transpose_2x2_b64_LH(const double* __restrict src, 
                                     double* __restrict dst,
                                     const uint2 proc_dims_src, 
                                     const uint32_t Wsrc_v1, 
                                     const uint32_t Wdst_v1)
{
    const uint32_t _integral_x_v2 = (proc_dims_src.x >> 1);
    const uint8_t _LX = (proc_dims_src.x % 2);
    const uint32_t _integral_y_v2 = (proc_dims_src.y >> 1);
    const uint8_t _LY = (proc_dims_src.y % 2);

    decx::bp::CPUK::transpose_2x2_b64(src, dst, make_uint2(proc_dims_src.x, _integral_y_v2 << 1), Wsrc_v1, Wdst_v1);

    if (_LY) {
        decx::bp::CPUK::transpose_Nx2_b64(src + (_integral_y_v2 << 1) * Wsrc_v1, dst + (_integral_y_v2 << 1), proc_dims_src.x, Wsrc_v1, Wdst_v1);
    }
}



// --------------------------------------------------- callers ---------------------------------------------------

void decx::bp::transpose_8x8_caller(const double* src,                                       double* dst, 
                                    const uint32_t Wsrc,                                    const uint32_t Wdst,
                                    const decx::bp::_cpu_transpose_config<1>* _config,      decx::utils::_thread_arrange_1D* t1D)
{
    const decx::utils::frag_manager* f_mgr = &_config->_f_mgr;
    const uint2& proc_dim_src = _config->_src_proc_dims;

    const uint8_t* loc_src = (uint8_t*)src;
    uint8_t* loc_dst = (uint8_t*)dst;

    const uint64_t frag_src = Wsrc * f_mgr->frag_len,
        frag_dst = f_mgr->frag_len;

    for (int i = 0; i < t1D->total_thread - 1; ++i) {
        t1D->_async_thread[i] = decx::cpu::register_task_default( decx::bp::CPUK::transpose_8x8_b8,
            (double*)loc_src, (double*)loc_dst, make_uint2(proc_dim_src.x, f_mgr->frag_len), Wsrc >> 3, Wdst >> 3);

        loc_src += frag_src;
        loc_dst += frag_dst;
    }
    const uint _L = f_mgr->is_left ? f_mgr->frag_left_over : f_mgr->frag_len;
    t1D->_async_thread[t1D->total_thread - 1] = decx::cpu::register_task_default( decx::bp::CPUK::transpose_8x8_b8_LH,
        (double*)loc_src, (double*)loc_dst, make_uint2(proc_dim_src.x, _L), Wsrc >> 3, Wdst >> 3);

    t1D->__sync_all_threads();
}



void decx::bp::transpose_4x4_caller(const float* src,                                       float* dst, 
                                    const uint32_t Wsrc,                                    const uint32_t Wdst,
                                    const decx::bp::_cpu_transpose_config<4>* _config,      decx::utils::_thread_arrange_1D* t1D)
{
    const decx::utils::frag_manager* f_mgr = &_config->_f_mgr;
    const uint2& proc_dim_src = _config->_src_proc_dims;

    const float* loc_src = src;
    float* loc_dst = dst;

    const uint64_t frag_src = Wsrc * f_mgr->frag_len,
        frag_dst = f_mgr->frag_len;

    for (int i = 0; i < t1D->total_thread - 1; ++i) {
        t1D->_async_thread[i] = decx::cpu::register_task_default( decx::bp::CPUK::transpose_4x4_b32,
            loc_src, loc_dst, make_uint2(proc_dim_src.x, f_mgr->frag_len), Wsrc, Wdst);

        loc_src += frag_src;
        loc_dst += frag_dst;
    }
    const uint _L = f_mgr->is_left ? f_mgr->frag_left_over : f_mgr->frag_len;
    t1D->_async_thread[t1D->total_thread - 1] = decx::cpu::register_task_default( decx::bp::CPUK::transpose_4x4_b32_LH,
        loc_src, loc_dst, make_uint2(proc_dim_src.x, _L), Wsrc, Wdst);

    t1D->__sync_all_threads();
}




void decx::bp::transpose_2x2_caller(const double* src,                                      double* dst, 
                                    const uint32_t Wsrc,                                    const uint32_t Wdst,
                                    const decx::bp::_cpu_transpose_config<8>* _config,      decx::utils::_thread_arrange_1D* t1D)
{
    const double* loc_src = src;
    double* loc_dst = dst;

    const decx::utils::frag_manager* f_mgr = &_config->_f_mgr;
    const uint2& proc_dim_src = _config->_src_proc_dims;

    const size_t frag_src = Wsrc * f_mgr->frag_len,
        frag_dst = f_mgr->frag_len;

    for (int i = 0; i < t1D->total_thread - 1; ++i) {
        t1D->_async_thread[i] = decx::cpu::register_task_default( decx::bp::CPUK::transpose_2x2_b64,
            loc_src, loc_dst, make_uint2(proc_dim_src.x, f_mgr->frag_len), Wsrc, Wdst);

        loc_src += frag_src;
        loc_dst += frag_dst;
    }
    const uint _L = f_mgr->is_left ? f_mgr->frag_left_over : f_mgr->frag_len;
    t1D->_async_thread[t1D->total_thread - 1] = decx::cpu::register_task_default( decx::bp::CPUK::transpose_2x2_b64_LH,
        loc_src, loc_dst, make_uint2(proc_dim_src.x, _L), Wsrc, Wdst);

    t1D->__sync_all_threads();
}