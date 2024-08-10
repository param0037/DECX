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


#include "fill_constant_exec.h"


template <bool _is_left>
_THREAD_FUNCTION_ void
decx::bp::CPUK::fill_v256_b32_1D_end(float* __restrict src, const __m256 val, const size_t fill_len, const __m256 _end_val)
{
    for (int i = 0; i < fill_len - 1; ++i) {
        _mm256_store_ps(src + ((size_t)i << 3), val);
    }
    if (_is_left) {
        _mm256_store_ps(src + ((size_t)(fill_len - 1) << 3), _end_val);
    }
    else {
        _mm256_store_ps(src + ((size_t)(fill_len - 1) << 3), val);
    }
}


template <bool _is_left>
_THREAD_FUNCTION_ void
decx::bp::CPUK::fill_v256_b64_1D_end(double* __restrict src, const __m256d val, const size_t fill_len, const __m256d _end_val)
{
    for (int i = 0; i < fill_len - 1; ++i) {
        _mm256_store_pd(src + ((size_t)i << 2), val);
    }
    if (_is_left) {
        _mm256_store_pd(src + ((size_t)(fill_len - 1) << 2), _end_val);
    }
    else{
        _mm256_store_pd(src + ((size_t)(fill_len - 1) << 2), val);
    }
}



void decx::bp::fill1D_v256_b32_caller_MT(float* src, const float val, const size_t fill_len,
    decx::utils::_thread_arrange_1D* t1D)
{
    decx::utils::frag_manager f_mgr;
    decx::utils::frag_manager_gen_Nx(&f_mgr, fill_len, t1D->total_thread, 8);

    float* loc_src = src;

    for (int i = 0; i < t1D->total_thread - 1; ++i) {
        t1D->_async_thread[i] = decx::cpu::register_task_default( decx::bp::CPUK::fill_v256_b32_1D_end<false>,
            loc_src, _mm256_set1_ps(val), f_mgr.frag_len / 8, _mm256_set1_ps(0));

        loc_src += f_mgr.frag_len;
    }
    const uint _L = f_mgr.is_left ? f_mgr.frag_left_over : f_mgr.frag_len;

    if (fill_len % 8) {
        __m256 _end_val = _mm256_set1_ps(0);
        for (int i = 0; i < (fill_len % 8); ++i) {
#ifdef _MSC_VER
            _end_val.m256_f32[i] = val;
#endif
#ifdef __GNUC__
            ((float*)&_end_val)[i] = val;
#endif
        }
        t1D->_async_thread[t1D->total_thread - 1] = decx::cpu::register_task_default( 
            decx::bp::CPUK::fill_v256_b32_1D_end<true>, loc_src, _mm256_set1_ps(val), decx::utils::ceil<size_t>(_L, 8), _end_val);
    }
    else {
        t1D->_async_thread[t1D->total_thread - 1] = decx::cpu::register_task_default(
            decx::bp::CPUK::fill_v256_b32_1D_end<false>, loc_src, _mm256_set1_ps(val), _L / 8, _mm256_set1_ps(0));
    }

    t1D->__sync_all_threads();
}





void decx::bp::fill1D_v256_b64_caller_MT(double* src, const double val, const size_t fill_len,
    decx::utils::_thread_arrange_1D* t1D)
{
    decx::utils::frag_manager f_mgr;
    decx::utils::frag_manager_gen_Nx(&f_mgr, fill_len, t1D->total_thread, 4);

    double* loc_src = src;

    for (int i = 0; i < t1D->total_thread - 1; ++i) {
        t1D->_async_thread[i] = decx::cpu::register_task_default( decx::bp::CPUK::fill_v256_b64_1D_end<false>,
            loc_src, _mm256_set1_pd(val), f_mgr.frag_len / 4, _mm256_set1_pd(0));

        loc_src += f_mgr.frag_len;
    }
    const uint _L = f_mgr.is_left ? f_mgr.frag_left_over : f_mgr.frag_len;

    if (fill_len % 4) {
        __m256d _end_val = _mm256_set1_pd(0);
        for (int i = 0; i < (fill_len % 4); ++i) {
#ifdef _MSC_VER
            _end_val.m256d_f64[i] = val;
#endif
#ifdef __GNUC__
            ((double*)&_end_val)[i] = val;
#endif
        }
        t1D->_async_thread[t1D->total_thread - 1] = decx::cpu::register_task_default(
            decx::bp::CPUK::fill_v256_b64_1D_end<true>, loc_src, _mm256_set1_pd(val), decx::utils::ceil<size_t>(_L, 4), _end_val);
    }
    else {
        t1D->_async_thread[t1D->total_thread - 1] = decx::cpu::register_task_default(
            decx::bp::CPUK::fill_v256_b64_1D_end<true>, loc_src, _mm256_set1_pd(val), _L / 4, _mm256_set1_pd(0));
    }

    t1D->__sync_all_threads();
}




void decx::bp::fill1D_v256_b32_caller_ST(float* src, const float val, const size_t fill_len)
{
    if (fill_len % 8) {
        __m256 _end_val = _mm256_set1_ps(0);
        for (int i = 0; i < (fill_len % 8); ++i) {
#ifdef _MSC_VER
            _end_val.m256_f32[i] = val;
#endif
#ifdef __GNUC__
            ((float*)&_end_val)[i] = val;
#endif
        }

        decx::bp::CPUK::fill_v256_b32_1D_end<true>(src, _mm256_set1_ps(val), decx::utils::ceil<size_t>(fill_len, 8), _end_val);
    }
    else {
        decx::bp::CPUK::fill_v256_b32_1D_end<false>(src, _mm256_set1_ps(val), fill_len / 8);
    }
}




void decx::bp::fill1D_v256_b64_caller_ST(double* src, const double val, const size_t fill_len)
{
    if (fill_len % 4) {
        __m256d _end_val = _mm256_set1_pd(0);
        for (int i = 0; i < (fill_len % 4); ++i) {
#ifdef _MSC_VER
            _end_val.m256d_f64[i] = val;
#endif
#ifdef __GNUC__
            ((double*)&_end_val)[i] = val;
#endif
        }

        decx::bp::CPUK::fill_v256_b64_1D_end<true>(src, _mm256_set1_pd(val), decx::utils::ceil<size_t>(fill_len, 4), _end_val);
    }
    else {
        decx::bp::CPUK::fill_v256_b64_1D_end<false>(src, _mm256_set1_pd(val), fill_len / 4, _mm256_set1_pd(0));
    }
}



// ------------------------------------------------------- 2D ------------------------------------------------------------------


template <bool _is_left>
_THREAD_FUNCTION_ void
decx::bp::CPUK::fill_v256_b32_2D_LF(float* __restrict src, const __m256 val, const uint2 proc_dims, const uint Wsrc,
    const __m256 _end_val)
{
    size_t dex = 0;

    for (int i = 0; i < proc_dims.y; ++i) {
        dex = i * Wsrc;
        for (int j = 0; j < proc_dims.x - 1; ++j) {
            _mm256_store_ps(src + dex, val);
            dex += 8;
        }
        if (_is_left) {
            _mm256_store_ps(src + dex, _end_val);
        }
        else {
            _mm256_store_ps(src + dex, val);
        }
    }
}



template <bool _is_left>
_THREAD_FUNCTION_ void
decx::bp::CPUK::fill_v256_b64_2D_LF(double* __restrict src, const __m256d val, const uint2 proc_dims, const uint Wsrc, 
    const __m256d _end_val)
{
    size_t dex = 0;

    for (int i = 0; i < proc_dims.y; ++i) {
        dex = i * Wsrc;
        for (int j = 0; j < proc_dims.x - 1; ++j) {
            _mm256_store_pd(src + dex, val);

            dex += 4;
        }
        if (_is_left) {
            _mm256_store_pd(src + dex, _end_val);
        }
        else {
            _mm256_store_pd(src + dex, val);
        }
    }
}



void decx::bp::fill2D_v256_b32_caller_MT(float* src, const float val, const uint2 proc_dims, const uint Wsrc,
    decx::utils::_thread_arrange_1D* t1D)
{
    decx::utils::frag_manager f_mgr;
    decx::utils::frag_manager_gen(&f_mgr, proc_dims.y, t1D->total_thread);

    float* loc_src = src;
    const size_t frag_size = f_mgr.frag_len * Wsrc;

    if (proc_dims.x % 8) {
        __m256 _end_val = _mm256_set1_ps(0);
        for (int i = 0; i < (proc_dims.x % 8); ++i) {
#ifdef _MSC_VER
            _end_val.m256_f32[i] = val;
#endif
#ifdef __GNUC__
            ((float*)&_end_val)[i] = val;
#endif
        }

        for (int i = 0; i < t1D->total_thread - 1; ++i) {
            t1D->_async_thread[i] = decx::cpu::register_task_default(decx::bp::CPUK::fill_v256_b32_2D_LF<true>,
                loc_src, 
                _mm256_set1_ps(val), 
                make_uint2(decx::utils::ceil<uint>(proc_dims.x, 8), f_mgr.frag_len), 
                Wsrc, _end_val);

            loc_src += frag_size;
        }
        const uint _L = f_mgr.is_left ? f_mgr.frag_left_over : f_mgr.frag_len;

        t1D->_async_thread[t1D->total_thread - 1] = decx::cpu::register_task_default( decx::bp::CPUK::fill_v256_b32_2D_LF<true>,
            loc_src, 
            _mm256_set1_ps(val), 
            make_uint2(decx::utils::ceil<uint>(proc_dims.x, 8), _L),
            Wsrc, _end_val);
    }
    else {
        for (int i = 0; i < t1D->total_thread - 1; ++i) {
            t1D->_async_thread[i] = decx::cpu::register_task_default( decx::bp::CPUK::fill_v256_b32_2D_LF<false>,
                loc_src, _mm256_set1_ps(val), make_uint2(proc_dims.x / 8, f_mgr.frag_len), Wsrc, _mm256_set1_ps(0));

            loc_src += frag_size;
        }
        const uint _L = f_mgr.is_left ? f_mgr.frag_left_over : f_mgr.frag_len;

        t1D->_async_thread[t1D->total_thread - 1] = decx::cpu::register_task_default( decx::bp::CPUK::fill_v256_b32_2D_LF<false>,
            loc_src, _mm256_set1_ps(val), make_uint2(proc_dims.x / 8, _L), Wsrc, _mm256_set1_ps(0));
    }

    t1D->__sync_all_threads();
}




void decx::bp::fill2D_v256_b64_caller_MT(double* src, const double val, const uint2 proc_dims, const uint Wsrc,
    decx::utils::_thread_arrange_1D* t1D)
{
    decx::utils::frag_manager f_mgr;
    decx::utils::frag_manager_gen(&f_mgr, proc_dims.y, t1D->total_thread);

    double* loc_src = src;
    const size_t frag_size = f_mgr.frag_len * Wsrc;

    if (proc_dims.x % 4) {
        __m256d _end_val = _mm256_set1_pd(0);
        for (int i = 0; i < (proc_dims.x % 4); ++i) {
#ifdef _MSC_VER
            _end_val.m256d_f64[i] = val;
#endif
#ifdef __GNUC__
            ((double*)&_end_val)[i] = val;
#endif
        }

        for (int i = 0; i < t1D->total_thread - 1; ++i) {
            t1D->_async_thread[i] = decx::cpu::register_task_default(decx::bp::CPUK::fill_v256_b64_2D_LF<true>,
                loc_src,
                _mm256_set1_pd(val),
                make_uint2(decx::utils::ceil<uint>(proc_dims.x, 4), f_mgr.frag_len),
                Wsrc, _end_val);

            loc_src += frag_size;
        }
        const uint _L = f_mgr.is_left ? f_mgr.frag_left_over : f_mgr.frag_len;

        t1D->_async_thread[t1D->total_thread - 1] = decx::cpu::register_task_default(decx::bp::CPUK::fill_v256_b64_2D_LF<true>,
            loc_src,
            _mm256_set1_pd(val),
            make_uint2(decx::utils::ceil<uint>(proc_dims.x, 4), _L),
            Wsrc, _end_val);
    }
    else {
        for (int i = 0; i < t1D->total_thread - 1; ++i) {
            t1D->_async_thread[i] = decx::cpu::register_task_default(decx::bp::CPUK::fill_v256_b64_2D_LF<false>,
                loc_src, _mm256_set1_pd(val), make_uint2(proc_dims.x / 4, f_mgr.frag_len), Wsrc, _mm256_set1_pd(0));

            loc_src += frag_size;
        }
        const uint _L = f_mgr.is_left ? f_mgr.frag_left_over : f_mgr.frag_len;

        t1D->_async_thread[t1D->total_thread - 1] = decx::cpu::register_task_default( decx::bp::CPUK::fill_v256_b64_2D_LF<false>,
            loc_src, _mm256_set1_pd(val), make_uint2(proc_dims.x / 4, _L), Wsrc, _mm256_set1_pd(0));
    }

    t1D->__sync_all_threads();
}



void decx::bp::fill2D_v256_b32_caller_ST(float* src, const float val, const uint2 proc_dims, const uint Wsrc)
{
    if (proc_dims.x % 8) {
        __m256 _end_val = _mm256_set1_ps(0);
        for (int i = 0; i < (proc_dims.x % 8); ++i) {
#ifdef _MSC_VER
            _end_val.m256_f32[i] = val;
#endif
#ifdef __GNUC__
            ((float*)&_end_val)[i] = val;
#endif
        }
        decx::bp::CPUK::fill_v256_b32_2D_LF<true>(src, _mm256_set1_ps(val), make_uint2(decx::utils::ceil<uint>(proc_dims.x, 8), 
            proc_dims.y), Wsrc, _end_val);
    }
    else {
        decx::bp::CPUK::fill_v256_b32_2D_LF<false>(src, _mm256_set1_ps(val), make_uint2(proc_dims.x / 8, proc_dims.y), Wsrc);
    }
}




void decx::bp::fill2D_v256_b64_caller_ST(double* src, const double val, const uint2 proc_dims, const uint Wsrc)
{
    if (proc_dims.x % 4) {
        __m256d _end_val = _mm256_set1_pd(0);
        for (int i = 0; i < (proc_dims.x % 4); ++i) {
#ifdef _MSC_VER
            _end_val.m256d_f64[i] = val;
#endif
#ifdef __GNUC__
            ((double*)&_end_val)[i] = val;
#endif
        }
        decx::bp::CPUK::fill_v256_b64_2D_LF<true>(src, _mm256_set1_pd(val), make_uint2(decx::utils::ceil<uint>(proc_dims.x, 4), proc_dims.y), 
            Wsrc, _end_val);
    }
    else {
        decx::bp::CPUK::fill_v256_b64_2D_LF<false>(src, _mm256_set1_pd(val), make_uint2(proc_dims.x / 4, proc_dims.y), Wsrc);
    }
}