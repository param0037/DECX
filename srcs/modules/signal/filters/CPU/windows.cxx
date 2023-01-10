/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/

#include "windows.h"


_THREAD_FUNCTION_ void
decx::signal::CPUK::Gaussian_Window1D_cpl32(const double* __restrict    src, 
                                            double* __restrict          dst, 
                                            const float                 u, 
                                            const float                 sigma, 
                                            const size_t                _proc_len,
                                            const size_t                real_bound,
                                            const size_t                global_dex_offset)
{
    size_t loc_dex = 0;

    const __m256i half_len = _mm256_set1_epi64x(real_bound / 2);
    __m256i current_dex, _half_pass;
    __m256 current_dex_f;

    const __m256 _means = _mm256_set1_ps(u), _2_x_sigmas = _mm256_set1_ps(-2 * sigma * sigma);

    decx::utils::simd::xmm256_reg g_weight, real_bounds,
        recv, store;
    real_bounds._vf = _mm256_set1_ps(real_bound - 1);

    for (int i = 0; i < _proc_len; ++i) {
        recv._vd = _mm256_load_pd(src + loc_dex);

        current_dex = _mm256_setr_epi64x(global_dex_offset + loc_dex,
                                         global_dex_offset + loc_dex + 1,
                                         global_dex_offset + loc_dex + 2,
                                         global_dex_offset + loc_dex + 3);

        current_dex_f = _mm256_setr_ps(global_dex_offset + loc_dex, global_dex_offset + loc_dex,
            global_dex_offset + loc_dex + 1, global_dex_offset + loc_dex + 1,
            global_dex_offset + loc_dex + 2, global_dex_offset + loc_dex + 2,
            global_dex_offset + loc_dex + 3, global_dex_offset + loc_dex + 3);

        _half_pass = _mm256_sub_epi64(half_len, current_dex);
        _half_pass = _mm256_srai_epi32(_half_pass, 31);
        _half_pass = _mm256_shuffle_epi32(_half_pass, 0b11110101);

        g_weight._vf = _mm256_sub_ps(current_dex_f, _mm256_castsi256_ps(_mm256_and_si256(_half_pass, real_bounds._vi)));
        
        g_weight._vf = _mm256_sub_ps(g_weight._vf, _means);
        g_weight._vf = _mm256_div_ps(_mm256_mul_ps(g_weight._vf, g_weight._vf), _2_x_sigmas);
        g_weight._vf = _mm256_exp_ps(g_weight._vf);
        
        store._vf = _mm256_mul_ps(recv._vf, g_weight._vf);

        _mm256_store_pd(dst + loc_dex, store._vd);
        loc_dex += 4;
    }
}



_THREAD_FUNCTION_ void
decx::signal::CPUK::Triangular_Window1D_cpl32(const double* __restrict      src, 
                                              double* __restrict            dst, 
                                              const long long               center, 
                                              const size_t                  radius,
                                              const size_t                  _proc_len,
                                              const size_t                  real_bound, 
                                              const size_t                  global_dex_offset)
{
    size_t loc_dex = 0;

    decx::utils::simd::xmm256_reg recv, store, _not_exceeded;
    const __m256i half_len = _mm256_set1_epi64x(real_bound / 2);
    __m256i current_dex, _half_pass;
    __m256 _weight;
    const __m256 real_bounds = _mm256_set1_ps(real_bound - 1),
                 _centers = _mm256_set1_ps(center),
                 _radius = _mm256_set1_ps(radius);

    for (int i = 0; i < _proc_len; ++i) {
        recv._vd = _mm256_load_pd(src + loc_dex);

        current_dex = _mm256_setr_epi64x(global_dex_offset + loc_dex,
                                         global_dex_offset + loc_dex + 1,
                                         global_dex_offset + loc_dex + 2,
                                         global_dex_offset + loc_dex + 3);

        _half_pass = _mm256_sub_epi64(half_len, current_dex);
        _half_pass = _mm256_srai_epi32(_half_pass, 31);
        _half_pass = _mm256_shuffle_epi32(_half_pass, 0b11110101);
        
        _weight = _mm256_setr_ps(global_dex_offset + loc_dex, global_dex_offset + loc_dex,
            global_dex_offset + loc_dex + 1, global_dex_offset + loc_dex + 1,
            global_dex_offset + loc_dex + 2, global_dex_offset + loc_dex + 2,
            global_dex_offset + loc_dex + 3, global_dex_offset + loc_dex + 3);

        _weight = _mm256_sub_ps(_weight, _mm256_and_ps(_mm256_castsi256_ps(_half_pass), real_bounds));
        _weight = _mm256_sub_ps(_weight, _centers);
        _weight = decx::utils::simd::_mm256_abs_ps(_weight);        // distance

        _not_exceeded._vf = _mm256_sub_ps(_weight, _radius);
        _not_exceeded._vi = _mm256_srai_epi32(_not_exceeded._vi, 31);

        _weight = _mm256_sub_ps(_mm256_set1_ps(1.f), _mm256_div_ps(_weight, _radius));

        store._vf = _mm256_mul_ps(recv._vf, _weight);
        store._vi = _mm256_and_si256(store._vi, _not_exceeded._vi);

        _mm256_store_pd(dst + loc_dex, store._vd);
        loc_dex += 4;
    }
}



_THREAD_FUNCTION_ void
decx::signal::CPUK::Gaussian_Window2D_cpl32_no_corrolation(const double* __restrict    src,
                                                           double* __restrict          dst, 
                                                           const float2                u, 
                                                           const float2                sigma, 
                                                           const uint2                 proc_dims,
                                                           const uint2                 real_bound,
                                                           const uint                  global_dex_offset_Y,
                                                           const uint                  pitch)
{
    size_t loc_dex = 0;

    __m256i axis_valueXY, _half_pass, dex_values;
    const uint2 _half_dims = make_uint2(real_bound.x / 2, real_bound.y / 2);

    const __m256i _half_dimsXY = _mm256_setr_epi32(real_bound.x / 2, real_bound.y / 2,
                                                   real_bound.x / 2, real_bound.y / 2,
                                                   real_bound.x / 2, real_bound.y / 2,
                                                   real_bound.x / 2, real_bound.y / 2),

                  _real_dimsXY = _mm256_setr_epi32(real_bound.x - 1, real_bound.y - 1,
                                                   real_bound.x - 1, real_bound.y - 1, 
                                                   real_bound.x - 1, real_bound.y - 1, 
                                                   real_bound.x - 1, real_bound.y - 1);
    const __m256 _us = _mm256_castsi256_ps(_mm256_set1_epi64x(*((size_t*)&u))),
                 _sigmas_sq = _mm256_setr_ps(-sigma.x * sigma.x * 2, -sigma.y * sigma.y * 2, 
                     -sigma.x * sigma.x * 2, -sigma.y * sigma.y * 2, 
                     -sigma.x * sigma.x * 2, -sigma.y * sigma.y * 2, 
                     -sigma.x * sigma.x * 2, -sigma.y * sigma.y * 2);
    __m256 g_weight;
    decx::utils::simd::xmm256_reg recv, store;

    for (int i = 0; i < proc_dims.y; ++i) 
    {
        loc_dex = i * pitch * 4;

        for (int j = 0; j < proc_dims.x; ++j) {
            recv._vd = _mm256_load_pd(src + loc_dex);

            dex_values = _mm256_setr_epi32(j * 4, global_dex_offset_Y + i,
                j * 4 + 1, global_dex_offset_Y + i,
                j * 4 + 2, global_dex_offset_Y + i,
                j * 4 + 3, global_dex_offset_Y + i);

            _half_pass = _mm256_sub_epi32(_half_dimsXY, dex_values);
            _half_pass = _mm256_srai_epi32(_half_pass, 31);
            axis_valueXY = _mm256_sub_epi32(dex_values, _mm256_and_si256(_real_dimsXY, _half_pass));

            g_weight = _mm256_sub_ps(_mm256_cvtepi32_ps(axis_valueXY), _us);
            g_weight = _mm256_div_ps(_mm256_mul_ps(g_weight, g_weight), _sigmas_sq);
            g_weight = _mm256_add_ps(g_weight, _mm256_permute_ps(g_weight, 0b10110001));
            g_weight = _mm256_exp_ps(g_weight);

            store._vf = _mm256_mul_ps(recv._vf, g_weight);

            _mm256_store_pd(dst + loc_dex, store._vd);
            loc_dex += 4;
        }
    }
}



_THREAD_FUNCTION_ void
decx::signal::CPUK::Gaussian_Window2D_cpl32(const double* __restrict    src,
                                            double* __restrict          dst, 
                                            const float2                u, 
                                            const float2                sigma, 
                                            const float                 p,
                                            const uint2                 proc_dims,
                                            const uint2                 real_bound,
                                            const uint                  global_dex_offset_Y,
                                            const uint                  pitch)
{
    size_t loc_dex = 0;

    __m256i axis_valueXY, _half_pass, dex_values;
    const uint2 _half_dims = make_uint2(real_bound.x / 2, real_bound.y / 2);
    const float _mid_term_coef = 2.f * p / (sigma.x * sigma.y),
                _exp_p_term_coef = -0.5f / (1.f - (p * p));

    const __m256i _half_dimsXY = _mm256_setr_epi32(real_bound.x / 2, real_bound.y / 2,
                                                   real_bound.x / 2, real_bound.y / 2,
                                                   real_bound.x / 2, real_bound.y / 2,
                                                   real_bound.x / 2, real_bound.y / 2),

                  _real_dimsXY = _mm256_setr_epi32(real_bound.x - 1, real_bound.y - 1,
                                                   real_bound.x - 1, real_bound.y - 1, 
                                                   real_bound.x - 1, real_bound.y - 1, 
                                                   real_bound.x - 1, real_bound.y - 1);
    const __m256 _us = _mm256_castsi256_ps(_mm256_set1_epi64x(*((size_t*)&u))),
                 _sigmas_sq = _mm256_setr_ps(sigma.x * sigma.x, sigma.y * sigma.y,
                     sigma.x * sigma.x, sigma.y * sigma.y,
                     sigma.x * sigma.x, sigma.y * sigma.y,
                     sigma.x * sigma.x, sigma.y * sigma.y);
    __m256 g_weight, reg_tmp;
    decx::utils::simd::xmm256_reg recv, store;

    for (int i = 0; i < proc_dims.y; ++i) 
    {
        loc_dex = i * pitch * 4;

        for (int j = 0; j < proc_dims.x; ++j) {
            recv._vd = _mm256_load_pd(src + loc_dex);

            dex_values = _mm256_setr_epi32(j * 4, global_dex_offset_Y + i,
                j * 4 + 1, global_dex_offset_Y + i,
                j * 4 + 2, global_dex_offset_Y + i,
                j * 4 + 3, global_dex_offset_Y + i);

            _half_pass = _mm256_sub_epi32(_half_dimsXY, dex_values);
            _half_pass = _mm256_srai_epi32(_half_pass, 31);
            axis_valueXY = _mm256_sub_epi32(dex_values, _mm256_and_si256(_real_dimsXY, _half_pass));

            reg_tmp = _mm256_sub_ps(_mm256_cvtepi32_ps(axis_valueXY), _us);
            g_weight = _mm256_div_ps(_mm256_mul_ps(reg_tmp, reg_tmp), _sigmas_sq);
            reg_tmp = _mm256_mul_ps(reg_tmp, _mm256_permute_ps(reg_tmp, 0b10110001));
            reg_tmp = _mm256_mul_ps(reg_tmp, _mm256_set1_ps(_mid_term_coef));
            // reg_tmp = (x1 - u1)(x2 - u2) / (sigma.x * sigma.y)

            g_weight = _mm256_sub_ps(_mm256_add_ps(g_weight, _mm256_permute_ps(g_weight, 0b10110001)), reg_tmp);
            g_weight = _mm256_exp_ps(_mm256_mul_ps(g_weight, _mm256_set1_ps(_exp_p_term_coef)));

            store._vf = _mm256_mul_ps(recv._vf, g_weight);

            _mm256_store_pd(dst + loc_dex, store._vd);
            loc_dex += 4;
        }
    }
}




_THREAD_FUNCTION_ void
decx::signal::CPUK::Cone_Window2D_cpl32(const double* __restrict        src, 
                                        double* __restrict              dst, 
                                        const uint2                     origin, 
                                        const float                     radius, 
                                        const uint2                     proc_dims,
                                        const uint2                     real_bound, 
                                        const uint                      global_dex_offset_Y, 
                                        const uint                      pitch)
{
    size_t loc_dex = 0;
    decx::utils::simd::xmm256_reg recv, store, _weights, _not_exceeded;

    __m256i current_dex, _hfps;
    const __m256i _half_dims = _mm256_setr_epi32(real_bound.x / 2, real_bound.y / 2,
        real_bound.x / 2, real_bound.y / 2,
        real_bound.x / 2, real_bound.y / 2,
        real_bound.x / 2, real_bound.y / 2),
                  _bounds = _mm256_setr_epi32(real_bound.x - 1, real_bound.y - 1,
                      real_bound.x - 1, real_bound.y - 1,
                      real_bound.x - 1, real_bound.y - 1,
                      real_bound.x - 1, real_bound.y - 1);
    const __m256 _radius = _mm256_set1_ps(radius);

    for (int i = 0; i < proc_dims.y; ++i)
    {
        loc_dex = i * pitch * 4;

        for (int j = 0; j < proc_dims.x; ++j) {
            recv._vd = _mm256_load_pd(src + loc_dex);

            current_dex = _mm256_setr_epi32(j * 4, global_dex_offset_Y + i,
                j * 4 + 1, global_dex_offset_Y + i,
                j * 4 + 2, global_dex_offset_Y + i,
                j * 4 + 3, global_dex_offset_Y + i);

            _hfps = _mm256_sub_epi32(_half_dims, current_dex);
            _hfps = _mm256_srai_epi32(_hfps, 31);

            _weights._vf = _mm256_cvtepi32_ps(_mm256_sub_epi32(current_dex, _mm256_and_si256(_hfps, _bounds)));
            _weights._vf = _mm256_mul_ps(_weights._vf, _weights._vf);       // [distX ^ 2, distY ^ 2]
            _weights._vf = _mm256_sqrt_ps(_mm256_add_ps(_weights._vf, _mm256_permute_ps(_weights._vf, 0b10110001)));        // distance

            _not_exceeded._vf = _mm256_sub_ps(_weights._vf, _radius);
            _not_exceeded._vi = _mm256_srai_epi32(_not_exceeded._vi, 31);

            _weights._vf = _mm256_sub_ps(_mm256_set1_ps(1.f), _mm256_div_ps(_weights._vf, _radius));

            store._vf = _mm256_mul_ps(recv._vf, _weights._vf);
            store._vi = _mm256_and_si256(store._vi, _not_exceeded._vi);
            _mm256_store_pd(dst + loc_dex, store._vd);
            loc_dex += 4;
        }
    }
}




_DECX_API_ de::DH
de::signal::cpu::Gaussian_Window1D(de::Vector& src, de::Vector& dst, const float u, const float sigma)
{
    de::DH handle;
    if (!decx::cpI.is_init) {
        decx::err::CPU_Not_init(&handle);
        Print_Error_Message(4, CPU_NOT_INIT);
        return handle;
    }

    decx::_Vector* _src = dynamic_cast<decx::_Vector*>(&src);
    decx::_Vector* _dst = dynamic_cast<decx::_Vector*>(&dst);

    const size_t proc_len = _src->_length / 4;
    if (proc_len > decx::cpI.cpu_concurrency) {
        decx::utils::_thread_arrange_1D t1D(decx::cpI.cpu_concurrency);
        decx::utils::frag_manager f_mgr;
        decx::utils::frag_manager_gen(&f_mgr, proc_len, t1D.total_thread);

        const double* _loc_src = reinterpret_cast<const double*>(_src->Vec.ptr);
        double* _loc_dst = reinterpret_cast<double*>(_dst->Vec.ptr);
        size_t _global_ptr_offset = 0;

        for (int i = 0; i < t1D.total_thread - 1; ++i) {
            t1D._async_thread[i] = decx::cpu::register_task(&decx::thread_pool,
                decx::signal::CPUK::Gaussian_Window1D_cpl32,
                _loc_src, _loc_dst, u, sigma,
                f_mgr.frag_num, _src->length, _global_ptr_offset);

            _global_ptr_offset += f_mgr.frag_len * 4;
            _loc_src += _global_ptr_offset;
            _loc_dst += _global_ptr_offset;
        }
        const size_t _L = f_mgr.is_left ? f_mgr.frag_left_over : f_mgr.frag_len;
        t1D._async_thread[t1D.total_thread - 1] = decx::cpu::register_task(&decx::thread_pool,
            decx::signal::CPUK::Gaussian_Window1D_cpl32,
            _loc_src, _loc_dst, u, sigma,
            _L, _src->length, _global_ptr_offset);

        t1D.__sync_all_threads();
    }
    else {
        decx::signal::CPUK::Gaussian_Window1D_cpl32((const double*)_src->Vec.ptr, (double*)_dst->Vec.ptr, u, sigma,
            proc_len, _src->length, 0);
    }

    decx::err::Success(&handle);
    return handle;
}




_DECX_API_ de::DH
de::signal::cpu::Triangular_Window1D(de::Vector& src, de::Vector& dst, const long long center, const size_t radius)
{
    de::DH handle;
    if (!decx::cpI.is_init) {
        decx::err::CPU_Not_init(&handle);
        Print_Error_Message(4, CPU_NOT_INIT);
        return handle;
    }

    decx::_Vector* _src = dynamic_cast<decx::_Vector*>(&src);
    decx::_Vector* _dst = dynamic_cast<decx::_Vector*>(&dst);

    const size_t proc_len = _src->_length / 4;
    if (proc_len > decx::cpI.cpu_concurrency) {
        decx::utils::_thread_arrange_1D t1D(decx::cpI.cpu_concurrency);
        decx::utils::frag_manager f_mgr;
        decx::utils::frag_manager_gen(&f_mgr, proc_len, t1D.total_thread);

        const double* _loc_src = reinterpret_cast<const double*>(_src->Vec.ptr);
        double* _loc_dst = reinterpret_cast<double*>(_dst->Vec.ptr);
        size_t _global_ptr_offset = 0;

        for (int i = 0; i < t1D.total_thread - 1; ++i) {
            t1D._async_thread[i] = decx::cpu::register_task(&decx::thread_pool,
                decx::signal::CPUK::Triangular_Window1D_cpl32,
                _loc_src, _loc_dst, center, radius,
                f_mgr.frag_num, _src->length, _global_ptr_offset);

            _global_ptr_offset += f_mgr.frag_len * 4;
            _loc_src += _global_ptr_offset;
            _loc_dst += _global_ptr_offset;
        }
        const size_t _L = f_mgr.is_left ? f_mgr.frag_left_over : f_mgr.frag_len;
        t1D._async_thread[t1D.total_thread - 1] = decx::cpu::register_task(&decx::thread_pool,
            decx::signal::CPUK::Triangular_Window1D_cpl32,
            _loc_src, _loc_dst, center, radius,
            _L, _src->length, _global_ptr_offset);

        t1D.__sync_all_threads();
    }
    else {
        decx::signal::CPUK::Triangular_Window1D_cpl32((const double*)_src->Vec.ptr, (double*)_dst->Vec.ptr, center, radius,
            proc_len, _src->length, 0);
    }

    decx::err::Success(&handle);
    return handle;
}



_DECX_API_ de::DH 
de::signal::cpu::Gaussian_Window2D(de::Matrix& src, de::Matrix& dst, const de::Point2D_f u, const de::Point2D_f sigma, const float p)
{
    de::DH handle;
    if (!decx::cpI.is_init) {
        decx::err::CPU_Not_init(&handle);
        Print_Error_Message(4, CPU_NOT_INIT);
        return handle;
    }

    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);
    decx::_Matrix* _dst = dynamic_cast<decx::_Matrix*>(&dst);

    const uint pitch = _src->pitch / 4;
    decx::utils::_thread_arrange_1D t1D(decx::cpI.cpu_concurrency);
    decx::utils::frag_manager f_mgr;
    decx::utils::frag_manager_gen(&f_mgr, _src->height, t1D.total_thread);

    const size_t frag_size = (size_t)f_mgr.frag_len * (size_t)pitch * 4;

    uint2 _proc_dims = make_uint2(pitch, f_mgr.frag_len);
    const uint2 real_bound = make_uint2(_src->width, _src->height);

    bool _corrolated = (p != 0);
    if (_corrolated) {
        if (!(p < 1.f && p > -1.f)) {
            decx::err::InvalidParam(&handle);
            Print_Error_Message(4, INVALID_PARAM);
            return handle;
        }
    }

    if (_src->height > decx::cpI.cpu_concurrency) {
        const double* _loc_src = reinterpret_cast<const double*>(_src->Mat.ptr);
        double* _loc_dst = reinterpret_cast<double*>(_dst->Mat.ptr);

        for (int i = 0; i < t1D.total_thread - 1; ++i) {
            if (_corrolated) {
                t1D._async_thread[i] = decx::cpu::register_task(&decx::thread_pool, decx::signal::CPUK::Gaussian_Window2D_cpl32,
                    _loc_src, _loc_dst,
                    make_float2(u.x, u.y), make_float2(sigma.x, sigma.y), p,
                    _proc_dims, real_bound,
                    f_mgr.frag_len * i, pitch);
            }
            else {
                t1D._async_thread[i] = decx::cpu::register_task(&decx::thread_pool, decx::signal::CPUK::Gaussian_Window2D_cpl32_no_corrolation,
                    _loc_src, _loc_dst,
                    make_float2(u.x, u.y), make_float2(sigma.x, sigma.y),
                    _proc_dims, real_bound,
                    f_mgr.frag_len * i, pitch);
            }

            _loc_src += frag_size;
            _loc_dst += frag_size;
        }
        _proc_dims.y = f_mgr.is_left ? f_mgr.frag_left_over : f_mgr.frag_len;
        if (_corrolated) {
            t1D._async_thread[t1D.total_thread - 1] = decx::cpu::register_task(&decx::thread_pool, decx::signal::CPUK::Gaussian_Window2D_cpl32,
                _loc_src, _loc_dst,
                make_float2(u.x, u.y), make_float2(sigma.x, sigma.y), p,
                _proc_dims, real_bound,
                f_mgr.frag_len * (t1D.total_thread - 1), pitch);
        }
        else {
            t1D._async_thread[t1D.total_thread - 1] = decx::cpu::register_task(&decx::thread_pool, decx::signal::CPUK::Gaussian_Window2D_cpl32_no_corrolation,
                _loc_src, _loc_dst,
                make_float2(u.x, u.y), make_float2(sigma.x, sigma.y),
                _proc_dims, real_bound,
                f_mgr.frag_len * (t1D.total_thread - 1), pitch);
        }

        t1D.__sync_all_threads();
    }
    else {
        if (_corrolated) {
            decx::signal::CPUK::Gaussian_Window2D_cpl32(reinterpret_cast<const double*>(_src->Mat.ptr),
                reinterpret_cast<double*>(_dst->Mat.ptr), make_float2(u.x, u.y), make_float2(sigma.x, sigma.y), p, _proc_dims, real_bound,
                0, pitch);
        }
        else {
            decx::signal::CPUK::Gaussian_Window2D_cpl32_no_corrolation(reinterpret_cast<const double*>(_src->Mat.ptr),
                reinterpret_cast<double*>(_dst->Mat.ptr), make_float2(u.x, u.y), make_float2(sigma.x, sigma.y), _proc_dims, real_bound,
                0, pitch);
        }
    }
    decx::err::Success(&handle);
    return handle;
}




_DECX_API_ de::DH 
de::signal::cpu::Cone_Window2D(de::Matrix& src, de::Matrix& dst, const de::Point2D origin, const float radius)
{
    de::DH handle;
    if (!decx::cpI.is_init) {
        decx::err::CPU_Not_init(&handle);
        Print_Error_Message(4, CPU_NOT_INIT);
        return handle;
    }

    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);
    decx::_Matrix* _dst = dynamic_cast<decx::_Matrix*>(&dst);

    const uint pitch = _src->pitch / 4;
    decx::utils::_thread_arrange_1D t1D(decx::cpI.cpu_concurrency);
    decx::utils::frag_manager f_mgr;
    decx::utils::frag_manager_gen(&f_mgr, _src->height, t1D.total_thread);

    const size_t frag_size = (size_t)f_mgr.frag_len * (size_t)pitch * 4;

    uint2 _proc_dims = make_uint2(pitch, f_mgr.frag_len);
    const uint2 real_bound = make_uint2(_src->width, _src->height);

    if (_src->height > decx::cpI.cpu_concurrency) {
        const double* _loc_src = reinterpret_cast<const double*>(_src->Mat.ptr);
        double* _loc_dst = reinterpret_cast<double*>(_dst->Mat.ptr);

        for (int i = 0; i < t1D.total_thread - 1; ++i) {
            t1D._async_thread[i] = decx::cpu::register_task(&decx::thread_pool, decx::signal::CPUK::Cone_Window2D_cpl32,
                _loc_src, _loc_dst, make_uint2(origin.x, origin.y), radius,
                _proc_dims, real_bound,
                f_mgr.frag_len * i, pitch);

            _loc_src += frag_size;
            _loc_dst += frag_size;
        }
        _proc_dims.y = f_mgr.is_left ? f_mgr.frag_left_over : f_mgr.frag_len;
        t1D._async_thread[t1D.total_thread - 1] = decx::cpu::register_task(&decx::thread_pool, decx::signal::CPUK::Cone_Window2D_cpl32,
            _loc_src, _loc_dst, make_uint2(origin.x, origin.y), radius,
            _proc_dims, real_bound,
            f_mgr.frag_len * (t1D.total_thread - 1), pitch);

        t1D.__sync_all_threads();
    }
    else {
        decx::signal::CPUK::Cone_Window2D_cpl32(reinterpret_cast<const double*>(_src->Mat.ptr), reinterpret_cast<double*>(_dst->Mat.ptr), 
            make_uint2(origin.x, origin.y), radius, _proc_dims, real_bound, 0, pitch);
    }
    decx::err::Success(&handle);
    return handle;
}