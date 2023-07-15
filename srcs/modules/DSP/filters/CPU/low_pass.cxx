/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#include "low_pass.h"


_THREAD_FUNCTION_ void
decx::signal::CPUK::ideal_LP1D_cpl32_ST(const double* __restrict    src, 
                                        double* __restrict          dst, 
                                        const size_t                cutoff_freq,
                                        const size_t                _proc_len,
                                        const size_t                real_bound, 
                                        const size_t                global_dex_offset)
{
    size_t loc_dex = 0;

    decx::utils::simd::xmm256_reg recv, store;
    const __m256i _lower_bound = _mm256_set1_epi64x(cutoff_freq),
                  _upper_bound = _mm256_set1_epi64x(real_bound - cutoff_freq - 1);
    __m256i _is_eff, _current_dex_global;

    for (int i = 0; i < _proc_len; ++i) {
        recv._vd = _mm256_load_pd(src + loc_dex);

        _current_dex_global = _mm256_setr_epi64x(global_dex_offset + loc_dex, 
                                                 global_dex_offset + loc_dex + 1, 
                                                 global_dex_offset + loc_dex + 2, 
                                                 global_dex_offset + loc_dex + 3);
        _is_eff = _mm256_sub_epi64(_current_dex_global, _lower_bound);      // real_dex < cutoff_freq ? 1 : 0
        _is_eff = _mm256_or_si256(_is_eff, _mm256_sub_epi64(_upper_bound, _current_dex_global));

        _is_eff = _mm256_srai_epi32(_is_eff, 31);
        _is_eff = _mm256_shuffle_epi32(_is_eff, 0b11110101);

        store._vi = _mm256_and_si256(recv._vi, _is_eff);

        _mm256_store_pd(dst + loc_dex, store._vd);

        loc_dex += 4;
    }
}




_THREAD_FUNCTION_ void
decx::signal::CPUK::ButterWorth_Window2D_cpl32(const double* __restrict    src,
                                            double* __restrict          dst, 
                                            const int                   order, 
                                            const float                 cutoff_freq,
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
    
    __m256 b_weight, reg_tmp;
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

            reg_tmp = _mm256_cvtepi32_ps(axis_valueXY);
            b_weight = _mm256_mul_ps(reg_tmp, reg_tmp);
            // distance
            reg_tmp = _mm256_sqrt_ps(_mm256_add_ps(b_weight, _mm256_permute_ps(b_weight, 0b10110001)));

            b_weight = _mm256_pow_ps(_mm256_div_ps(reg_tmp, _mm256_set1_ps(cutoff_freq)), _mm256_set1_ps(2 * order));
            b_weight = _mm256_div_ps(_mm256_set1_ps(1.f), _mm256_add_ps(_mm256_set1_ps(1.f), b_weight));            //b_weight = _mm256_set1_ps(1);
            store._vf = _mm256_mul_ps(recv._vf, b_weight);

            _mm256_store_pd(dst + loc_dex, store._vd);
            loc_dex += 4;
        }
    }
}




_THREAD_FUNCTION_ void
decx::signal::CPUK::ideal_LP2D_cpl32_ST(const double* __restrict    src, 
                                        double* __restrict          dst, 
                                        const uint2                 _proc_dims, 
                                        const uint2                 real_bound,
                                        const uint2                 cutoff_freq, 
                                        const uint                  pitch,
                                        const uint2                 global_dex_offset)
{
    size_t dex = 0, row_base = 0;
    
    __m256i _is_eff;
    decx::utils::simd::xmm256_reg recv, store;
    // (x1, y1), (x2, y2), (x3, y3), (x4, y4)
    __m256i current_dex;
    const __m256i _lower_bounds = _mm256_set1_epi64x(*((size_t*)&cutoff_freq)),
                  _upper_bounds = _mm256_setr_epi32(real_bound.x - cutoff_freq.x, real_bound.y - cutoff_freq.y,
                                                    real_bound.x - cutoff_freq.x, real_bound.y - cutoff_freq.y,
                                                    real_bound.x - cutoff_freq.x, real_bound.y - cutoff_freq.y,
                                                    real_bound.x - cutoff_freq.x, real_bound.y - cutoff_freq.y);

    uint row_dex = 0;

    for (int i = 0; i < _proc_dims.y; ++i) 
    {
        row_dex = global_dex_offset.y + i;
        dex = row_base;
        for (int j = 0; j < _proc_dims.x; ++j) 
        {
            recv._vd = _mm256_load_pd(src + dex);

            current_dex = _mm256_setr_epi32(global_dex_offset.x + j * 4, row_dex,
                                            global_dex_offset.x + j * 4 + 1, row_dex,
                                            global_dex_offset.x + j * 4 + 2, row_dex,
                                            global_dex_offset.x + j * 4 + 3, row_dex);
            _is_eff = _mm256_sub_epi32(current_dex, _lower_bounds);
            _is_eff = _mm256_or_si256(_is_eff, _mm256_sub_epi32(_upper_bounds, current_dex));
            _is_eff = _mm256_and_si256(_is_eff, _mm256_shuffle_epi32(_is_eff, 0b10110001));
            _is_eff = _mm256_srai_epi32(_is_eff, 31);

            store._vi = _mm256_and_si256(recv._vi, _is_eff);

            _mm256_store_pd(dst + dex, store._vd);
            dex += 4;
        }
        row_base += pitch * 4;
    }
}



_DECX_API_ de::DH
de::signal::cpu::LowPass1D_Ideal(de::Vector& src, de::Vector& dst, const size_t cutoff_frequency)
{
    de::DH handle;
    if (!decx::cpu::_is_CPU_init()) {
        decx::err::CPU_Not_init(&handle);
        Print_Error_Message(4, CPU_NOT_INIT);
        return handle;
    }

    decx::_Vector* _src = dynamic_cast<decx::_Vector*>(&src);
    decx::_Vector* _dst = dynamic_cast<decx::_Vector*>(&dst);
    
    const size_t proc_len = _src->_length / 4;
    if (proc_len > decx::cpu::_get_permitted_concurrency()) {
        decx::utils::_thread_arrange_1D t1D(decx::cpu::_get_permitted_concurrency());
        decx::utils::frag_manager f_mgr;
        decx::utils::frag_manager_gen(&f_mgr, proc_len, t1D.total_thread);

        const double* _loc_src = reinterpret_cast<const double*>(_src->Vec.ptr);
        double* _loc_dst = reinterpret_cast<double*>(_dst->Vec.ptr);
        size_t _global_ptr_offset = 0;

        for (int i = 0; i < t1D.total_thread - 1; ++i) {
            t1D._async_thread[i] = decx::cpu::register_task_default(
                decx::signal::CPUK::ideal_LP1D_cpl32_ST,
                _loc_src, _loc_dst, cutoff_frequency,
                f_mgr.frag_num, _src->length, _global_ptr_offset);

            _global_ptr_offset += f_mgr.frag_len * 4;
            _loc_src += _global_ptr_offset;
            _loc_dst += _global_ptr_offset;
        }
        const size_t _L = f_mgr.is_left ? f_mgr.frag_left_over : f_mgr.frag_len;
        t1D._async_thread[t1D.total_thread - 1] = decx::cpu::register_task_default(
            decx::signal::CPUK::ideal_LP1D_cpl32_ST,
            _loc_src, _loc_dst, cutoff_frequency,
            _L, _src->length, _global_ptr_offset);

        t1D.__sync_all_threads();
    }
    else {
        decx::signal::CPUK::ideal_LP1D_cpl32_ST((const double*)_src->Vec.ptr, (double*)_dst->Vec.ptr, cutoff_frequency,
            proc_len, _src->length, 0);
    }
    
    decx::err::Success(&handle);
    return handle;
}




_DECX_API_ de::DH
de::signal::cpu::LowPass2D_Ideal(de::Matrix& src, de::Matrix& dst, const de::Point2D cutoff_frequency)
{
    de::DH handle;
    if (!decx::cpu::_is_CPU_init()) {
        decx::err::CPU_Not_init(&handle);
        Print_Error_Message(4, CPU_NOT_INIT);
        return handle;
    }

    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);
    decx::_Matrix* _dst = dynamic_cast<decx::_Matrix*>(&dst);

    const uint pitch = _src->Pitch() / 4;
    decx::utils::_thread_arrange_1D t1D(decx::cpu::_get_permitted_concurrency());
    decx::utils::frag_manager f_mgr;
    decx::utils::frag_manager_gen(&f_mgr, _src->Height(), t1D.total_thread);

    const size_t frag_size = (size_t)f_mgr.frag_len * (size_t)pitch * 4;

    uint2 _proc_dims = make_uint2(pitch, f_mgr.frag_len);
    const uint2 real_bound = make_uint2(_src->Width(), _src->Height());

    if (_src->Height() > decx::cpu::_get_permitted_concurrency()) {
        const double* _loc_src = reinterpret_cast<const double*>(_src->Mat.ptr);
        double* _loc_dst = reinterpret_cast<double*>(_dst->Mat.ptr);

        for (int i = 0; i < t1D.total_thread - 1; ++i) {
            t1D._async_thread[i] = decx::cpu::register_task_default( decx::signal::CPUK::ideal_LP2D_cpl32_ST,
                _loc_src, _loc_dst,
                _proc_dims, real_bound,
                make_uint2(cutoff_frequency.x, cutoff_frequency.y),
                pitch,
                make_uint2(0, f_mgr.frag_len * i));

            _loc_src += frag_size;
            _loc_dst += frag_size;
        }
        _proc_dims.y = f_mgr.is_left ? f_mgr.frag_left_over : f_mgr.frag_len;
        t1D._async_thread[t1D.total_thread - 1] = decx::cpu::register_task_default( decx::signal::CPUK::ideal_LP2D_cpl32_ST,
            _loc_src, _loc_dst,
            _proc_dims, real_bound,
            make_uint2(cutoff_frequency.x, cutoff_frequency.y),
            pitch,
            make_uint2(0, f_mgr.frag_len * (t1D.total_thread - 1)));

        t1D.__sync_all_threads();
    }
    else {
        _proc_dims.y = _src->Height();
        decx::signal::CPUK::ideal_LP2D_cpl32_ST(reinterpret_cast<const double*>(_src->Mat.ptr), 
            reinterpret_cast<double*>(_dst->Mat.ptr), _proc_dims, real_bound,
            make_uint2(cutoff_frequency.x, cutoff_frequency.y), pitch, make_uint2(0, 0));
    }
    decx::err::Success(&handle);
    return handle;
}



_DECX_API_ de::DH 
de::signal::cpu::ButterWorth_LP2D(de::Matrix& src, de::Matrix& dst, const float cutoff_freq, const int order)
{
    de::DH handle;
    if (!decx::cpu::_is_CPU_init()) {
        decx::err::CPU_Not_init(&handle);
        Print_Error_Message(4, CPU_NOT_INIT);
        return handle;
    }

    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);
    decx::_Matrix* _dst = dynamic_cast<decx::_Matrix*>(&dst);

    const uint pitch = _src->Pitch() / 4;
    decx::utils::_thread_arrange_1D t1D(decx::cpu::_get_permitted_concurrency());
    decx::utils::frag_manager f_mgr;
    decx::utils::frag_manager_gen(&f_mgr, _src->Height(), t1D.total_thread);

    const size_t frag_size = (size_t)f_mgr.frag_len * (size_t)pitch * 4;

    uint2 _proc_dims = make_uint2(pitch, f_mgr.frag_len);
    const uint2 real_bound = make_uint2(_src->Width(), _src->Height());

    if (_src->Height() > decx::cpu::_get_permitted_concurrency()) {
        const double* _loc_src = reinterpret_cast<const double*>(_src->Mat.ptr);
        double* _loc_dst = reinterpret_cast<double*>(_dst->Mat.ptr);

        for (int i = 0; i < t1D.total_thread - 1; ++i) {
            t1D._async_thread[i] = decx::cpu::register_task_default( decx::signal::CPUK::ButterWorth_Window2D_cpl32,
                _loc_src, _loc_dst, order, cutoff_freq, _proc_dims, real_bound,
                f_mgr.frag_len * i, pitch);

            _loc_src += frag_size;
            _loc_dst += frag_size;
        }
        _proc_dims.y = f_mgr.is_left ? f_mgr.frag_left_over : f_mgr.frag_len;
        t1D._async_thread[t1D.total_thread - 1] = decx::cpu::register_task_default( decx::signal::CPUK::ButterWorth_Window2D_cpl32,
            _loc_src, _loc_dst, order, cutoff_freq, _proc_dims, real_bound,
            f_mgr.frag_len * (t1D.total_thread - 1), pitch);

        t1D.__sync_all_threads();
    }
    else {
        _proc_dims.y = _src->Height();
        decx::signal::CPUK::ButterWorth_Window2D_cpl32(reinterpret_cast<const double*>(_src->Mat.ptr),
            reinterpret_cast<double*>(_dst->Mat.ptr), order, cutoff_freq, _proc_dims, real_bound,
            0, pitch);
    }
    decx::err::Success(&handle);
    return handle;
}