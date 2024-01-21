/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#include "W_table.h"


_THREAD_FUNCTION_ void decx::dsp::fft::CPUK::_W_table_gen_cplxf(double* __restrict      _W_table, 
                                                                const uint32_t          _proc_len_v4,
                                                                const uint64_t          _signal_length, 
                                                                const uint64_t          _index_start)
{
    decx::utils::simd::xmm128_reg _real_v4, _image_v4;
    decx::utils::simd::xmm256_reg _res;

    uint32_t _base_dex = _index_start;

    for (uint32_t i = 0; i < _proc_len_v4; ++i) {
        _real_v4._vf = _mm_setr_ps(_base_dex, _base_dex + 1, _base_dex + 2, _base_dex + 3);
        _image_v4._vf = _real_v4._vf;

        _real_v4._vf = _mm_mul_ps(_mm_div_ps(_real_v4._vf, _mm_set1_ps(_signal_length)), _mm_set1_ps(Two_Pi));
        _image_v4._vf = _mm_mul_ps(_mm_div_ps(_image_v4._vf, _mm_set1_ps(_signal_length)), _mm_set1_ps(Two_Pi));
        _real_v4._vf = decx::utils::simd::_mm_cos_ps(_real_v4._vf);
        _image_v4._vf = decx::utils::simd::_mm_sin_ps(_image_v4._vf);

        _res._vf = _mm256_permute2f128_ps(_mm256_castps128_ps256(_real_v4._vf), _mm256_castps128_ps256(_image_v4._vf), 0b00100000);
        _res._vf = _mm256_permutevar8x32_ps(_res._vf, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));

        _mm256_store_pd(_W_table + (i << 2), _res._vd);

        _base_dex += 4;
    }
}



template <>
void decx::dsp::fft::Rotational_Factors_Table<float>::_alloc_table_from_scratch(const uint64_t _len, de::DH *handle)
{
    this->_actual_len = _len;
    this->_alloc_len = decx::utils::ceil<uint64_t>(_len, 4) * 4;
    if (decx::alloc::_host_virtual_page_malloc(&_W_table, this->_alloc_len * sizeof(double), true)) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_ALLOCATION, ALLOC_FAIL);
        return;
    }
}


template <>
void decx::dsp::fft::Rotational_Factors_Table<float>::_realloc_table(const uint64_t _new_len, de::DH* handle)
{
    const uint64_t new_alloc_len = decx::utils::ceil<uint64_t>(_new_len, 4) * 4;
    
    if (new_alloc_len > this->_alloc_len) {
        decx::alloc::_host_virtual_page_dealloc(&this->_W_table);
        if (decx::alloc::_host_virtual_page_malloc(&this->_W_table, new_alloc_len * sizeof(double), true)) {
            decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_ALLOCATION, ALLOC_FAIL);
            return;
        }
    }

    this->_alloc_len = new_alloc_len;
    this->_actual_len = _new_len;
}


template <>
void decx::dsp::fft::Rotational_Factors_Table<float>::_alloc_table(const uint64_t _len, de::DH *handle)
{
    if (this->_W_table.ptr == NULL){
        this->_alloc_table_from_scratch(_len, handle);
    }
    else{
        this->_realloc_table(_len, handle);
    }
}



template <>
void decx::dsp::fft::Rotational_Factors_Table<float>::_generate_table(decx::utils::_thr_1D* t1D)
{
    decx::utils::frag_manager _f_mgr_WT;
    decx::utils::frag_manager_gen(&_f_mgr_WT, this->_alloc_len / 4, t1D->total_thread);
    
    double* _loc_ptr_WT = (double*)this->_W_table.ptr;

    for (int i = 0; i < _f_mgr_WT.frag_num - 1; ++i) {
        t1D->_async_thread[i] = decx::cpu::register_task_default(decx::dsp::fft::CPUK::_W_table_gen_cplxf, 
                                                                 _loc_ptr_WT,
                                                                 _f_mgr_WT.frag_len, 
                                                                 this->_actual_len, 
                                                                 i * (_f_mgr_WT.frag_len << 2));

        _loc_ptr_WT += (_f_mgr_WT.frag_len << 2);
    }
    const uint32_t _L_WT = _f_mgr_WT.is_left ? _f_mgr_WT.frag_left_over : _f_mgr_WT.frag_len;

    t1D->_async_thread[_f_mgr_WT.frag_num - 1] = 
        decx::cpu::register_task_default(decx::dsp::fft::CPUK::_W_table_gen_cplxf, 
                                         _loc_ptr_WT,
                                         _L_WT, 
                                         this->_actual_len, 
                                         (_f_mgr_WT.frag_num - 1) * (_f_mgr_WT.frag_len << 2));

    t1D->__sync_all_threads();
}



template <>
void decx::dsp::fft::Rotational_Factors_Table<float>::_release()
{
    decx::alloc::_host_virtual_page_dealloc(&this->_W_table);
}


template <>
decx::dsp::fft::Rotational_Factors_Table<float>::~Rotational_Factors_Table()
{
    this->_release();
}