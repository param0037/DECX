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


#include "../../1D/FFT1D_kernels.h"
#include "../../2D/FFT2D_kernel_utils.h"
#include "../FFT3D_kernel_utils.h"
#include "../FFT3D_kernels.h"


template <typename _type_in, bool _conj> 
_THREAD_FUNCTION_ void decx::dsp::fft::CPUK::
_FFT3D_smaller_2rows_cplxd(const _type_in* __restrict src_head_ptr, de::CPd* __restrict dst_head_ptr,
                           const decx::dsp::fft::FKT1D* _tiles,     const uint32_t _pitch_src, 
                           const uint32_t _pitch_dst,               const uint32_t _proc_H_r1,
                           const uint32_t start_dex,                const decx::dsp::fft::cpu_FFT3D_subproc<double>* _FFT_info)
{
    decx::utils::double_buffer_manager _double_buffer(_tiles->get_tile1<void>(), _tiles->get_tile2<void>());
    decx::utils::frag_manager _f_mgr_H;
    decx::utils::frag_manager_gen_from_fragLen(&_f_mgr_H, _proc_H_r1, 2);
    
    uint32_t start_dex_H = start_dex;
    
    _tiles->flush();
    
    for (uint32_t i = 0; i < _f_mgr_H.frag_num; ++i) 
    {
        _double_buffer.reset_buffer1_leading();
        if constexpr (std::is_same_v<_type_in, double>)
        {
            // Load and transpose data from global memory
            decx::dsp::fft::CPUK::load_entire_row_transpose_fp64_zip(src_head_ptr,  &_double_buffer,
                                                                     _tiles,        decx::utils::ceil<uint32_t>(_FFT_info->_FFT_info.get_signal_len(), 2),
                                                                     _pitch_src,    &_FFT_info->_FFT_zip_info_LDG,
                                                                     start_dex_H, i == (_f_mgr_H.frag_num - 1) ? _f_mgr_H.last_frag_len : 2);

            // Call vec4 smaller FFT
		    decx::dsp::fft::CPUK::_FFT1D_caller_cplxd64_1st_R2C(_double_buffer.get_leading_ptr<double>(), 
															    _double_buffer.get_lagging_ptr<de::CPd>(),
															    _FFT_info->_FFT_info.get_kernel_info_ptr(0));
        }
        else if constexpr (std::is_same_v<_type_in, uint8_t>) {
            // Load and transpose data from global memory
            decx::dsp::fft::CPUK::load_entire_row_transpose_u8_fp64_zip(src_head_ptr,           &_double_buffer,
                                                                        _tiles,                 _pitch_src >> 1, 
                                                                        _pitch_src,             &_FFT_info->_FFT_zip_info_LDG,
                                                                        start_dex_H, i == (_f_mgr_H.frag_num - 1) ? _f_mgr_H.last_frag_len : 2);

            // Call vec4 smaller FFT
		    decx::dsp::fft::CPUK::_FFT1D_caller_cplxd64_1st_R2C(_double_buffer.get_leading_ptr<double>(), 
															    _double_buffer.get_lagging_ptr<de::CPd>(),
															    _FFT_info->_FFT_info.get_kernel_info_ptr(0));
        }
        else {
            // Load and transpose data from global memory
            decx::dsp::fft::CPUK::load_entire_row_transpose_cplxd_zip<false>(src_head_ptr,      &_double_buffer,
                                                                             _tiles,            _pitch_src >> 1,
                                                                             _pitch_src,        &_FFT_info->_FFT_zip_info_LDG,
                                                                             start_dex_H,       _FFT_info->_FFT_info.get_signal_len(),
                                                                             i == (_f_mgr_H.frag_num - 1) ? _f_mgr_H.last_frag_len : 2);

            // Call vec4 smaller FFT
		    decx::dsp::fft::CPUK::_FFT1D_caller_cplxd64_1st_C2C(_double_buffer.get_leading_ptr<de::CPd>(), 
															    _double_buffer.get_lagging_ptr<de::CPd>(),
															    _FFT_info->_FFT_info.get_kernel_info_ptr(0));
        }
        _double_buffer.update_states();

		for (uint32_t _FFT_index = 1; _FFT_index < _FFT_info->_FFT_info.get_kernel_call_num(); ++_FFT_index) {
			decx::dsp::fft::CPUK::_FFT1D_caller_cplxd64_mid_C2C(_double_buffer.get_leading_ptr<de::CPd>(), 
																_double_buffer.get_lagging_ptr<de::CPd>(),
																_FFT_info->_FFT_info.get_W_table<de::CPd>(),
																_FFT_info->_FFT_info.get_kernel_info_ptr(_FFT_index));

			_double_buffer.update_states();
		}

        // Store back to global memory
        decx::dsp::fft::CPUK::
            store_entire_row_transpose_cplxd_zip<_conj>(&_double_buffer,  dst_head_ptr, 
                                                        _tiles,           decx::utils::ceil<uint32_t>(_FFT_info->_FFT_info.get_signal_len(), 2), 
                                                        _pitch_dst,       &_FFT_info->_FFT_zip_info_STG, 
                                                        start_dex_H,      i == (_f_mgr_H.frag_num - 1) ? _f_mgr_H.last_frag_len : 2);

        start_dex_H += 2;
    }
}


template _THREAD_FUNCTION_ void decx::dsp::fft::CPUK::_FFT3D_smaller_2rows_cplxd<double, true>(const double* __restrict, de::CPd* __restrict,
    const decx::dsp::fft::FKT1D*, const uint32_t, const uint32_t, const uint32_t, const uint32_t, const decx::dsp::fft::cpu_FFT3D_subproc<double>*);

template _THREAD_FUNCTION_ void decx::dsp::fft::CPUK::_FFT3D_smaller_2rows_cplxd<double, false>(const double* __restrict, de::CPd* __restrict,
    const decx::dsp::fft::FKT1D*, const uint32_t, const uint32_t, const uint32_t, const uint32_t, const decx::dsp::fft::cpu_FFT3D_subproc<double>*);

template _THREAD_FUNCTION_ void decx::dsp::fft::CPUK::_FFT3D_smaller_2rows_cplxd<de::CPd, true>(const de::CPd* __restrict, de::CPd* __restrict,
    const decx::dsp::fft::FKT1D*, const uint32_t, const uint32_t, const uint32_t, const uint32_t, const decx::dsp::fft::cpu_FFT3D_subproc<double>*);

template _THREAD_FUNCTION_ void decx::dsp::fft::CPUK::_FFT3D_smaller_2rows_cplxd<de::CPd, false>(const de::CPd* __restrict, de::CPd* __restrict,
    const decx::dsp::fft::FKT1D*, const uint32_t, const uint32_t, const uint32_t, const uint32_t, const decx::dsp::fft::cpu_FFT3D_subproc<double>*);

template _THREAD_FUNCTION_ void decx::dsp::fft::CPUK::_FFT3D_smaller_2rows_cplxd<uint8_t, true>(const uint8_t* __restrict, de::CPd* __restrict,
    const decx::dsp::fft::FKT1D*, const uint32_t, const uint32_t, const uint32_t, const uint32_t, const decx::dsp::fft::cpu_FFT3D_subproc<double>*);

template _THREAD_FUNCTION_ void decx::dsp::fft::CPUK::_FFT3D_smaller_2rows_cplxd<uint8_t, false>(const uint8_t* __restrict, de::CPd* __restrict,
    const decx::dsp::fft::FKT1D*, const uint32_t, const uint32_t, const uint32_t, const uint32_t, const decx::dsp::fft::cpu_FFT3D_subproc<double>*);



template <typename _type_out> 
_THREAD_FUNCTION_ void decx::dsp::fft::CPUK::
_IFFT3D_smaller_2rows_cplxd(const de::CPd* __restrict src_head_ptr, _type_out* __restrict dst_head_ptr,
                            const decx::dsp::fft::FKT1D* _tiles,    const uint32_t _pitch_src, 
                            const uint32_t _pitch_dst,              const uint32_t _proc_H_r1,
                            const uint32_t start_dex,               const decx::dsp::fft::cpu_FFT3D_subproc<double>* _FFT_info)
{
    decx::utils::double_buffer_manager _double_buffer(_tiles->get_tile1<void>(), _tiles->get_tile2<void>());
    decx::utils::frag_manager _f_mgr_H;
    decx::utils::frag_manager_gen_from_fragLen(&_f_mgr_H, _proc_H_r1, 2);
    
    uint32_t start_dex_H = start_dex;
    
    for (uint32_t i = 0; i < _f_mgr_H.frag_num; ++i) 
    {
        _double_buffer.reset_buffer1_leading();
        
        // Load and transpose data from global memory
        decx::dsp::fft::CPUK::
            load_entire_row_transpose_cplxd_zip<true>(src_head_ptr,      &_double_buffer,
                                                      _tiles,            _pitch_src >> 1,
                                                      _pitch_src,        &_FFT_info->_FFT_zip_info_LDG,
                                                      start_dex_H,       _FFT_info->_FFT_info.get_signal_len(),
                                                      i == (_f_mgr_H.frag_num - 1) ? _f_mgr_H.last_frag_len : 2);

        // Call vec4 smaller FFT
		decx::dsp::fft::CPUK::_FFT1D_caller_cplxd64_1st_C2C(_double_buffer.get_leading_ptr<de::CPd>(), 
															_double_buffer.get_lagging_ptr<de::CPd>(),
															_FFT_info->_FFT_info.get_kernel_info_ptr(0));
        _double_buffer.update_states();

		for (uint32_t _FFT_index = 1; _FFT_index < _FFT_info->_FFT_info.get_kernel_call_num(); ++_FFT_index) {
			decx::dsp::fft::CPUK::_FFT1D_caller_cplxd64_mid_C2C(_double_buffer.get_leading_ptr<de::CPd>(), 
																_double_buffer.get_lagging_ptr<de::CPd>(),
																_FFT_info->_FFT_info.get_W_table<de::CPd>(),
																_FFT_info->_FFT_info.get_kernel_info_ptr(_FFT_index));

			_double_buffer.update_states();
		}

        // Store back to global memory
        if constexpr (std::is_same_v<_type_out, double>){
            decx::dsp::fft::CPUK::
                store_entire_row_transpose_cplxd_fp64_zip(&_double_buffer,  dst_head_ptr, 
                                                          _tiles,           _pitch_dst >> 1, 
                                                          _pitch_dst,       &_FFT_info->_FFT_zip_info_STG, 
                                                          start_dex_H,      i == (_f_mgr_H.frag_num - 1) ? _f_mgr_H.last_frag_len : 2);
        }
        else if constexpr (std::is_same_v<_type_out, uint8_t>) {
            decx::dsp::fft::CPUK::
                store_entire_row_transpose_cplxd_u8_zip(&_double_buffer,  dst_head_ptr, 
                                                        _tiles,           _pitch_dst >> 1, 
                                                        _pitch_dst,       &_FFT_info->_FFT_zip_info_STG, 
                                                        start_dex_H,      i == (_f_mgr_H.frag_num - 1) ? _f_mgr_H.last_frag_len : 2);
        }
        else {
            decx::dsp::fft::CPUK::
                store_entire_row_transpose_cplxd_zip<false>(&_double_buffer,  dst_head_ptr, 
                                                            _tiles,           _pitch_dst >> 1, 
                                                            _pitch_dst,       &_FFT_info->_FFT_zip_info_STG, 
                                                            start_dex_H,      i == (_f_mgr_H.frag_num - 1) ? _f_mgr_H.last_frag_len : 2);
        }
        start_dex_H += 2;
    }
}

template _THREAD_FUNCTION_ void decx::dsp::fft::CPUK::_IFFT3D_smaller_2rows_cplxd<de::CPd>(const de::CPd* __restrict, de::CPd* __restrict,
    const decx::dsp::fft::FKT1D*, const uint32_t, const uint32_t, const uint32_t, const uint32_t, const decx::dsp::fft::cpu_FFT3D_subproc<double>*);

template _THREAD_FUNCTION_ void decx::dsp::fft::CPUK::_IFFT3D_smaller_2rows_cplxd<double>(const de::CPd* __restrict, double* __restrict,
    const decx::dsp::fft::FKT1D*, const uint32_t, const uint32_t, const uint32_t, const uint32_t, const decx::dsp::fft::cpu_FFT3D_subproc<double>*);

template _THREAD_FUNCTION_ void decx::dsp::fft::CPUK::_IFFT3D_smaller_2rows_cplxd<uint8_t>(const de::CPd* __restrict, uint8_t* __restrict,
    const decx::dsp::fft::FKT1D*, const uint32_t, const uint32_t, const uint32_t, const uint32_t, const decx::dsp::fft::cpu_FFT3D_subproc<double>*);



template <typename _type_in, bool _conj>
void decx::dsp::fft::_FFT3D_H_entire_rows_cplxd(const _type_in* __restrict src_head_ptr, 
                                                 de::CPd* __restrict dst_head_ptr, 
                                                 const decx::dsp::fft::cpu_FFT3D_planner<double>* planner,
                                                 decx::utils::_thread_arrange_1D* t1D, 
                                                 decx::dsp::fft::FFT_directions _proc_dir)
{
    const decx::dsp::fft::cpu_FFT3D_subproc<double>* FFT_info = planner->get_subproc(_proc_dir);
    const decx::utils::frag_manager* f_mgr = &FFT_info->_f_mgr;
    
    for (uint32_t i = 0; i < f_mgr->frag_num - 1; ++i) {
        t1D->_async_thread[i] = decx::cpu::register_task_default(
            decx::dsp::fft::CPUK::_FFT3D_smaller_2rows_cplxd<_type_in, _conj>,
            src_head_ptr,               dst_head_ptr, 
            planner->get_tile_ptr(i),   
            FFT_info->_pitchsrc,        FFT_info->_pitchdst, 
            f_mgr->frag_len,            i * f_mgr->frag_len, FFT_info);
    }

    const uint32_t _L = f_mgr->is_left ? f_mgr->frag_left_over : f_mgr->frag_len;
    t1D->_async_thread[f_mgr->frag_num - 1] = decx::cpu::register_task_default(
        decx::dsp::fft::CPUK::_FFT3D_smaller_2rows_cplxd<_type_in, _conj>,
            src_head_ptr,               dst_head_ptr, 
            planner->get_tile_ptr(f_mgr->frag_num - 1),
            FFT_info->_pitchsrc,        FFT_info->_pitchdst, 
            _L,                         (f_mgr->frag_num - 1) * f_mgr->frag_len, FFT_info);

    t1D->__sync_all_threads(make_uint2(0, f_mgr->frag_num));
}

template void decx::dsp::fft::_FFT3D_H_entire_rows_cplxd<double, true>(const double* __restrict, de::CPd* __restrict, const decx::dsp::fft::cpu_FFT3D_planner<double>*,
    decx::utils::_thread_arrange_1D* t1D, decx::dsp::fft::FFT_directions);

template void decx::dsp::fft::_FFT3D_H_entire_rows_cplxd<double, false>(const double* __restrict, de::CPd* __restrict, const decx::dsp::fft::cpu_FFT3D_planner<double>*,
    decx::utils::_thread_arrange_1D* t1D, decx::dsp::fft::FFT_directions);


template void decx::dsp::fft::_FFT3D_H_entire_rows_cplxd<uint8_t, true>(const uint8_t* __restrict, de::CPd* __restrict, const decx::dsp::fft::cpu_FFT3D_planner<double>*,
    decx::utils::_thread_arrange_1D* t1D, decx::dsp::fft::FFT_directions);

template void decx::dsp::fft::_FFT3D_H_entire_rows_cplxd<uint8_t, false>(const uint8_t* __restrict, de::CPd* __restrict, const decx::dsp::fft::cpu_FFT3D_planner<double>*,
    decx::utils::_thread_arrange_1D* t1D, decx::dsp::fft::FFT_directions);


template void decx::dsp::fft::_FFT3D_H_entire_rows_cplxd<de::CPd, true>(const de::CPd* __restrict, de::CPd* __restrict, const decx::dsp::fft::cpu_FFT3D_planner<double>*,
    decx::utils::_thread_arrange_1D* t1D, decx::dsp::fft::FFT_directions);

template void decx::dsp::fft::_FFT3D_H_entire_rows_cplxd<de::CPd, false>(const de::CPd* __restrict, de::CPd* __restrict, const decx::dsp::fft::cpu_FFT3D_planner<double>*,
    decx::utils::_thread_arrange_1D* t1D, decx::dsp::fft::FFT_directions);




template <typename _type_out>
void decx::dsp::fft::_IFFT3D_H_entire_rows_cplxd(const de::CPd* __restrict src_head_ptr, 
                                                 _type_out* __restrict dst_head_ptr, 
                                                 const decx::dsp::fft::cpu_FFT3D_planner<double>* planner,
                                                 decx::utils::_thread_arrange_1D* t1D, 
                                                 decx::dsp::fft::FFT_directions _proc_dir)
{
    const decx::dsp::fft::cpu_FFT3D_subproc<double>* FFT_info = planner->get_subproc(_proc_dir);
    const decx::utils::frag_manager* f_mgr = &FFT_info->_f_mgr;
    
    for (uint32_t i = 0; i < f_mgr->frag_num - 1; ++i) {
        t1D->_async_thread[i] = decx::cpu::register_task_default(
            decx::dsp::fft::CPUK::_IFFT3D_smaller_2rows_cplxd<_type_out>,
            src_head_ptr,               dst_head_ptr, 
            planner->get_tile_ptr(i),   
            FFT_info->_pitchsrc,        FFT_info->_pitchdst, 
            f_mgr->frag_len,            i * f_mgr->frag_len, FFT_info);
    }
    
    const uint32_t _L = f_mgr->is_left ? f_mgr->frag_left_over : f_mgr->frag_len;
    t1D->_async_thread[f_mgr->frag_num - 1] = decx::cpu::register_task_default(
            decx::dsp::fft::CPUK::_IFFT3D_smaller_2rows_cplxd<_type_out>,
            src_head_ptr,               dst_head_ptr, 
            planner->get_tile_ptr(f_mgr->frag_num - 1),
            FFT_info->_pitchsrc,        FFT_info->_pitchdst, 
            _L,            (f_mgr->frag_num - 1) * f_mgr->frag_len, FFT_info);

    t1D->__sync_all_threads(make_uint2(0, f_mgr->frag_num));
}


template void decx::dsp::fft::_IFFT3D_H_entire_rows_cplxd<de::CPd>(const de::CPd* __restrict, de::CPd* __restrict, const decx::dsp::fft::cpu_FFT3D_planner<double>*,
    decx::utils::_thread_arrange_1D* t1D, decx::dsp::fft::FFT_directions);

template void decx::dsp::fft::_IFFT3D_H_entire_rows_cplxd<double>(const de::CPd* __restrict, double* __restrict, const decx::dsp::fft::cpu_FFT3D_planner<double>*,
    decx::utils::_thread_arrange_1D* t1D, decx::dsp::fft::FFT_directions);

template void decx::dsp::fft::_IFFT3D_H_entire_rows_cplxd<uint8_t>(const de::CPd* __restrict, uint8_t* __restrict, const decx::dsp::fft::cpu_FFT3D_planner<double>*,
    decx::utils::_thread_arrange_1D* t1D, decx::dsp::fft::FFT_directions);
