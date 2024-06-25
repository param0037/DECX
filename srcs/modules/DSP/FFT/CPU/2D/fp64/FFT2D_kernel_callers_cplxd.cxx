/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#include "../../1D/FFT1D_kernels.h"
#include "../FFT2D_kernel_utils.h"
#include "../FFT2D_kernels.h"


template <typename _type_in, bool _conj> _THREAD_FUNCTION_ void
decx::dsp::fft::CPUK::_FFT2D_smaller_2rows_cplxd(const _type_in* __restrict                      src,
                                                 de::CPd* __restrict                              dst,
                                                 const decx::dsp::fft::FKT1D*               _tiles,
                                                 const uint32_t                                  _pitch_src, 
                                                 const uint32_t                                  _pitch_dst, 
                                                 const uint32_t                                  _proc_H_r1,
                                                 const decx::dsp::fft::cpu_FFT1D_smaller<double>* _FFT_info)
{
    decx::utils::double_buffer_manager _double_buffer(_tiles->get_tile1<void>(), _tiles->get_tile2<void>());
    decx::utils::frag_manager _f_mgr_H;
    decx::utils::frag_manager_gen_from_fragLen(&_f_mgr_H, _proc_H_r1, 2);
    
    const _type_in* _src_loc_ptr = src;
    de::CPd* _dst_loc_ptr = dst;

    for (uint32_t i = 0; i < _f_mgr_H.frag_num; ++i) 
    {
        _double_buffer.reset_buffer1_leading();
        if constexpr (std::is_same_v<_type_in, double>){
            // Load and transpose data from global memory
            decx::dsp::fft::CPUK::
                load_entire_row_transpose_fp64(_src_loc_ptr, 
                                               &_double_buffer, 
                                               _tiles, 
                                               _pitch_src >> 1, 
                                               _pitch_src,
                                               i == (_f_mgr_H.frag_num - 1) ? _f_mgr_H.last_frag_len : 2);

            // Call vec2 smaller FFT
		    decx::dsp::fft::CPUK::_FFT1D_caller_cplxd64_1st_R2C(_double_buffer.get_leading_ptr<double>(), 
															    _double_buffer.get_lagging_ptr<de::CPd>(),
															    _FFT_info->get_kernel_info_ptr(0));
        }
        else if constexpr (std::is_same_v<_type_in, uint8_t>) {
            // Load and transpose data from global memory
            decx::dsp::fft::CPUK::load_entire_row_transpose_u8_fp64((int16_t*)_src_loc_ptr, &_double_buffer, _tiles, _pitch_src >> 1, _pitch_src,
                i == (_f_mgr_H.frag_num - 1) ? _f_mgr_H.last_frag_len : 2);

            // Call vec4 smaller FFT
		    decx::dsp::fft::CPUK::_FFT1D_caller_cplxd64_1st_R2C(_double_buffer.get_leading_ptr<double>(), 
															    _double_buffer.get_lagging_ptr<de::CPd>(),
															    _FFT_info->get_kernel_info_ptr(0));
        }
        else {
            // Load and transpose data from global memory
            decx::dsp::fft::CPUK::
                load_entire_row_transpose_cplxd<false>(_src_loc_ptr, 
                                                       &_double_buffer, 
                                                       _tiles, 
                                                       _pitch_src >> 1, 
                                                       _pitch_src,
                                                       i == (_f_mgr_H.frag_num - 1) ? _f_mgr_H.last_frag_len : 2);

            // Call vec4 smaller FFT
		    decx::dsp::fft::CPUK::_FFT1D_caller_cplxd64_1st_C2C(_double_buffer.get_leading_ptr<de::CPd>(), 
															    _double_buffer.get_lagging_ptr<de::CPd>(),
															    _FFT_info->get_kernel_info_ptr(0));
        }
        _double_buffer.update_states();

		for (uint32_t _FFT_index = 1; _FFT_index < _FFT_info->get_kernel_call_num(); ++_FFT_index) {
			decx::dsp::fft::CPUK::_FFT1D_caller_cplxd64_mid_C2C(_double_buffer.get_leading_ptr<de::CPd>(), 
																_double_buffer.get_lagging_ptr<de::CPd>(),
																_FFT_info->get_W_table<de::CPd>(), 
																_FFT_info->get_kernel_info_ptr(_FFT_index));

			_double_buffer.update_states();
		}

        // Store back to global memory
        decx::dsp::fft::CPUK::
            store_entire_row_transpose_cplxd<_conj>(&_double_buffer, 
                                                    _dst_loc_ptr, 
                                                    _tiles, 
                                                    _pitch_dst >> 1, 
                                                    _pitch_dst,
                                                    i == (_f_mgr_H.frag_num - 1) ? _f_mgr_H.last_frag_len : 2);

        _src_loc_ptr += (_pitch_src << 1);
        _dst_loc_ptr += (_pitch_dst << 1);
    }
}



template <typename _type_in, bool _conj> void
decx::dsp::fft::_FFT2D_H_entire_rows_cplxd(const _type_in* __restrict                       src,
                                           de::CPd* __restrict                               dst, 
                                           const decx::dsp::fft::cpu_FFT2D_planner<double>*  planner,
                                           const uint32_t                                   pitch_src,
                                           const uint32_t                                   pitch_dst,
                                           decx::utils::_thread_arrange_1D*                 t1D,
                                           bool _is_FFTH)
{
    const decx::utils::frag_manager* f_mgr = _is_FFTH ? planner->get_thread_dist_H() : planner->get_thread_dist_V();
    const decx::dsp::fft::cpu_FFT1D_smaller<double>* _FFT1D = _is_FFTH ? planner->get_FFTH_info() : planner->get_FFTV_info();

    const _type_in* _src_loc_ptr = src;
    de::CPd* _dst_loc_ptr = dst;

    for (uint32_t i = 0; i < f_mgr->frag_num - 1; ++i) {
        t1D->_async_thread[i] = decx::cpu::register_task_default(decx::dsp::fft::CPUK::_FFT2D_smaller_2rows_cplxd<_type_in, _conj>,
            _src_loc_ptr,               _dst_loc_ptr, 
            planner->get_tile_ptr(i),
            pitch_src,                  pitch_dst, 
            f_mgr->frag_len,            _FFT1D);

        _src_loc_ptr += f_mgr->frag_len * pitch_src;
        _dst_loc_ptr += f_mgr->frag_len * pitch_dst;
    }

    t1D->_async_thread[f_mgr->frag_num - 1] = decx::cpu::register_task_default(decx::dsp::fft::CPUK::_FFT2D_smaller_2rows_cplxd<_type_in, _conj>,
            _src_loc_ptr,               _dst_loc_ptr, 
            planner->get_tile_ptr(f_mgr->frag_num - 1),
            pitch_src,                  pitch_dst, 
            f_mgr->last_frag_len,       _FFT1D);

    t1D->__sync_all_threads(make_uint2(0, f_mgr->frag_num));
}

template void decx::dsp::fft::_FFT2D_H_entire_rows_cplxd<double, true>(const double* __restrict, de::CPd* __restrict, const decx::dsp::fft::cpu_FFT2D_planner<double>*,
    const uint32_t, const uint32_t, decx::utils::_thread_arrange_1D*, const bool);

template void decx::dsp::fft::_FFT2D_H_entire_rows_cplxd<de::CPd, true>(const de::CPd* __restrict, de::CPd* __restrict, const decx::dsp::fft::cpu_FFT2D_planner<double>*,
    const uint32_t, const uint32_t, decx::utils::_thread_arrange_1D*, const bool);

template void decx::dsp::fft::_FFT2D_H_entire_rows_cplxd<double, false>(const double* __restrict, de::CPd* __restrict, const decx::dsp::fft::cpu_FFT2D_planner<double>*,
    const uint32_t, const uint32_t, decx::utils::_thread_arrange_1D*, const bool);

template void decx::dsp::fft::_FFT2D_H_entire_rows_cplxd<de::CPd, false>(const de::CPd* __restrict, de::CPd* __restrict, const decx::dsp::fft::cpu_FFT2D_planner<double>*,
    const uint32_t, const uint32_t, decx::utils::_thread_arrange_1D*, const bool);

template void decx::dsp::fft::_FFT2D_H_entire_rows_cplxd<uint8_t, true>(const uint8_t* __restrict, de::CPd* __restrict, const decx::dsp::fft::cpu_FFT2D_planner<double>*,
    const uint32_t, const uint32_t, decx::utils::_thread_arrange_1D*, const bool);

template void decx::dsp::fft::_FFT2D_H_entire_rows_cplxd<uint8_t, false>(const uint8_t* __restrict, de::CPd* __restrict, const decx::dsp::fft::cpu_FFT2D_planner<double>*,
    const uint32_t, const uint32_t, decx::utils::_thread_arrange_1D*, const bool);




template <typename _type_out> _THREAD_FUNCTION_ void
decx::dsp::fft::CPUK::_IFFT2D_smaller_2rows_cplxd(const de::CPd* __restrict                       src,
                                                  _type_out* __restrict                           dst,
                                                  const decx::dsp::fft::FKT1D*                    _tiles,
                                                  const uint32_t                                  _pitch_src, 
                                                  const uint32_t                                  _pitch_dst, 
                                                  const uint32_t                                  _proc_H_r1,
                                                  const decx::dsp::fft::cpu_FFT1D_smaller<double>* _FFT_info)
{
    decx::utils::double_buffer_manager _double_buffer(_tiles->get_tile1<void>(), _tiles->get_tile2<void>());
    decx::utils::frag_manager _f_mgr_H;
    decx::utils::frag_manager_gen_from_fragLen(&_f_mgr_H, _proc_H_r1, 2);
    const uint32_t _L = _f_mgr_H.is_left ? _f_mgr_H.frag_left_over : 2;
    
    const de::CPd* _src_loc_ptr = src;
    _type_out* _dst_loc_ptr = dst;

    for (uint32_t i = 0; i < _f_mgr_H.frag_num; ++i) 
    {
        _double_buffer.reset_buffer1_leading();
        
        // Load and transpose data from global memory
        decx::dsp::fft::CPUK::load_entire_row_transpose_cplxd<true>(_src_loc_ptr, &_double_buffer, _tiles, _pitch_src >> 1, _pitch_src,
            i == (_f_mgr_H.frag_num - 1) ? _L : 2, _FFT_info->get_signal_len());

        // Call vec4 smaller FFT
		decx::dsp::fft::CPUK::_FFT1D_caller_cplxd64_1st_C2C(_double_buffer.get_leading_ptr<de::CPd>(), 
															_double_buffer.get_lagging_ptr<de::CPd>(),
															_FFT_info->get_kernel_info_ptr(0));
        _double_buffer.update_states();

		for (uint32_t _FFT_index = 1; _FFT_index < _FFT_info->get_kernel_call_num(); ++_FFT_index) {
			decx::dsp::fft::CPUK::_FFT1D_caller_cplxd64_mid_C2C(_double_buffer.get_leading_ptr<de::CPd>(), 
																_double_buffer.get_lagging_ptr<de::CPd>(),
																_FFT_info->get_W_table<de::CPd>(), 
																_FFT_info->get_kernel_info_ptr(_FFT_index));

			_double_buffer.update_states();
		}

        // Store back to global memory
        if constexpr (std::is_same_v<_type_out, de::CPd>) {
            decx::dsp::fft::CPUK::store_entire_row_transpose_cplxd<false>(&_double_buffer,  _dst_loc_ptr, 
                                                                          _tiles,           _pitch_dst >> 1, 
                                                                          _pitch_dst,       i == (_f_mgr_H.frag_num - 1) ? _L : 2);
        }
        else if constexpr (std::is_same_v<_type_out, uint8_t>) {
            decx::dsp::fft::CPUK::
                store_entire_row_transpose_cplxd_u8(&_double_buffer,  (int16_t*)_dst_loc_ptr, 
                                                    _tiles,           decx::utils::ceil<uint32_t>(_FFT_info->get_signal_len(), 2), 
                                                    _pitch_dst,       i == (_f_mgr_H.frag_num - 1) ? _L : 2);
        }
        else {
            decx::dsp::fft::CPUK::store_entire_row_transpose_cplxd_fp64(&_double_buffer, _dst_loc_ptr, _tiles, _pitch_dst >> 1, _pitch_dst,
                i == (_f_mgr_H.frag_num - 1) ? _L : 2);
        }
        _src_loc_ptr += (_pitch_src << 1);
        _dst_loc_ptr += (_pitch_dst << 1);
    }
}


template _THREAD_FUNCTION_ void decx::dsp::fft::CPUK::_IFFT2D_smaller_2rows_cplxd<de::CPd>(const de::CPd* __restrict, de::CPd* __restrict, const decx::dsp::fft::FKT1D*,
    const uint32_t, const uint32_t, const uint32_t, const decx::dsp::fft::cpu_FFT1D_smaller<double>*);

template _THREAD_FUNCTION_ void decx::dsp::fft::CPUK::_IFFT2D_smaller_2rows_cplxd<double>(const de::CPd* __restrict, double* __restrict, const decx::dsp::fft::FKT1D*,
    const uint32_t, const uint32_t, const uint32_t, const decx::dsp::fft::cpu_FFT1D_smaller<double>*);

template _THREAD_FUNCTION_ void decx::dsp::fft::CPUK::_IFFT2D_smaller_2rows_cplxd<uint8_t>(const de::CPd* __restrict, uint8_t* __restrict, const decx::dsp::fft::FKT1D*,
    const uint32_t, const uint32_t, const uint32_t, const decx::dsp::fft::cpu_FFT1D_smaller<double>*);



template <typename _type_out> void
decx::dsp::fft::_IFFT2D_H_entire_rows_cplxd(const de::CPd* __restrict                         src,
                                            _type_out* __restrict                            dst, 
                                            const decx::dsp::fft::cpu_FFT2D_planner<double>*  planner,
                                            const uint32_t                                   pitch_src,
                                            const uint32_t                                   pitch_dst,
                                            decx::utils::_thread_arrange_1D*                 t1D,
                                            bool _is_FFTH)
{
    const decx::utils::frag_manager* f_mgr = _is_FFTH ? planner->get_thread_dist_H() : planner->get_thread_dist_V();
    const decx::dsp::fft::cpu_FFT1D_smaller<double>* _FFT1D = _is_FFTH ? planner->get_FFTH_info() : planner->get_FFTV_info();

    const de::CPd* _src_loc_ptr = src;
    _type_out* _dst_loc_ptr = dst;

    for (uint32_t i = 0; i < f_mgr->frag_num - 1; ++i) {
        t1D->_async_thread[i] = decx::cpu::register_task_default(decx::dsp::fft::CPUK::_IFFT2D_smaller_2rows_cplxd<_type_out>,
            _src_loc_ptr,               _dst_loc_ptr, 
            planner->get_tile_ptr(i),
            pitch_src,                  pitch_dst, 
            f_mgr->frag_len,            _FFT1D);

        _src_loc_ptr += f_mgr->frag_len * pitch_src;
        _dst_loc_ptr += f_mgr->frag_len * pitch_dst;
    }
    const uint32_t _L = f_mgr->is_left ? f_mgr->frag_left_over : f_mgr->frag_len;
    t1D->_async_thread[f_mgr->frag_num - 1] = decx::cpu::register_task_default(decx::dsp::fft::CPUK::_IFFT2D_smaller_2rows_cplxd<_type_out>,
            _src_loc_ptr,               _dst_loc_ptr, 
            planner->get_tile_ptr(f_mgr->frag_num - 1),
            pitch_src,                  pitch_dst, 
            _L,                         _FFT1D);

    t1D->__sync_all_threads(make_uint2(0, f_mgr->frag_num));
}

template void decx::dsp::fft::_IFFT2D_H_entire_rows_cplxd<double>(const de::CPd* __restrict, double* __restrict, const decx::dsp::fft::cpu_FFT2D_planner<double>*,
    const uint32_t, const uint32_t, decx::utils::_thread_arrange_1D*, bool);

template void decx::dsp::fft::_IFFT2D_H_entire_rows_cplxd<de::CPd>(const de::CPd* __restrict, de::CPd* __restrict, const decx::dsp::fft::cpu_FFT2D_planner<double>*,
    const uint32_t, const uint32_t, decx::utils::_thread_arrange_1D*, bool);

template void decx::dsp::fft::_IFFT2D_H_entire_rows_cplxd<uint8_t>(const de::CPd* __restrict, uint8_t* __restrict, const decx::dsp::fft::cpu_FFT2D_planner<double>*,
    const uint32_t, const uint32_t, decx::utils::_thread_arrange_1D*, bool);
