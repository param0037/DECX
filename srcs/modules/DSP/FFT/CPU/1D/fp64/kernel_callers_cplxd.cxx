/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#include "../FFT1D_kernels.h"
#include "../FFT1D_kernel_utils.h"
#include "../../CPU_FFT_defs.h"



_THREAD_CALL_ void
decx::dsp::fft::CPUK::_FFT1D_caller_cplxd64_1st_R2C(const double* __restrict						src, 
                                                    de::CPd* __restrict							dst, 
                                                    const decx::dsp::fft::_FFT1D_kernel_info*	_kernel_info)
{
	switch (_kernel_info->_radix)
	{
	case 2:
		decx::dsp::fft::CPUK::_FFT1D_R2_cplxd64_1st_R2C(src, dst, _kernel_info->_signal_len);
		break;

	case 3:
		decx::dsp::fft::CPUK::_FFT1D_R3_cplxd64_1st_R2C(src, dst, _kernel_info->_signal_len);
		break;

	case 4:
		decx::dsp::fft::CPUK::_FFT1D_R4_cplxd64_1st_R2C(src, dst, _kernel_info->_signal_len);
		break;

	case 5:
		decx::dsp::fft::CPUK::_FFT1D_R5_cplxd64_1st_R2C(src, dst, _kernel_info->_signal_len);
		break;
	default:
		break;
	}
}



_THREAD_CALL_ void
decx::dsp::fft::CPUK::_FFT1D_caller_cplxd64_1st_C2C(const de::CPd* __restrict					src, 
                                                    de::CPd* __restrict							dst, 
                                                    const decx::dsp::fft::_FFT1D_kernel_info*	_kernel_info)
{
	switch (_kernel_info->_radix)
	{
	case 2:
		decx::dsp::fft::CPUK::_FFT1D_R2_cplxd64_1st_C2C(src, dst, _kernel_info->_signal_len);
		break;

	case 3:
		decx::dsp::fft::CPUK::_FFT1D_R3_cplxd64_1st_C2C(src, dst, _kernel_info->_signal_len);
		break;

	case 4:
		decx::dsp::fft::CPUK::_FFT1D_R4_cplxd64_1st_C2C(src, dst, _kernel_info->_signal_len);
		break;

	case 5:
		decx::dsp::fft::CPUK::_FFT1D_R5_cplxd64_1st_C2C(src, dst, _kernel_info->_signal_len);
		break;
	default:
		break;
	}
}



_THREAD_CALL_ void
decx::dsp::fft::CPUK::_FFT1D_caller_cplxd64_mid_C2C(const de::CPd* __restrict					src, 
													de::CPd* __restrict							dst, 
													const de::CPd* __restrict					_W_table,
													const decx::dsp::fft::_FFT1D_kernel_info*	_kernel_info)
{
	switch (_kernel_info->_radix)
	{
	case 2:
		decx::dsp::fft::CPUK::_FFT1D_R2_cplxd64_mid_C2C(src, dst, _W_table, _kernel_info);
		break;

	case 3:
		decx::dsp::fft::CPUK::_FFT1D_R3_cplxd64_mid_C2C(src, dst, _W_table, _kernel_info);
		break;

	case 4:
		decx::dsp::fft::CPUK::_FFT1D_R4_cplxd64_mid_C2C(src, dst, _W_table, _kernel_info);
		break;

	case 5:
		decx::dsp::fft::CPUK::_FFT1D_R5_cplxd64_mid_C2C(src, dst, _W_table, _kernel_info);
		break;
	default:
		break;
	}
}




template <bool _IFFT, typename _type_in> _THREAD_FUNCTION_ void
decx::dsp::fft::CPUK::_FFT1D_smaller_1st_cplxd64(const _type_in* __restrict							src,
												 de::CPd* __restrict								dst, 
												 const decx::dsp::fft::FKT1D_fp32*					_tiles,
												 const uint64_t										_signal_length,
												 const decx::dsp::fft::cpu_FFT1D_smaller<double>*	_FFT_info,
												 const uint32_t										FFT_call_times,
												 const uint32_t										FFT_call_time_start,
												 const decx::dsp::fft::FIMT1D*						_Twd_info)
{
	decx::utils::double_buffer_manager _double_buffer(_tiles->get_tile1<void>(), _tiles->get_tile2<void>());

	const uint64_t _load_pitch = _signal_length / _FFT_info->get_signal_len();

	uint32_t _call_time_base = FFT_call_time_start;
	const uint32_t FFT_call_times_v2 = decx::utils::ceil<uint32_t>(FFT_call_times, 2);
	const uint8_t _L_v2 = FFT_call_times % 2;

	decx::utils::frag_manager _store_linearly_config;
	decx::utils::frag_manager_gen(&_store_linearly_config, _FFT_info->get_signal_len(), 2);

	for (uint32_t _call_times = 0; _call_times < FFT_call_times_v2; ++_call_times)
	{
		if constexpr (std::is_same_v<_type_in, double>){		// R2C
		decx::dsp::fft::CPUK::_load_1st_v2_fp64(src, _tiles->get_tile1<double>(), _FFT_info, _call_times, _signal_length);
		
		decx::dsp::fft::CPUK::_FFT1D_caller_cplxd64_1st_R2C(_tiles->get_tile1<double>(),
															_tiles->get_tile2<de::CPd>(),
															_FFT_info->get_kernel_info_ptr(0));
		}
		else {		// C2C
		decx::dsp::fft::CPUK::_load_1st_v2_cplxd<_IFFT>(src, _tiles->get_tile1<de::CPd>(), _FFT_info, _call_times, _signal_length);

		decx::dsp::fft::CPUK::_FFT1D_caller_cplxd64_1st_C2C(_tiles->get_tile1<de::CPd>(),
															_tiles->get_tile2<de::CPd>(),
															_FFT_info->get_kernel_info_ptr(0));
		}
		_double_buffer.reset_buffer2_leading();

		for (uint32_t i = 1; i < _FFT_info->get_kernel_call_num(); ++i) {
			decx::dsp::fft::CPUK::_FFT1D_caller_cplxd64_mid_C2C(_double_buffer.get_leading_ptr<de::CPd>(), 
																_double_buffer.get_lagging_ptr<de::CPd>(),
																_FFT_info->get_W_table<de::CPd>(), 
																_FFT_info->get_kernel_info_ptr(i));

			_double_buffer.update_states();
		}

		if (_Twd_info != NULL) 
		{
			decx::dsp::fft::CPUK::_FFT1D_Twd_smaller_kernels_v2_1st(_double_buffer.get_leading_ptr<double>(),
																	_double_buffer.get_lagging_ptr<double>(),
																	_FFT_info->get_signal_len(),
																	_call_time_base,
																	_Twd_info);

			_double_buffer.update_states();
		}

		if (_call_times < FFT_call_times_v2 - 1 || _L_v2 == 0) {
			decx::dsp::fft::CPUK::_1st_FFT1D_frag_transpose_v2_cplxd(&_double_buffer, _tiles, dst, _call_times, FFT_call_times_v2,
				2, _FFT_info->get_signal_len());
		}
		else {
			decx::dsp::fft::CPUK::_1st_FFT1D_frag_transpose_v2_cplxd(&_double_buffer, _tiles, dst, _call_times, FFT_call_times_v2,
				_L_v2, _FFT_info->get_signal_len());
		}

		_call_time_base += 2;
	}
}

template _THREAD_FUNCTION_ void
decx::dsp::fft::CPUK::_FFT1D_smaller_1st_cplxd64<true, double>(const double* __restrict, de::CPd* __restrict, const decx::dsp::fft::FKT1D_fp32*,
	const uint64_t, const decx::dsp::fft::cpu_FFT1D_smaller<double>*, const uint32_t, const uint32_t, const decx::dsp::fft::FIMT1D*);

template _THREAD_FUNCTION_ void
decx::dsp::fft::CPUK::_FFT1D_smaller_1st_cplxd64<true, de::CPd>(const de::CPd* __restrict, de::CPd* __restrict, const decx::dsp::fft::FKT1D_fp32*,
	const uint64_t, const decx::dsp::fft::cpu_FFT1D_smaller<double>*, const uint32_t, const uint32_t, const decx::dsp::fft::FIMT1D*);

template _THREAD_FUNCTION_ void
decx::dsp::fft::CPUK::_FFT1D_smaller_1st_cplxd64<false, double>(const double* __restrict, de::CPd* __restrict, const decx::dsp::fft::FKT1D_fp32*,
	const uint64_t, const decx::dsp::fft::cpu_FFT1D_smaller<double>*, const uint32_t, const uint32_t, const decx::dsp::fft::FIMT1D*);

template _THREAD_FUNCTION_ void
decx::dsp::fft::CPUK::_FFT1D_smaller_1st_cplxd64<false, de::CPd>(const de::CPd* __restrict, de::CPd* __restrict, const decx::dsp::fft::FKT1D_fp32*,
	const uint64_t, const decx::dsp::fft::cpu_FFT1D_smaller<double>*, const uint32_t, const uint32_t, const decx::dsp::fft::FIMT1D*);



template <typename _type_out, bool _conj> _THREAD_FUNCTION_ void
decx::dsp::fft::CPUK::_FFT1D_smaller_mid_cplxd64_C2C(const de::CPd* __restrict							src, 
												     _type_out* __restrict								dst, 
												     void* __restrict									_tmp1_ptr, 
												     void* __restrict									_tmp2_ptr, 
												     const decx::dsp::fft::FKI1D*						_global_kernel_info,
												     const decx::dsp::fft::cpu_FFT1D_smaller<double>*	_FFT_info,
													 const uint32_t										_FFT_times_v2,
													 const uint32_t										FFT_call_time_start_v1,
													 const decx::dsp::fft::FIMT1D*						_Twd_info)
{
	decx::utils::double_buffer_manager _double_buffer(_tmp1_ptr, _tmp2_ptr);
	const uint64_t _load_pitch = _global_kernel_info->_signal_len / _FFT_info->get_signal_len();

	const uint32_t FFT_call_times_v2 = decx::utils::ceil<uint32_t>(_FFT_times_v2, 2);
	const uint8_t _L_v2 = _FFT_times_v2 % 2;

	const de::CPd* _src_start_ptr = src;
	_type_out* _dst_start_ptr = dst;

	const uint32_t _high_level_warp_proc_len = _global_kernel_info->_warp_proc_len / _FFT_info->get_signal_len();

	for (uint32_t _warp_id = 0; _warp_id < _global_kernel_info->get_warp_num(); ++_warp_id)
	{
		uint32_t FFT_warp_loc_id_base = FFT_call_time_start_v1;
		for (uint32_t _call_times_in_warp = 0; _call_times_in_warp < FFT_call_times_v2; ++_call_times_in_warp)
		{
			decx::dsp::fft::CPUK::_load_1st_v2_cplxd<false>(_src_start_ptr, (de::CPd*)_tmp1_ptr, _FFT_info, _call_times_in_warp, _global_kernel_info->_signal_len);

			decx::dsp::fft::CPUK::_FFT1D_caller_cplxd64_1st_C2C((de::CPd*)_tmp1_ptr, 
																(de::CPd*)_tmp2_ptr, 
																_FFT_info->get_kernel_info_ptr(0));

			_double_buffer.reset_buffer2_leading();

			for (uint32_t i = 1; i < _FFT_info->get_kernel_call_num(); ++i) {
				decx::dsp::fft::CPUK::_FFT1D_caller_cplxd64_mid_C2C(_double_buffer.get_leading_ptr<de::CPd>(), 
																	_double_buffer.get_lagging_ptr<de::CPd>(), 
																	_FFT_info->get_W_table<de::CPd>(), 
																	_FFT_info->get_kernel_info_ptr(i));

				_double_buffer.update_states();
			}

			if (_Twd_info != NULL) {
				decx::dsp::fft::CPUK::_FFT1D_Twd_smaller_kernels_v2_mid(_double_buffer.get_leading_ptr<de::CPd>(),
																		_double_buffer.get_lagging_ptr<de::CPd>(),
																		_FFT_info->get_signal_len(),
																		_call_times_in_warp,
																		_global_kernel_info->_warp_proc_len * _warp_id + FFT_call_time_start_v1,
																		_global_kernel_info->_store_pitch,
																		_Twd_info);

				_double_buffer.update_states();
			}
			if constexpr (std::is_same_v<_type_out, de::CPd>){
				decx::dsp::fft::CPUK::_store_fragment_to_DRAM_cplxd<_conj>(_double_buffer.get_leading_ptr<de::CPd>(), _dst_start_ptr, 
																		   _call_times_in_warp,						 FFT_call_times_v2, 
																		   _L_v2,									 _global_kernel_info, 
																		   _FFT_info->get_signal_len());
			}
			else {
				decx::dsp::fft::CPUK::_store_fragment_to_DRAM_cplxd_fp64(_double_buffer.get_leading_ptr<de::CPd>(), _dst_start_ptr, 
																	   _call_times_in_warp,							FFT_call_times_v2, 
																	   _L_v2,										_global_kernel_info, 
																	   _FFT_info->get_signal_len());
			}
			FFT_warp_loc_id_base += 2;
		}
		_src_start_ptr += _global_kernel_info->_store_pitch;
		_dst_start_ptr += _global_kernel_info->_warp_proc_len;
	}
}

template _THREAD_FUNCTION_ void
decx::dsp::fft::CPUK::_FFT1D_smaller_mid_cplxd64_C2C<de::CPd, true>(const de::CPd* __restrict, de::CPd* __restrict, void* __restrict, void* __restrict,
	const decx::dsp::fft::FKI1D*, const decx::dsp::fft::cpu_FFT1D_smaller<double>*, const uint32_t, const uint32_t, const decx::dsp::fft::FIMT1D*);

template _THREAD_FUNCTION_ void
decx::dsp::fft::CPUK::_FFT1D_smaller_mid_cplxd64_C2C<de::CPd, false>(const de::CPd* __restrict, de::CPd* __restrict, void* __restrict, void* __restrict,
	const decx::dsp::fft::FKI1D*, const decx::dsp::fft::cpu_FFT1D_smaller<double>*, const uint32_t, const uint32_t, const decx::dsp::fft::FIMT1D*);

template _THREAD_FUNCTION_ void
decx::dsp::fft::CPUK::_FFT1D_smaller_mid_cplxd64_C2C<double, true>(const de::CPd* __restrict, double* __restrict, void* __restrict, void* __restrict,
	const decx::dsp::fft::FKI1D*, const decx::dsp::fft::cpu_FFT1D_smaller<double>*, const uint32_t, const uint32_t, const decx::dsp::fft::FIMT1D*);

template _THREAD_FUNCTION_ void
decx::dsp::fft::CPUK::_FFT1D_smaller_mid_cplxd64_C2C<double, false>(const de::CPd* __restrict, double* __restrict, void* __restrict, void* __restrict,
	const decx::dsp::fft::FKI1D*, const decx::dsp::fft::cpu_FFT1D_smaller<double>*, const uint32_t, const uint32_t, const decx::dsp::fft::FIMT1D*);




template <bool _IFFT, typename _type_in>
void decx::dsp::fft::_FFT1D_cplxd64_1st(const _type_in* __restrict						src, 
										de::CPd* __restrict								dst,
										const decx::dsp::fft::cpu_FFT1D_planner<double>* _FFT_frame, 
										decx::utils::_thr_1D*							t1D,
										const decx::dsp::fft::FIMT1D*					_Twd_info)
{
	const _type_in* _src_ptr = src;
	de::CPd* _dst_ptr = dst;

	const decx::dsp::fft::cpu_FFT1D_smaller<double>* _inner_FFT_info = _FFT_frame->get_smaller_FFT_info_ptr(0);
	const decx::utils::frag_manager* _f_mgr = _inner_FFT_info->get_thread_patching();

	for (uint32_t i = 0; i < t1D->total_thread - 1; ++i) {
		t1D->_async_thread[i] = decx::cpu::register_task_default(decx::dsp::fft::CPUK::_FFT1D_smaller_1st_cplxd64<_IFFT, _type_in>,
			_src_ptr,											_dst_ptr,
			_FFT_frame->get_tile_ptr(i),
			_FFT_frame->get_signal_len(),						_inner_FFT_info,
			_f_mgr->frag_len,									_f_mgr->frag_len * i,
			_Twd_info);

		_src_ptr += _f_mgr->frag_len;
		_dst_ptr += _f_mgr->frag_len * _inner_FFT_info->get_signal_len();
	}
	uint32_t _L_FFT_smaller_num = _f_mgr->is_left ? _f_mgr->frag_left_over : _f_mgr->frag_len;
	t1D->_async_thread[t1D->total_thread - 1] = decx::cpu::register_task_default(decx::dsp::fft::CPUK::_FFT1D_smaller_1st_cplxd64<_IFFT, _type_in>,
		_src_ptr,																_dst_ptr,
		_FFT_frame->get_tile_ptr(t1D->total_thread - 1),
		_FFT_frame->get_signal_len(),											_inner_FFT_info,
		_L_FFT_smaller_num,														_f_mgr->frag_len * (t1D->total_thread - 1),
		_Twd_info);

	t1D->__sync_all_threads();
}

template void decx::dsp::fft::_FFT1D_cplxd64_1st<true, double>(const double* __restrict, de::CPd* __restrict,
	const decx::dsp::fft::cpu_FFT1D_planner<double>*, decx::utils::_thr_1D*, const decx::dsp::fft::FIMT1D*);

template void decx::dsp::fft::_FFT1D_cplxd64_1st<true, de::CPd>(const de::CPd* __restrict, de::CPd* __restrict,
	const decx::dsp::fft::cpu_FFT1D_planner<double>*, decx::utils::_thr_1D*, const decx::dsp::fft::FIMT1D*);

template void decx::dsp::fft::_FFT1D_cplxd64_1st<false, double>(const double* __restrict, de::CPd* __restrict,
	const decx::dsp::fft::cpu_FFT1D_planner<double>*, decx::utils::_thr_1D*, const decx::dsp::fft::FIMT1D*);

template void decx::dsp::fft::_FFT1D_cplxd64_1st<false, de::CPd>(const de::CPd* __restrict, de::CPd* __restrict,
	const decx::dsp::fft::cpu_FFT1D_planner<double>*, decx::utils::_thr_1D*, const decx::dsp::fft::FIMT1D*);



template <typename _type_out, bool _conj>
void decx::dsp::fft::_FFT1D_cplxd64_mid(const de::CPd* __restrict							src, 
										_type_out* __restrict								dst,
										const decx::dsp::fft::cpu_FFT1D_planner<double>*	_FFT_frame, 
										decx::utils::_thr_1D*								t1D, 
										const uint32_t										_call_order,
										const decx::dsp::fft::FIMT1D*						_Twd_info)
{
	const de::CPd* _src_ptr = src;
	_type_out* _dst_ptr = dst;
	
	const decx::dsp::fft::cpu_FFT1D_smaller<double>* _inner_FFT_info = _FFT_frame->get_smaller_FFT_info_ptr(_call_order);
	const decx::dsp::fft::FKI1D* _outer_kernel_info = _FFT_frame->get_outer_kernel_info(_call_order);
	const decx::utils::frag_manager* _f_mgr = _inner_FFT_info->get_thread_patching();

	for (uint32_t i = 0; i < t1D->total_thread - 1; ++i) {
		t1D->_async_thread[i] = decx::cpu::register_task_default(decx::dsp::fft::CPUK::_FFT1D_smaller_mid_cplxd64_C2C<_type_out, _conj>,
			_src_ptr,											_dst_ptr,
			_FFT_frame->get_tile_ptr(i)->get_tile1<void>(),		_FFT_frame->get_tile_ptr(i)->get_tile2<void>(),
			_outer_kernel_info,									_inner_FFT_info,
			_f_mgr->frag_len,									_f_mgr->frag_len * i,
			_Twd_info);

		_src_ptr += _f_mgr->frag_len;
		_dst_ptr += _f_mgr->frag_len;
	}
	const uint32_t _L_FFT_smaller_num = _f_mgr->is_left ? _f_mgr->frag_left_over : _f_mgr->frag_len;
	t1D->_async_thread[t1D->total_thread - 1] = decx::cpu::register_task_default(decx::dsp::fft::CPUK::_FFT1D_smaller_mid_cplxd64_C2C<_type_out, _conj>,
		_src_ptr,																_dst_ptr,
		_FFT_frame->get_tile_ptr(t1D->total_thread - 1)->get_tile1<void>(),		_FFT_frame->get_tile_ptr(t1D->total_thread - 1)->get_tile2<void>(),
		_outer_kernel_info,														_inner_FFT_info,
		_L_FFT_smaller_num,														_f_mgr->frag_len * (t1D->total_thread - 1),
		_Twd_info);

	t1D->__sync_all_threads();
}


template void decx::dsp::fft::_FFT1D_cplxd64_mid<de::CPd, true>(const de::CPd* __restrict, de::CPd* __restrict,
	const decx::dsp::fft::cpu_FFT1D_planner<double>*, decx::utils::_thr_1D*, const uint32_t, const decx::dsp::fft::FIMT1D*);

template void decx::dsp::fft::_FFT1D_cplxd64_mid<de::CPd, false>(const de::CPd* __restrict, de::CPd* __restrict,
	const decx::dsp::fft::cpu_FFT1D_planner<double>*, decx::utils::_thr_1D*, const uint32_t, const decx::dsp::fft::FIMT1D*);

template void decx::dsp::fft::_FFT1D_cplxd64_mid<double, true>(const de::CPd* __restrict, double* __restrict,
	const decx::dsp::fft::cpu_FFT1D_planner<double>*, decx::utils::_thr_1D*, const uint32_t, const decx::dsp::fft::FIMT1D*);

template void decx::dsp::fft::_FFT1D_cplxd64_mid<double, false>(const de::CPd* __restrict, double* __restrict,
	const decx::dsp::fft::cpu_FFT1D_planner<double>*, decx::utils::_thr_1D*, const uint32_t, const decx::dsp::fft::FIMT1D*);
