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

#include "../FFT1D_kernels.h"
#include "../FFT1D_kernel_utils.h"
#include "../../CPU_FFT_defs.h"



_THREAD_CALL_ void
decx::dsp::fft::CPUK::_FFT1D_caller_cplxf32_1st_R2C(const float* __restrict						src, 
                                                    de::CPf* __restrict							dst, 
                                                    const decx::dsp::fft::_FFT1D_kernel_info*	_kernel_info)
{
	switch (_kernel_info->_radix)
	{
	case 2:
		decx::dsp::fft::CPUK::_FFT1D_R2_cplxf32_1st_R2C(src, dst, _kernel_info->_signal_len);
		break;

	case 3:
		decx::dsp::fft::CPUK::_FFT1D_R3_cplxf32_1st_R2C(src, dst, _kernel_info->_signal_len);
		break;

	case 4:
		decx::dsp::fft::CPUK::_FFT1D_R4_cplxf32_1st_R2C(src, dst, _kernel_info->_signal_len);
		break;

	case 5:
		decx::dsp::fft::CPUK::_FFT1D_R5_cplxf32_1st_R2C(src, dst, _kernel_info->_signal_len);
		break;
	default:
		break;
	}
}



_THREAD_CALL_ void
decx::dsp::fft::CPUK::_FFT1D_caller_cplxf32_1st_C2C(const de::CPf* __restrict					src, 
                                                    de::CPf* __restrict							dst, 
                                                    const decx::dsp::fft::_FFT1D_kernel_info*	_kernel_info)
{
	switch (_kernel_info->_radix)
	{
	case 2:
		decx::dsp::fft::CPUK::_FFT1D_R2_cplxf32_1st_C2C(src, dst, _kernel_info->_signal_len);
		break;

	case 3:
		decx::dsp::fft::CPUK::_FFT1D_R3_cplxf32_1st_C2C(src, dst, _kernel_info->_signal_len);
		break;

	case 4:
		decx::dsp::fft::CPUK::_FFT1D_R4_cplxf32_1st_C2C(src, dst, _kernel_info->_signal_len);
		break;

	case 5:
		decx::dsp::fft::CPUK::_FFT1D_R5_cplxf32_1st_C2C(src, dst, _kernel_info->_signal_len);
		break;
	default:
		break;
	}
}



_THREAD_CALL_ void
decx::dsp::fft::CPUK::_FFT1D_caller_cplxf32_mid_C2C(const de::CPf* __restrict					src, 
													de::CPf* __restrict							dst, 
													const de::CPf* __restrict					_W_table,
													const decx::dsp::fft::_FFT1D_kernel_info*	_kernel_info)
{
	switch (_kernel_info->_radix)
	{
	case 2:
		decx::dsp::fft::CPUK::_FFT1D_R2_cplxf32_mid_C2C(src, dst, _W_table, _kernel_info);
		break;

	case 3:
		decx::dsp::fft::CPUK::_FFT1D_R3_cplxf32_mid_C2C(src, dst, _W_table, _kernel_info);
		break;

	case 4:
		decx::dsp::fft::CPUK::_FFT1D_R4_cplxf32_mid_C2C(src, dst, _W_table, _kernel_info);
		break;

	case 5:
		decx::dsp::fft::CPUK::_FFT1D_R5_cplxf32_mid_C2C(src, dst, _W_table, _kernel_info);
		break;
	default:
		break;
	}
}




template <bool _IFFT, typename _type_in> _THREAD_FUNCTION_ void
decx::dsp::fft::CPUK::_FFT1D_smaller_1st_cplxf32(const _type_in* __restrict							src,
												 de::CPf* __restrict								dst, 
												 const decx::dsp::fft::FKT1D*					_tiles,
												 const uint64_t										_signal_length,
												 const decx::dsp::fft::cpu_FFT1D_smaller<float>*	_FFT_info,
												 const uint32_t										FFT_call_times,
												 const uint32_t										FFT_call_time_start,
												 const decx::dsp::fft::FIMT1D*						_Twd_info)
{
	decx::utils::double_buffer_manager _double_buffer(_tiles->get_tile1<void>(), _tiles->get_tile2<void>());

	const uint64_t _load_pitch = _signal_length / _FFT_info->get_signal_len();

	uint32_t _call_time_base = FFT_call_time_start;
	const uint32_t FFT_call_times_v4 = decx::utils::ceil<uint32_t>(FFT_call_times, 4);
	const uint8_t _L_v4 = FFT_call_times % 4;

	decx::utils::frag_manager _store_linearly_config;
	decx::utils::frag_manager_gen(&_store_linearly_config, _FFT_info->get_signal_len(), 4);

	for (uint32_t _call_times = 0; _call_times < FFT_call_times_v4; ++_call_times)
	{
		if constexpr (std::is_same_v<_type_in, float>){
		decx::dsp::fft::CPUK::_load_1st_v4_fp32(src, _tiles->get_tile1<float>(), _FFT_info, _call_times, _signal_length);

		decx::dsp::fft::CPUK::_FFT1D_caller_cplxf32_1st_R2C(_tiles->get_tile1<float>(),
															_tiles->get_tile2<de::CPf>(),
															_FFT_info->get_kernel_info_ptr(0));
		}
		else {
		decx::dsp::fft::CPUK::_load_1st_v4_cplxf<_IFFT>(src, _tiles->get_tile1<de::CPf>(), _FFT_info, _call_times, _signal_length);

		decx::dsp::fft::CPUK::_FFT1D_caller_cplxf32_1st_C2C(_tiles->get_tile1<de::CPf>(),
															_tiles->get_tile2<de::CPf>(),
															_FFT_info->get_kernel_info_ptr(0));
		}
		_double_buffer.reset_buffer2_leading();

		for (uint32_t i = 1; i < _FFT_info->get_kernel_call_num(); ++i) {
			decx::dsp::fft::CPUK::_FFT1D_caller_cplxf32_mid_C2C(_double_buffer.get_leading_ptr<de::CPf>(), 
																_double_buffer.get_lagging_ptr<de::CPf>(),
																_FFT_info->get_W_table<de::CPf>(), 
																_FFT_info->get_kernel_info_ptr(i));

			_double_buffer.update_states();
		}

		if (_Twd_info != NULL) 
		{
			decx::dsp::fft::CPUK::_FFT1D_Twd_smaller_kernels_v4_1st(_double_buffer.get_leading_ptr<de::CPf>(),
																	_double_buffer.get_lagging_ptr<de::CPf>(),
																	_FFT_info->get_signal_len(),
																	_call_time_base,
																	_Twd_info);

			_double_buffer.update_states();
		}

		if (_call_times < FFT_call_times_v4 - 1 || _L_v4 == 0) {
			decx::dsp::fft::CPUK::_1st_FFT1D_frag_transpose_v4_cplxf(&_double_buffer, _tiles, dst, _call_times, FFT_call_times_v4,
				4, _FFT_info->get_signal_len());
		}
		else {
			decx::dsp::fft::CPUK::_1st_FFT1D_frag_transpose_v4_cplxf(&_double_buffer, _tiles, dst, _call_times, FFT_call_times_v4,
				_L_v4, _FFT_info->get_signal_len());
		}

		_call_time_base += 4;
	}
}

template _THREAD_FUNCTION_ void
decx::dsp::fft::CPUK::_FFT1D_smaller_1st_cplxf32<true, float>(const float* __restrict, de::CPf* __restrict, const decx::dsp::fft::FKT1D*,
	const uint64_t, const decx::dsp::fft::cpu_FFT1D_smaller<float>*, const uint32_t, const uint32_t, const decx::dsp::fft::FIMT1D*);

template _THREAD_FUNCTION_ void
decx::dsp::fft::CPUK::_FFT1D_smaller_1st_cplxf32<true, de::CPf>(const de::CPf* __restrict, de::CPf* __restrict, const decx::dsp::fft::FKT1D*,
	const uint64_t, const decx::dsp::fft::cpu_FFT1D_smaller<float>*, const uint32_t, const uint32_t, const decx::dsp::fft::FIMT1D*);

template _THREAD_FUNCTION_ void
decx::dsp::fft::CPUK::_FFT1D_smaller_1st_cplxf32<false, float>(const float* __restrict, de::CPf* __restrict, const decx::dsp::fft::FKT1D*,
	const uint64_t, const decx::dsp::fft::cpu_FFT1D_smaller<float>*, const uint32_t, const uint32_t, const decx::dsp::fft::FIMT1D*);

template _THREAD_FUNCTION_ void
decx::dsp::fft::CPUK::_FFT1D_smaller_1st_cplxf32<false, de::CPf>(const de::CPf* __restrict, de::CPf* __restrict, const decx::dsp::fft::FKT1D*,
	const uint64_t, const decx::dsp::fft::cpu_FFT1D_smaller<float>*, const uint32_t, const uint32_t, const decx::dsp::fft::FIMT1D*);




template <typename _type_out, bool _conj> _THREAD_FUNCTION_ void
decx::dsp::fft::CPUK::_FFT1D_smaller_mid_cplxf32_C2C(const de::CPf* __restrict							src, 
												     _type_out* __restrict								dst, 
												     void* __restrict									_tmp1_ptr, 
												     void* __restrict									_tmp2_ptr, 
												     const decx::dsp::fft::FKI1D*						_global_kernel_info,
												     const decx::dsp::fft::cpu_FFT1D_smaller<float>*	_FFT_info,
													 const uint32_t										_FFT_times_v4,
													 const uint32_t										FFT_call_time_start_v1,
													 const decx::dsp::fft::FIMT1D*						_Twd_info)
{
	decx::utils::double_buffer_manager _double_buffer(_tmp1_ptr, _tmp2_ptr);
	const uint64_t _load_pitch = _global_kernel_info->_signal_len / _FFT_info->get_signal_len();

	const uint32_t FFT_call_times_v4 = decx::utils::ceil<uint32_t>(_FFT_times_v4, 4);
	const uint8_t _L_v4 = _FFT_times_v4 % 4;

	const de::CPf* _src_start_ptr = src;
	_type_out* _dst_start_ptr = dst;

	for (uint32_t _warp_id = 0; _warp_id < _global_kernel_info->get_warp_num(); ++_warp_id)
	{
		uint32_t FFT_warp_loc_id_base = FFT_call_time_start_v1;
		for (uint32_t _call_times_in_warp = 0; _call_times_in_warp < FFT_call_times_v4; ++_call_times_in_warp)
		{
			decx::dsp::fft::CPUK::_load_1st_v4_cplxf<false>(_src_start_ptr, (de::CPf*)_tmp1_ptr, _FFT_info, _call_times_in_warp, _global_kernel_info->_signal_len);

			decx::dsp::fft::CPUK::_FFT1D_caller_cplxf32_1st_C2C((de::CPf*)_tmp1_ptr, 
																(de::CPf*)_tmp2_ptr, 
																_FFT_info->get_kernel_info_ptr(0));

			_double_buffer.reset_buffer2_leading();

			for (uint32_t i = 1; i < _FFT_info->get_kernel_call_num(); ++i) {
				decx::dsp::fft::CPUK::_FFT1D_caller_cplxf32_mid_C2C(_double_buffer.get_leading_ptr<de::CPf>(), 
																	_double_buffer.get_lagging_ptr<de::CPf>(), 
																	_FFT_info->get_W_table<de::CPf>(), 
																	_FFT_info->get_kernel_info_ptr(i));

				_double_buffer.update_states();
			}

			if (_Twd_info != NULL) {
				decx::dsp::fft::CPUK::_FFT1D_Twd_smaller_kernels_v4_mid(_double_buffer.get_leading_ptr<de::CPf>(),
																		_double_buffer.get_lagging_ptr<de::CPf>(),
																		_FFT_info->get_signal_len(),
																		_call_times_in_warp,
																		_global_kernel_info->_warp_proc_len * _warp_id + FFT_call_time_start_v1,
																		_global_kernel_info->_store_pitch,
																		_Twd_info);

				_double_buffer.update_states();
			}
			if constexpr (std::is_same_v<_type_out, de::CPf>){
				decx::dsp::fft::CPUK::_store_fragment_to_DRAM_cplxf<_conj>(_double_buffer.get_leading_ptr<de::CPf>(), _dst_start_ptr, 
																		   _call_times_in_warp,						 FFT_call_times_v4, 
																		   _L_v4,									 _global_kernel_info, 
																		   _FFT_info->get_signal_len());
			}
			else {
				decx::dsp::fft::CPUK::_store_fragment_to_DRAM_cplxf_fp32(_double_buffer.get_leading_ptr<de::CPf>(), _dst_start_ptr, 
																	   _call_times_in_warp,						 FFT_call_times_v4, 
																	   _L_v4,									 _global_kernel_info, 
																	   _FFT_info->get_signal_len());
			}
			FFT_warp_loc_id_base += 4;
		}
		_src_start_ptr += _global_kernel_info->_store_pitch;
		_dst_start_ptr += _global_kernel_info->_warp_proc_len;
	}
}

template _THREAD_FUNCTION_ void
decx::dsp::fft::CPUK::_FFT1D_smaller_mid_cplxf32_C2C<de::CPf, true>(const de::CPf* __restrict, de::CPf* __restrict, void* __restrict, void* __restrict,
	const decx::dsp::fft::FKI1D*, const decx::dsp::fft::cpu_FFT1D_smaller<float>*, const uint32_t, const uint32_t, const decx::dsp::fft::FIMT1D*);

template _THREAD_FUNCTION_ void
decx::dsp::fft::CPUK::_FFT1D_smaller_mid_cplxf32_C2C<de::CPf, false>(const de::CPf* __restrict, de::CPf* __restrict, void* __restrict, void* __restrict,
	const decx::dsp::fft::FKI1D*, const decx::dsp::fft::cpu_FFT1D_smaller<float>*, const uint32_t, const uint32_t, const decx::dsp::fft::FIMT1D*);

template _THREAD_FUNCTION_ void
decx::dsp::fft::CPUK::_FFT1D_smaller_mid_cplxf32_C2C<float, true>(const de::CPf* __restrict, float* __restrict, void* __restrict, void* __restrict,
	const decx::dsp::fft::FKI1D*, const decx::dsp::fft::cpu_FFT1D_smaller<float>*, const uint32_t, const uint32_t, const decx::dsp::fft::FIMT1D*);

template _THREAD_FUNCTION_ void
decx::dsp::fft::CPUK::_FFT1D_smaller_mid_cplxf32_C2C<float, false>(const de::CPf* __restrict, float* __restrict, void* __restrict, void* __restrict,
	const decx::dsp::fft::FKI1D*, const decx::dsp::fft::cpu_FFT1D_smaller<float>*, const uint32_t, const uint32_t, const decx::dsp::fft::FIMT1D*);



// ---------------------------------------------- THREAD_CALLERS ----------------------------------------------
template <bool _IFFT, typename _type_in>
void decx::dsp::fft::_FFT1D_cplxf32_1st(const _type_in* __restrict						src, 
										de::CPf* __restrict								dst,
										const decx::dsp::fft::cpu_FFT1D_planner<float>* _FFT_frame, 
										decx::utils::_thr_1D*							t1D,
										const decx::dsp::fft::FIMT1D*					_Twd_info)
{
	const _type_in* _src_ptr = src;
	de::CPf* _dst_ptr = dst;

	const decx::dsp::fft::cpu_FFT1D_smaller<float>* _inner_FFT_info = _FFT_frame->get_smaller_FFT_info_ptr(0);
	const decx::utils::frag_manager* _f_mgr = _inner_FFT_info->get_thread_patching();

	for (uint32_t i = 0; i < t1D->total_thread - 1; ++i) {
		t1D->_async_thread[i] = decx::cpu::register_task_default(decx::dsp::fft::CPUK::_FFT1D_smaller_1st_cplxf32<_IFFT, _type_in>,
			_src_ptr,											_dst_ptr,
			_FFT_frame->get_tile_ptr(i),
			_FFT_frame->get_signal_len(),						_inner_FFT_info,
			_f_mgr->frag_len,									_f_mgr->frag_len * i,
			_Twd_info);

		_src_ptr += _f_mgr->frag_len;
		_dst_ptr += _f_mgr->frag_len * _inner_FFT_info->get_signal_len();
	}
	uint32_t _L_FFT_smaller_num = _f_mgr->is_left ? _f_mgr->frag_left_over : _f_mgr->frag_len;
	t1D->_async_thread[t1D->total_thread - 1] = decx::cpu::register_task_default(decx::dsp::fft::CPUK::_FFT1D_smaller_1st_cplxf32<_IFFT, _type_in>,
		_src_ptr,																_dst_ptr,
		_FFT_frame->get_tile_ptr(t1D->total_thread - 1),
		_FFT_frame->get_signal_len(),											_inner_FFT_info,
		_L_FFT_smaller_num,														_f_mgr->frag_len * (t1D->total_thread - 1),
		_Twd_info);

	t1D->__sync_all_threads();
}

template void decx::dsp::fft::_FFT1D_cplxf32_1st<true, float>(const float* __restrict, de::CPf* __restrict,
	const decx::dsp::fft::cpu_FFT1D_planner<float>*, decx::utils::_thr_1D*, const decx::dsp::fft::FIMT1D*);

template void decx::dsp::fft::_FFT1D_cplxf32_1st<true, de::CPf>(const de::CPf* __restrict, de::CPf* __restrict,
	const decx::dsp::fft::cpu_FFT1D_planner<float>*, decx::utils::_thr_1D*, const decx::dsp::fft::FIMT1D*);

template void decx::dsp::fft::_FFT1D_cplxf32_1st<false, float>(const float* __restrict, de::CPf* __restrict,
	const decx::dsp::fft::cpu_FFT1D_planner<float>*, decx::utils::_thr_1D*, const decx::dsp::fft::FIMT1D*);

template void decx::dsp::fft::_FFT1D_cplxf32_1st<false, de::CPf>(const de::CPf* __restrict, de::CPf* __restrict,
	const decx::dsp::fft::cpu_FFT1D_planner<float>*, decx::utils::_thr_1D*, const decx::dsp::fft::FIMT1D*);



template <typename _type_out, bool _conj>
void decx::dsp::fft::_FFT1D_cplxf32_mid(const de::CPf* __restrict							src, 
										_type_out* __restrict								dst,
										const decx::dsp::fft::cpu_FFT1D_planner<float>*		_FFT_frame, 
										decx::utils::_thr_1D*								t1D, 
										const uint32_t										_call_order,
										const decx::dsp::fft::FIMT1D*						_Twd_info)
{
	const de::CPf* _src_ptr = src;
	_type_out* _dst_ptr = dst;
	
	const decx::dsp::fft::cpu_FFT1D_smaller<float>* _inner_FFT_info = _FFT_frame->get_smaller_FFT_info_ptr(_call_order);
	const decx::dsp::fft::FKI1D* _outer_kernel_info = _FFT_frame->get_outer_kernel_info(_call_order);
	const decx::utils::frag_manager* _f_mgr = _inner_FFT_info->get_thread_patching();
	
	for (uint32_t i = 0; i < t1D->total_thread - 1; ++i) {
		t1D->_async_thread[i] = decx::cpu::register_task_default(decx::dsp::fft::CPUK::_FFT1D_smaller_mid_cplxf32_C2C<_type_out, _conj>,
			_src_ptr,											_dst_ptr,
			_FFT_frame->get_tile_ptr(i)->get_tile1<void>(),		_FFT_frame->get_tile_ptr(i)->get_tile2<void>(),
			_outer_kernel_info,									_inner_FFT_info,
			_f_mgr->frag_len,									_f_mgr->frag_len * i,
			_Twd_info);

		_src_ptr += _f_mgr->frag_len;
		_dst_ptr += _f_mgr->frag_len;
	}
	const uint32_t _L_FFT_smaller_num = _f_mgr->is_left ? _f_mgr->frag_left_over : _f_mgr->frag_len;
	t1D->_async_thread[t1D->total_thread - 1] = decx::cpu::register_task_default(decx::dsp::fft::CPUK::_FFT1D_smaller_mid_cplxf32_C2C<_type_out, _conj>,
		_src_ptr,																_dst_ptr,
		_FFT_frame->get_tile_ptr(t1D->total_thread - 1)->get_tile1<void>(),		_FFT_frame->get_tile_ptr(t1D->total_thread - 1)->get_tile2<void>(),
		_outer_kernel_info,														_inner_FFT_info,
		_L_FFT_smaller_num,														_f_mgr->frag_len * (t1D->total_thread - 1),
		_Twd_info);

	t1D->__sync_all_threads();
}


template void decx::dsp::fft::_FFT1D_cplxf32_mid<de::CPf, true>(const de::CPf* __restrict, de::CPf* __restrict,
	const decx::dsp::fft::cpu_FFT1D_planner<float>*, decx::utils::_thr_1D*, const uint32_t, const decx::dsp::fft::FIMT1D*);

template void decx::dsp::fft::_FFT1D_cplxf32_mid<de::CPf, false>(const de::CPf* __restrict, de::CPf* __restrict,
	const decx::dsp::fft::cpu_FFT1D_planner<float>*, decx::utils::_thr_1D*, const uint32_t, const decx::dsp::fft::FIMT1D*);

template void decx::dsp::fft::_FFT1D_cplxf32_mid<float, true>(const de::CPf* __restrict, float* __restrict,
	const decx::dsp::fft::cpu_FFT1D_planner<float>*, decx::utils::_thr_1D*, const uint32_t, const decx::dsp::fft::FIMT1D*);

template void decx::dsp::fft::_FFT1D_cplxf32_mid<float, false>(const de::CPf* __restrict, float* __restrict,
	const decx::dsp::fft::cpu_FFT1D_planner<float>*, decx::utils::_thr_1D*, const uint32_t, const decx::dsp::fft::FIMT1D*);
