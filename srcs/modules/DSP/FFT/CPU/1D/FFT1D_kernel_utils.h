/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _FFT1D_KERNLE_UTILS_H_
#define _FFT1D_KERNLE_UTILS_H_


#include "../../../../core/basic.h"
#include "../../../../core/thread_management/thread_pool.h"
#include "../../../../core/thread_management/thread_arrange.h"
#include "../CPU_FFT_tiles.h"
#include "../../../../core/utils/fragment_arrangment.h"
#include "CPU_FFT1D_planner.h"
#include "../../../../BLAS/basic_process/transpose/CPU/transpose_exec.h"


namespace decx
{
namespace dsp {
namespace fft {
namespace CPUK 
{
	_THREAD_CALL_ static void 
	_1st_FFT1D_frag_transpose_v4_cplxf(decx::utils::double_buffer_manager*	_double_buffer,
									   const decx::dsp::fft::FKT1D*	_tiles,
									   de::CPf* __restrict					dst,
									   const uint32_t						_call_times,
									   const uint32_t						FFT_call_times_v4,
									   const uint8_t						_L_v4,
									   const uint32_t						_sub_FFT_length)
	{
		double* _local_dst_ptr = (double*)dst + _call_times * (_sub_FFT_length << 2);
		
		const double* _read_ptr = _double_buffer->get_leading_ptr<double>();
		double* _write_ptr = _double_buffer->get_lagging_ptr<double>();

		_tiles->_inblock_transpose_vecAdj_2_VecDist_cplxf(_double_buffer);

		_read_ptr = _double_buffer->get_leading_ptr<double>();

		const uint32_t _int_area_len = (_sub_FFT_length >> 2) << 2;
		for (uint8_t j = 0; j < _L_v4; ++j)
		{
			for (uint32_t i = 0; i < _sub_FFT_length / 4; ++i) {
				_mm256_storeu_pd(_local_dst_ptr + (i << 2), _mm256_load_pd(_read_ptr + (i << 2)));
			}
			for (uint8_t i = 0; i < (_sub_FFT_length % 4); ++i) {
				_local_dst_ptr[i + _int_area_len] = _read_ptr[i + _int_area_len];
			}
			_local_dst_ptr += _sub_FFT_length;
			_read_ptr += _tiles->_tile_row_pitch;
		}
	}


	_THREAD_CALL_ static void 
	_1st_FFT1D_frag_transpose_v2_cplxd(decx::utils::double_buffer_manager*	_double_buffer,
									   const decx::dsp::fft::FKT1D*	_tiles,
									   de::CPd* __restrict					dst,
									   const uint32_t						_call_times,
									   const uint32_t						FFT_call_times_v2,
									   const uint8_t						_L_v2,
									   const uint32_t						_sub_FFT_length)
	{
		de::CPd* _local_dst_ptr = dst + _call_times * (_sub_FFT_length << 1);
		
		de::CPd* src_test = _double_buffer->get_leading_ptr<de::CPd>();

		_tiles->_inblock_transpose_vecAdj_2_VecDist_cplxd(_double_buffer);

		const de::CPd* _read_ptr = _double_buffer->get_leading_ptr<de::CPd>();

		const uint32_t _int_area_len = (_sub_FFT_length >> 1) << 1;
		for (uint8_t j = 0; j < _L_v2; ++j)
		{
			for (uint32_t i = 0; i < _sub_FFT_length / 2; ++i) {
				_mm256_store_pd((double*)(_local_dst_ptr + (i << 1)), _mm256_load_pd((double*)(_read_ptr + (i << 1))));
			}
			if (_sub_FFT_length % 2) {
				_mm_storeu_pd((double*)(_local_dst_ptr + _int_area_len), _mm_loadu_pd((double*)(_read_ptr + _int_area_len)));
			}

			_local_dst_ptr += _sub_FFT_length;
			_read_ptr += _tiles->_tile_row_pitch;
		}
	}


	template <bool _conj> _THREAD_CALL_ static void 
	_store_fragment_to_DRAM_cplxf(const de::CPf* __restrict			_frag_ptr, 
								  de::CPf* __restrict				dst,
								  const uint32_t					_call_times_in_warp, 
								  const uint32_t					_FFT_call_times_v4,
								  const uint8_t						_L_v4,
								  const decx::dsp::fft::FKI1D*		_global_kernel_info,
								  const uint32_t					_small_signal_len)
	{
		decx::utils::simd::xmm256_reg _reg;

		if (_call_times_in_warp < _FFT_call_times_v4 - 1 || _L_v4 == 0) {
			for (uint32_t i = 0; i < _small_signal_len; ++i) {
				_reg._vd = _mm256_load_pd((double*)(_frag_ptr + (i << 2)));
				if constexpr (_conj) { _reg._vf = decx::dsp::CPUK::_cp4_conjugate_fp32(_reg._vf); }
				_mm256_store_pd((double*)(dst + i * _global_kernel_info->_store_pitch + (_call_times_in_warp << 2)), _reg._vd);
			}
		}
		else {
			for (uint32_t i = 0; i < _small_signal_len; ++i)
			{
				_reg._vd = _mm256_load_pd((double*)(_frag_ptr + (i << 2)));
				if constexpr (_conj) { _reg._vf = decx::dsp::CPUK::_cp4_conjugate_fp32(_reg._vf); }
				for (uint8_t j = 0; j < _L_v4; ++j) {
					((double*)(dst))[i * _global_kernel_info->_store_pitch + (_call_times_in_warp << 2) + j] = _reg._arrd[j];
				}
			}
		}
	}


	template <bool _conj> _THREAD_CALL_ static void 
	_store_fragment_to_DRAM_cplxd(const de::CPd* __restrict			_frag_ptr, 
								  de::CPd* __restrict				dst,
								  const uint32_t					_call_times_in_warp, 
								  const uint32_t					_FFT_call_times_v4,
								  const bool						_L_v2,
								  const decx::dsp::fft::FKI1D*		_global_kernel_info,
								  const uint32_t					_small_signal_len)
	{
		decx::utils::simd::xmm256_reg _reg;

		if (_call_times_in_warp < _FFT_call_times_v4 - 1 || !_L_v2) {
			for (uint32_t i = 0; i < _small_signal_len; ++i) {
				_reg._vd = _mm256_load_pd((double*)(_frag_ptr + (i << 1)));
				if constexpr (_conj) { _reg._vd = decx::dsp::CPUK::_cp2_conjugate_fp64(_reg._vd); }
				_mm256_store_pd((double*)(dst + i * _global_kernel_info->_store_pitch + (_call_times_in_warp << 1)), _reg._vd);
			}
		}
		else {
			for (uint32_t i = 0; i < _small_signal_len; ++i)
			{
				_reg._vd = _mm256_load_pd((double*)(_frag_ptr + (i << 1)));
				if constexpr (_conj) { _reg._vd = decx::dsp::CPUK::_cp2_conjugate_fp64(_reg._vd); }
				_mm_storeu_pd((double*)(dst + i * _global_kernel_info->_store_pitch + (_call_times_in_warp << 1)), _reg._vd2[0]);
			}
		}
	}


	_THREAD_CALL_ static void 
	_store_fragment_to_DRAM_cplxf_fp32(const de::CPf* __restrict			_frag_ptr, 
									   float* __restrict				dst,
									   const uint32_t					_call_times_in_warp, 
									   const uint32_t					_FFT_call_times_v4,
									   const uint8_t					_L_v4,
									   const decx::dsp::fft::FKI1D*		_global_kernel_info,
									   const uint32_t					_small_signal_len)
	{
		decx::utils::simd::xmm256_reg _reg;

		if (_call_times_in_warp < _FFT_call_times_v4 - 1 || _L_v4 == 0) {
			for (uint32_t i = 0; i < _small_signal_len; ++i) {
				_reg._vd = _mm256_load_pd((double*)(_frag_ptr + (i << 2)));
				_reg._vf = _mm256_permutevar8x32_ps(_reg._vf, _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7));
				_mm_store_ps(dst + i * _global_kernel_info->_store_pitch + (_call_times_in_warp << 2), _mm256_castps256_ps128(_reg._vf));
			}
		}
		else {
			for (uint32_t i = 0; i < _small_signal_len; ++i)
			{
				_reg._vd = _mm256_load_pd((double*)(_frag_ptr + (i << 2)));
				for (uint8_t j = 0; j < _L_v4; ++j) {
					dst[i * _global_kernel_info->_store_pitch + (_call_times_in_warp << 2) + j] = _reg._arrf[j * 2];
				}
			}
		}
	}


	
	_THREAD_CALL_ static void 
	_store_fragment_to_DRAM_cplxd_fp64(const de::CPd* __restrict		_frag_ptr, 
									   double* __restrict				dst,
									   const uint32_t					_call_times_in_warp, 
									   const uint32_t					_FFT_call_times_v2,
									   const bool						_L_v2,
									   const decx::dsp::fft::FKI1D*		_global_kernel_info,
									   const uint32_t					_small_signal_len)
	{
		decx::utils::simd::xmm256_reg _reg;

		if (_call_times_in_warp < _FFT_call_times_v2 - 1 || !_L_v2) {
			for (uint32_t i = 0; i < _small_signal_len; ++i) {
				_reg._vd = _mm256_load_pd((double*)(_frag_ptr + (i << 1)));
				_reg._vd = _mm256_permute4x64_pd(_reg._vd, 0b11011000);
				_mm_store_pd((double*)(dst + i * _global_kernel_info->_store_pitch + (_call_times_in_warp << 1)), 
					_mm256_castpd256_pd128(_reg._vd));
			}
		}
		else {
			for (uint32_t i = 0; i < _small_signal_len; ++i)
			{
				_reg._vd = _mm256_load_pd((double*)(_frag_ptr + (i << 1)));
				dst[i * _global_kernel_info->_store_pitch + (_call_times_in_warp << 1)] = _reg._arrd[0];
			}
		}
	}


	_THREAD_CALL_ static void 
	_store_fragment_to_DRAM_fp32(const float* __restrict		_frag_ptr, 
								 float* __restrict				dst,
								 const uint32_t					_call_times_in_warp, 
								 const uint32_t					_FFT_call_times_v4,
								 const uint8_t					_L_v4,
								 const decx::dsp::fft::FKI1D*	_global_kernel_info,
								 const uint32_t					_small_signal_len)
	{
		decx::utils::simd::xmm128_reg _reg;

		if (_call_times_in_warp < _FFT_call_times_v4 - 1 || _L_v4 == 0) {

			for (uint32_t i = 0; i < _small_signal_len; ++i)
			{
				_reg._vf = _mm_load_ps(_frag_ptr + (i << 2));
				_mm_store_ps(dst + i * _global_kernel_info->_store_pitch + (_call_times_in_warp << 2), _reg._vf);
			}
		}
		else {
			for (uint32_t i = 0; i < _small_signal_len; ++i)
			{
				_reg._vf = _mm_load_ps(_frag_ptr + (i << 2));
				
				for (uint8_t j = 0; j < _L_v4; ++j) {
					dst[i * _global_kernel_info->_store_pitch + (_call_times_in_warp << 2) + j] = _reg._arrf[j];
				}
			}
		}
	}


	_THREAD_CALL_ static void
	_load_1st_v4_fp32(const float* __restrict src, 
					  float* __restrict dst, 
				      const decx::dsp::fft::cpu_FFT1D_smaller<float>* _FFT_info,
				      const uint32_t _call_times,
				      const uint64_t _signal_length)
	{
		const uint64_t _load_pitch = _signal_length / _FFT_info->get_signal_len();

		for (uint32_t i = 0; i < _FFT_info->get_signal_len(); ++i) {
			_mm_store_ps(dst + (i << 2), _mm_loadu_ps(src + i * _load_pitch + (_call_times << 2)));
		}
	}


	_THREAD_CALL_ static void
	_load_1st_v2_fp64(const double* __restrict src, 
					  double* __restrict dst, 
				      const decx::dsp::fft::cpu_FFT1D_smaller<double>* _FFT_info,
				      const uint32_t _call_times,
				      const uint64_t _signal_length)
	{
		const uint64_t _load_pitch = _signal_length / _FFT_info->get_signal_len();

		for (uint32_t i = 0; i < _FFT_info->get_signal_len(); ++i) {
			_mm_store_pd(dst + (i << 1), _mm_loadu_pd(src + i * _load_pitch + (_call_times << 1)));
		}
	}


	template <bool _IFFT> _THREAD_CALL_ static void
	_load_1st_v4_cplxf(const de::CPf* __restrict						src, 
					   de::CPf* __restrict								dst, 
				       const decx::dsp::fft::cpu_FFT1D_smaller<float>*	_FFT_info,
				       const uint32_t									_call_times,
				       const uint64_t									_signal_length)
	{
		const uint64_t _load_pitch = _signal_length / _FFT_info->get_signal_len();
		decx::utils::simd::xmm256_reg _reg;
		for (uint32_t i = 0; i < _FFT_info->get_signal_len(); ++i) {
			_reg._vd = _mm256_loadu_pd((double*)(src + i * _load_pitch + (_call_times << 2)));
			if constexpr (_IFFT) { _reg._vf = _mm256_div_ps(_reg._vf, _mm256_set1_ps(_signal_length)); }
			_mm256_store_pd((double*)(dst + (i << 2)), _reg._vd);
		}
	}


	template <bool _IFFT> _THREAD_CALL_ static void
	_load_1st_v2_cplxd(const de::CPd* __restrict							src, 
					   de::CPd* __restrict								dst, 
				       const decx::dsp::fft::cpu_FFT1D_smaller<double>*	_FFT_info,
				       const uint32_t									_call_times,
				       const uint64_t									_signal_length)
	{
		const uint64_t _load_pitch = _signal_length / _FFT_info->get_signal_len();
		decx::utils::simd::xmm256_reg _reg;
		for (uint32_t i = 0; i < _FFT_info->get_signal_len(); ++i) {
			_reg._vd = _mm256_loadu_pd((double*)(src + i * _load_pitch + (_call_times << 1)));
			if constexpr (_IFFT) { _reg._vd = _mm256_div_pd(_reg._vd, _mm256_set1_pd(_signal_length)); }
			_mm256_store_pd((double*)(dst + (i << 1)), _reg._vd);
		}
	}
}
}
}
}


//void _mul_Twd_C2C(const de::CPd* __restrict src,
//	de::CPd* __restrict dst,
//	const uint32_t _signal_len,
//	const uint32_t _prev_FFT_radix_fact_sum,
//	const uint32_t _next_FFT_len);

namespace decx
{
namespace dsp
{
namespace fft
{
    namespace CPUK 
    {
        _THREAD_FUNCTION_ void _FFT1D_Twd_smaller_kernels_v4_1st(const de::CPf* __restrict src, de::CPf* __restrict dst,
            const uint32_t _smaller_signal_len, const uint64_t _outer_dex, const decx::dsp::fft::FIMT1D* _Twd);


		_THREAD_FUNCTION_ void _FFT1D_Twd_smaller_kernels_v2_1st(const double* __restrict src, double* __restrict dst,
			const uint32_t _smaller_signal_len, const uint64_t _outer_dex, const decx::dsp::fft::FIMT1D* _Twd);


        _THREAD_FUNCTION_ void _FFT1D_Twd_smaller_kernels_v4_mid(const de::CPf* __restrict src, de::CPf* __restrict dst,
            const uint32_t _smaller_signal_len, const uint32_t _call_times_in_warp, const uint64_t _dst_shf,
			const uint32_t _store_pitch, const decx::dsp::fft::FIMT1D* _Twd);


		_THREAD_FUNCTION_ void _FFT1D_Twd_smaller_kernels_v2_mid(const de::CPd* __restrict src, de::CPd* __restrict dst,
			const uint32_t _smaller_signal_len, const uint32_t _call_times_in_warp, const uint64_t _dst_shf, 
			const uint32_t _store_pitch, const decx::dsp::fft::FIMT1D* _Twd);
    }
}
}
}


#endif