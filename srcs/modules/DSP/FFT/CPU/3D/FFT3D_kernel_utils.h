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


#ifndef _FFT3D_KERNEL_UTILS_H_
#define _FFT3D_KERNEL_UTILS_H_


#include "../2D/FFT2D_kernel_utils.h"


namespace decx
{
namespace dsp {
namespace fft {
    namespace CPUK {
        /**
        * @brief Loads data of entire rows from global memory to the buffer (dst1) where smaller FFT performs. Then transpose
        *        to organize vec4 adjacent form (to dst2). The status of double buffer will be updated
        * @param src : The pointer where data block on global starts
        * @param dst1 : The pointer where untransposed data is temporarily stored.
        * @param dst2 : The pointer where the transposed data is stored.
        */
        static void load_entire_row_transpose_fp32_zip(const float* __restrict src_head_ptr, decx::utils::double_buffer_manager* __restrict _double_buffer,
            const decx::dsp::fft::FKT1D* _tiles, const uint32_t _load_len_v4, const uint32_t _pitch_src, const decx::utils::unpitched_frac_mapping<uint32_t>* zip_info,
            const uint32_t _start_dex, const uint8_t _load_H = 4);


        static void load_entire_row_transpose_fp64_zip(const double* __restrict src_head_ptr, decx::utils::double_buffer_manager* __restrict _double_buffer,
            const decx::dsp::fft::FKT1D* _tiles, const uint32_t _load_len_v2, const uint32_t _pitch_src, const decx::utils::unpitched_frac_mapping<uint32_t>* zip_info,
            const uint32_t _start_dex, const uint8_t _load_H = 2);


        static void load_entire_row_transpose_u8_fp32_zip(const uint8_t* __restrict src_head_ptr, decx::utils::double_buffer_manager* __restrict _double_buffer,
            const decx::dsp::fft::FKT1D* _tiles, const uint32_t _load_len_v4, const uint32_t _pitch_src, const decx::utils::unpitched_frac_mapping<uint32_t>* zip_info,
            const uint32_t _start_dex, const uint8_t _load_H = 4);

        static void load_entire_row_transpose_u8_fp64_zip(const uint8_t* __restrict src_head_ptr, decx::utils::double_buffer_manager* __restrict _double_buffer,
            const decx::dsp::fft::FKT1D* _tiles, const uint32_t _load_len_v2, const uint32_t _pitch_src, const decx::utils::unpitched_frac_mapping<uint32_t>* zip_info,
            const uint32_t _start_dex, const uint8_t _load_H = 2);


        template <bool is_IFFT>
        static void load_entire_row_transpose_cplxf_zip(const de::CPf* __restrict src_head_ptr, decx::utils::double_buffer_manager* __restrict _double_buffer,
            const decx::dsp::fft::FKT1D* _tiles, const uint32_t _load_len_v4, const uint32_t _pitch_src, const decx::utils::unpitched_frac_mapping<uint32_t>* zip_info,
            const uint32_t _start_dex, const uint32_t signal_len, const uint8_t _load_H = 4);


        template <bool is_IFFT>
        static void load_entire_row_transpose_cplxd_zip(const de::CPd* __restrict src_head_ptr, decx::utils::double_buffer_manager* __restrict _double_buffer,
            const decx::dsp::fft::FKT1D* _tiles, const uint32_t _load_len_v4, const uint32_t _pitch_src, const decx::utils::unpitched_frac_mapping<uint32_t>* zip_info,
            const uint32_t _start_dex, const uint32_t signal_len, const uint8_t _load_H = 2);


        template <bool _conj>
        static void store_entire_row_transpose_cplxf_zip(decx::utils::double_buffer_manager* __restrict _double_buffer, de::CPf* __restrict dst_head_ptr,
            const decx::dsp::fft::FKT1D* _tiles, const uint32_t _load_len_v4, const uint32_t _pitch_src, const decx::utils::unpitched_frac_mapping<uint32_t>* zip_info,
            const uint32_t _start_dex, const uint8_t _load_H = 4);


        template <bool _conj>
        static void store_entire_row_transpose_cplxd_zip(decx::utils::double_buffer_manager* __restrict _double_buffer, de::CPd* __restrict dst_head_ptr,
            const decx::dsp::fft::FKT1D* _tiles, const uint32_t _load_len_v2, const uint32_t _pitch_src, const decx::utils::unpitched_frac_mapping<uint32_t>* zip_info,
            const uint32_t _start_dex, const uint8_t _load_H = 2);


        static void store_entire_row_transpose_cplxf_fp32_zip(decx::utils::double_buffer_manager* __restrict _double_buffer, float* __restrict dst_head_ptr,
            const decx::dsp::fft::FKT1D* _tiles, const uint32_t _load_len_v4, const uint32_t _pitch_src, const decx::utils::unpitched_frac_mapping<uint32_t>* zip_info,
            const uint32_t _start_dex, const uint8_t _load_H = 4);


        static void store_entire_row_transpose_cplxd_fp64_zip(decx::utils::double_buffer_manager* __restrict _double_buffer, double* __restrict dst_head_ptr,
            const decx::dsp::fft::FKT1D* _tiles, const uint32_t _load_len_v2, const uint32_t _pitch_src, const decx::utils::unpitched_frac_mapping<uint32_t>* zip_info,
            const uint32_t _start_dex, const uint8_t _load_H = 2);


        static void store_entire_row_transpose_cplxf_u8_zip(decx::utils::double_buffer_manager* __restrict _double_buffer, uint8_t* __restrict dst_head_ptr,
            const decx::dsp::fft::FKT1D* _tiles, const uint32_t _load_len_v4, const uint32_t _pitch_src, const decx::utils::unpitched_frac_mapping<uint32_t>* zip_info,
            const uint32_t _start_dex, const uint8_t _load_H = 4);


        static void store_entire_row_transpose_cplxd_u8_zip(decx::utils::double_buffer_manager* __restrict _double_buffer, uint8_t* __restrict dst_head_ptr,
            const decx::dsp::fft::FKT1D* _tiles, const uint32_t _load_len_v2, const uint32_t _pitch_src, const decx::utils::unpitched_frac_mapping<uint32_t>* zip_info,
            const uint32_t _start_dex, const uint8_t _load_H = 2);
    }
}
}
}


static void decx::dsp::fft::CPUK::
load_entire_row_transpose_fp32_zip(const float* __restrict src_head_ptr, 
                                   decx::utils::double_buffer_manager* __restrict _double_buffer,
                                   const decx::dsp::fft::FKT1D*              _tiles, 
                                   const uint32_t                                 _load_len_v4, 
                                   const uint32_t                                 _pitch_src, 
                                   const decx::utils::unpitched_frac_mapping<uint32_t>* zip_info,
                                   const uint32_t                                 _start_dex, 
                                   const uint8_t                                  _load_H)
{
    if (zip_info->_is_zipped) 
    {
        const float* _src_row_ptr = NULL;
        float* _dst_row_ptr = NULL;

        for (uint8_t i = 0; i < _load_H; ++i)
        {
            const uint32_t _phyaddr_H = zip_info->_is_Level2 ? zip_info->get_phyaddr_L2(_start_dex + i) :
                zip_info->get_phyaddr_L1(_start_dex + i);
            _src_row_ptr = src_head_ptr + _phyaddr_H * _pitch_src;
            _dst_row_ptr = _double_buffer->get_lagging_ptr<float>() + i * _tiles->_tile_row_pitch;

            for (uint32_t j = 0; j < _load_len_v4; ++j) {
                _mm_store_ps(_dst_row_ptr + (j << 2), _mm_load_ps(_src_row_ptr + (j << 2)));
            }
        }
        // Update the status of the double buffer
        _double_buffer->update_states();
        // In-block transpose
        _tiles->_inblock_transpose_vecDist_2_VecAdj_fp32(_double_buffer);
    }
    else {
        decx::dsp::fft::CPUK::load_entire_row_transpose_fp32(src_head_ptr + _start_dex * _pitch_src, 
            _double_buffer, _tiles, _load_len_v4, _pitch_src, _load_H);
    }
}




static void decx::dsp::fft::CPUK::
load_entire_row_transpose_fp64_zip(const double* __restrict src_head_ptr, 
                                   decx::utils::double_buffer_manager* __restrict _double_buffer,
                                   const decx::dsp::fft::FKT1D*                   _tiles, 
                                   const uint32_t                                 _load_len_v2, 
                                   const uint32_t                                 _pitch_src, 
                                   const decx::utils::unpitched_frac_mapping<uint32_t>* zip_info,
                                   const uint32_t                                 _start_dex, 
                                   const uint8_t                                  _load_H)
{
    if (zip_info->_is_zipped) 
    {
        const double* _src_row_ptr = NULL;
        double* _dst_row_ptr = NULL;

        for (uint8_t i = 0; i < _load_H; ++i)
        {
            const uint32_t _phyaddr_H = zip_info->_is_Level2 ? zip_info->get_phyaddr_L2(_start_dex + i) :
                zip_info->get_phyaddr_L1(_start_dex + i);
            _src_row_ptr = src_head_ptr + _phyaddr_H * _pitch_src;
            _dst_row_ptr = _double_buffer->get_lagging_ptr<double>() + i * _tiles->_tile_row_pitch;

            for (uint32_t j = 0; j < _load_len_v2; ++j) {
                _mm_store_pd(_dst_row_ptr + (j << 1), _mm_load_pd(_src_row_ptr + (j << 1)));
            }
        }
        // Update the status of the double buffer
        _double_buffer->update_states();
        // In-block transpose
        _tiles->_inblock_transpose_vecDist_2_VecAdj_fp64(_double_buffer);
    }
    else {
        decx::dsp::fft::CPUK::load_entire_row_transpose_fp64(src_head_ptr + _start_dex * _pitch_src, 
            _double_buffer, _tiles, _load_len_v2, _pitch_src, _load_H);
    }
}



static void decx::dsp::fft::CPUK::
load_entire_row_transpose_u8_fp32_zip(const uint8_t* __restrict src_head_ptr, 
                                   decx::utils::double_buffer_manager* __restrict _double_buffer,
                                   const decx::dsp::fft::FKT1D*              _tiles, 
                                   const uint32_t                                 _load_len_v4, 
                                   const uint32_t                                 _pitch_src, 
                                   const decx::utils::unpitched_frac_mapping<uint32_t>* zip_info,
                                   const uint32_t                                 _start_dex, 
                                   const uint8_t                                  _load_H)
{
    if (zip_info->_is_zipped) 
    {
        const float* _src_row_ptr = NULL;
        float* _dst_row_ptr = NULL;

        decx::utils::simd::xmm128_reg _reg;

        for (uint8_t i = 0; i < _load_H; ++i)
        {
            const uint32_t _phyaddr_H = zip_info->_is_Level2 ? zip_info->get_phyaddr_L2(_start_dex + i) :
                                                               zip_info->get_phyaddr_L1(_start_dex + i);
            _src_row_ptr = (float*)(src_head_ptr + _phyaddr_H * _pitch_src);
            _dst_row_ptr = _double_buffer->get_lagging_ptr<float>() + i * _tiles->_tile_row_pitch;

            for (uint32_t j = 0; j < _load_len_v4; ++j) {
                _reg._vf = _mm_loadu_ps(_src_row_ptr + j);
                _reg._vi = _mm_cvtepu8_epi32(_reg._vi);
                _mm_store_ps(_dst_row_ptr + (j << 2), _mm_cvtepi32_ps(_reg._vi));
            }
        }
        // Update the status of the double buffer
        _double_buffer->update_states();
        // In-block transpose
        _tiles->_inblock_transpose_vecDist_2_VecAdj_fp32(_double_buffer);
    }
    else {
        decx::dsp::fft::CPUK::load_entire_row_transpose_u8_fp32((float*)(src_head_ptr + _start_dex * _pitch_src),
            _double_buffer, _tiles, _load_len_v4, _pitch_src, _load_H);
    }
}



static void decx::dsp::fft::CPUK::
load_entire_row_transpose_u8_fp64_zip(const uint8_t* __restrict src_head_ptr, 
                                   decx::utils::double_buffer_manager* __restrict _double_buffer,
                                   const decx::dsp::fft::FKT1D*              _tiles, 
                                   const uint32_t                                 _load_len_v2, 
                                   const uint32_t                                 _pitch_src, 
                                   const decx::utils::unpitched_frac_mapping<uint32_t>* zip_info,
                                   const uint32_t                                 _start_dex, 
                                   const uint8_t                                  _load_H)
{
    if (zip_info->_is_zipped) 
    {
        const int16_t* _src_row_ptr = NULL;
        double* _dst_row_ptr = NULL;

        decx::utils::simd::xmm128_reg _reg;

        for (uint8_t i = 0; i < _load_H; ++i)
        {
            const uint32_t _phyaddr_H = zip_info->_is_Level2 ? zip_info->get_phyaddr_L2(_start_dex + i) :
                                                               zip_info->get_phyaddr_L1(_start_dex + i);
            _src_row_ptr = (int16_t*)(src_head_ptr + _phyaddr_H * _pitch_src);
            _dst_row_ptr = _double_buffer->get_lagging_ptr<double>() + i * _tiles->_tile_row_pitch;

            for (uint32_t j = 0; j < _load_len_v2; ++j) {
                _reg._arrs[0] = _src_row_ptr[j];
                _reg._vi = _mm_cvtepu8_epi64(_reg._vi);

                //_mm_store_pd(_dst_row_ptr + (j << 1), _mm_cvtepi64_pd(_reg._vi));
                _reg._arrd[0] = (double)_reg._arrull[0];
                _reg._arrd[1] = (double)_reg._arrull[1];
                _mm_store_pd(_dst_row_ptr + (j << 1), _reg._vd);
            }
        }
        // Update the status of the double buffer
        _double_buffer->update_states();
        // In-block transpose
        _tiles->_inblock_transpose_vecDist_2_VecAdj_fp64(_double_buffer);
    }
    else {
        decx::dsp::fft::CPUK::load_entire_row_transpose_u8_fp64((int16_t*)(src_head_ptr + _start_dex * _pitch_src),
            _double_buffer, _tiles, _load_len_v2, _pitch_src, _load_H);
    }
}


template <bool _IFFT> static void decx::dsp::fft::CPUK::
load_entire_row_transpose_cplxf_zip(const de::CPf* __restrict src_head_ptr, 
                                    decx::utils::double_buffer_manager* __restrict  _double_buffer,
                                    const decx::dsp::fft::FKT1D*               _tiles, 
                                    const uint32_t                                  _load_len_v4, 
                                    const uint32_t                                  _pitch_src, 
                                    const decx::utils::unpitched_frac_mapping<uint32_t>* zip_info,
                                    const uint32_t                                  _start_dex, 
                                    const uint32_t                                  signal_len, 
                                    const uint8_t                                   _load_H)
{
    if (zip_info->_is_zipped)
    {
        const double* _src_row_ptr = NULL;
        double* _dst_row_ptr = NULL;

        decx::utils::simd::xmm256_reg _reg;

        for (uint8_t i = 0; i < _load_H; ++i)
        {
            const uint32_t _phyaddr_H = zip_info->_is_Level2 ? zip_info->get_phyaddr_L2(_start_dex + i) :
                                                               zip_info->get_phyaddr_L1(_start_dex + i);
            _src_row_ptr = (double*)src_head_ptr + _phyaddr_H * _pitch_src;
            _dst_row_ptr = _double_buffer->get_lagging_ptr<double>() + i * _tiles->_tile_row_pitch;

            for (uint32_t j = 0; j < _load_len_v4; ++j) {
                _reg._vd = _mm256_load_pd(_src_row_ptr + (j << 2));
                if constexpr (_IFFT) { _reg._vf = _mm256_div_ps(_reg._vf, _mm256_set1_ps(signal_len)); }
                //_reg._vf = _mm256_setr_ps(j * 4, 0, j * 4 + 1, 0, j * 4 + 2, 0, j * 4 + 3, 0);
                _mm256_store_pd(_dst_row_ptr + (j << 2), _reg._vd);
            }
        }
        // Update the status of the double buffer
        _double_buffer->update_states();
        // In-block transpose
        _tiles->_inblock_transpose_vecDist_2_VecAdj_cplxf(_double_buffer);
    }
    else {
        decx::dsp::fft::CPUK::load_entire_row_transpose_cplxf<_IFFT>(src_head_ptr + _start_dex * _pitch_src,
            _double_buffer, _tiles, _load_len_v4, _pitch_src, _load_H, signal_len);
    }
}



template <bool _IFFT> static void decx::dsp::fft::CPUK::
load_entire_row_transpose_cplxd_zip(const de::CPd* __restrict src_head_ptr, 
                                    decx::utils::double_buffer_manager* __restrict  _double_buffer,
                                    const decx::dsp::fft::FKT1D*                    _tiles, 
                                    const uint32_t                                  _load_len_v2, 
                                    const uint32_t                                  _pitch_src, 
                                    const decx::utils::unpitched_frac_mapping<uint32_t>* zip_info,
                                    const uint32_t                                  _start_dex, 
                                    const uint32_t                                  signal_len, 
                                    const uint8_t                                   _load_H)
{
    if (zip_info->_is_zipped)
    {
        const de::CPd* _src_row_ptr = NULL;
        de::CPd* _dst_row_ptr = NULL;

        decx::utils::simd::xmm256_reg _reg;

        for (uint8_t i = 0; i < _load_H; ++i)
        {
            const uint32_t _phyaddr_H = zip_info->_is_Level2 ? zip_info->get_phyaddr_L2(_start_dex + i) :
                                                               zip_info->get_phyaddr_L1(_start_dex + i);
            _src_row_ptr = src_head_ptr + _phyaddr_H * _pitch_src;
            _dst_row_ptr = _double_buffer->get_lagging_ptr<de::CPd>() + i * _tiles->_tile_row_pitch;

            for (uint32_t j = 0; j < _load_len_v2; ++j) {
                _reg._vd = _mm256_load_pd((double*)(_src_row_ptr + (j << 1)));
                if constexpr (_IFFT) { _reg._vd = _mm256_div_pd(_reg._vd, _mm256_set1_pd(signal_len)); }
                //_reg._vf = _mm256_setr_ps(j * 4, 0, j * 4 + 1, 0, j * 4 + 2, 0, j * 4 + 3, 0);
                _mm256_store_pd((double*)(_dst_row_ptr + (j << 1)), _reg._vd);
            }
        }
        // Update the status of the double buffer
        _double_buffer->update_states();
        // In-block transpose
        _tiles->_inblock_transpose_vecDist_2_VecAdj_cplxd(_double_buffer);
    }
    else {
        decx::dsp::fft::CPUK::load_entire_row_transpose_cplxd<_IFFT>(src_head_ptr + _start_dex * _pitch_src,
            _double_buffer, _tiles, _load_len_v2, _pitch_src, _load_H, signal_len);
    }
}



template <bool _conj> static void
decx::dsp::fft::CPUK::store_entire_row_transpose_cplxf_zip(decx::utils::double_buffer_manager* __restrict _double_buffer, 
                                                       de::CPf* __restrict dst_head_ptr,
                                                       const decx::dsp::fft::FKT1D* _tiles,
                                                       const uint32_t _load_len_v4, 
                                                       const uint32_t _pitch_dst, 
                                                       const decx::utils::unpitched_frac_mapping<uint32_t>* zip_info,
                                                       const uint32_t                                 _start_dex, 
                                                       const uint8_t _load_H)
{
    if (zip_info->_is_zipped)
    {
        // Directly store back after transposed back to vecDist
        _tiles->_inblock_transpose_vecAdj_2_VecDist_cplxf(_double_buffer);

        const double* _src_row_ptr = NULL;
        double* _dst_row_ptr = NULL;

        decx::utils::simd::xmm256_reg _reg;

        for (uint8_t i = 0; i < _load_H; ++i) 
        {
            _src_row_ptr = _double_buffer->get_leading_ptr<double>() + i * _tiles->_tile_row_pitch;
            const uint32_t _phyaddr_H = zip_info->_is_Level2 ? zip_info->get_phyaddr_L2(_start_dex + i) :
                                                               zip_info->get_phyaddr_L1(_start_dex + i);

            _dst_row_ptr = (double*)dst_head_ptr + _phyaddr_H * _pitch_dst;

            for (uint32_t j = 0; j < _load_len_v4; ++j)
            {
                _reg._vd = _mm256_load_pd(_src_row_ptr + (j << 2));

                if constexpr (_conj) {
                    _reg._vf = decx::dsp::CPUK::_cp4_conjugate_fp32(_reg._vf);
                }
                //_reg._vf = _mm256_set1_ps(37);
                _mm256_store_pd(_dst_row_ptr + (j << 2), _reg._vd);
            }
        }
    }
    else {
        decx::dsp::fft::CPUK::store_entire_row_transpose_cplxf<_conj>(_double_buffer, dst_head_ptr + _start_dex * _pitch_dst,
            _tiles, _load_len_v4, _pitch_dst, _load_H);
    }
}




template <bool _conj> static void
decx::dsp::fft::CPUK::
store_entire_row_transpose_cplxd_zip(decx::utils::double_buffer_manager* __restrict _double_buffer, 
                                     de::CPd* __restrict dst_head_ptr,
                                     const decx::dsp::fft::FKT1D* _tiles,
                                     const uint32_t _load_len_v4, 
                                     const uint32_t _pitch_dst, 
                                     const decx::utils::unpitched_frac_mapping<uint32_t>* zip_info,
                                     const uint32_t                                 _start_dex, 
                                     const uint8_t _load_H)
{
    if (zip_info->_is_zipped)
    {
        // Directly store back after transposed back to vecDist
        _tiles->_inblock_transpose_vecAdj_2_VecDist_cplxd(_double_buffer);

        const de::CPd* _src_row_ptr = NULL;
        de::CPd* _dst_row_ptr = NULL;

        decx::utils::simd::xmm256_reg _reg;

        for (uint8_t i = 0; i < _load_H; ++i) 
        {
            _src_row_ptr = _double_buffer->get_leading_ptr<de::CPd>() + i * _tiles->_tile_row_pitch;
            const uint32_t _phyaddr_H = zip_info->_is_Level2 ? zip_info->get_phyaddr_L2(_start_dex + i) :
                                                               zip_info->get_phyaddr_L1(_start_dex + i);

            _dst_row_ptr = dst_head_ptr + _phyaddr_H * _pitch_dst;

            for (uint32_t j = 0; j < _load_len_v4; ++j)
            {
                _reg._vd = _mm256_load_pd((double*)(_src_row_ptr + (j << 1)));

                if constexpr (_conj) {
                    _reg._vd = decx::dsp::CPUK::_cp2_conjugate_fp64(_reg._vd);
                }
                //_reg._vf = _mm256_set1_ps(37);
                _mm256_store_pd((double*)(_dst_row_ptr + (j << 1)), _reg._vd);
            }
        }
    }
    else {
        decx::dsp::fft::CPUK::store_entire_row_transpose_cplxd<_conj>(_double_buffer, 
            dst_head_ptr + _start_dex * _pitch_dst, _tiles, _load_len_v4, _pitch_dst, _load_H);
    }
}



static void decx::dsp::fft::CPUK::
store_entire_row_transpose_cplxf_fp32_zip(decx::utils::double_buffer_manager* __restrict _double_buffer, 
                                          float* __restrict dst_head_ptr,
                                          const decx::dsp::fft::FKT1D* _tiles,
                                          const uint32_t _load_len_v4, 
                                          const uint32_t _pitch_dst, 
                                          const decx::utils::unpitched_frac_mapping<uint32_t>* zip_info,
                                          const uint32_t                                 _start_dex, 
                                          const uint8_t _load_H)
{
    if (zip_info->_is_zipped)
    {
        // Directly store back after transposed back to vecDist
        _tiles->_inblock_transpose_vecAdj_2_VecDist_cplxf(_double_buffer);

        const double* _src_row_ptr = NULL;
        float* _dst_row_ptr = NULL;

        decx::utils::simd::xmm256_reg _reg;

        for (uint8_t i = 0; i < _load_H; ++i) 
        {
            _src_row_ptr = _double_buffer->get_leading_ptr<double>() + i * _tiles->_tile_row_pitch;
            const uint32_t _phyaddr_H = zip_info->_is_Level2 ? zip_info->get_phyaddr_L2(_start_dex + i) :
                                                               zip_info->get_phyaddr_L1(_start_dex + i);

            _dst_row_ptr = dst_head_ptr + _phyaddr_H * _pitch_dst;

            for (uint32_t j = 0; j < _load_len_v4; ++j)
            {
                _reg._vd = _mm256_load_pd(_src_row_ptr + (j << 2));
                _reg._vf = _mm256_permutevar8x32_ps(_reg._vf, _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7));
                
                _mm_store_ps(_dst_row_ptr + (j << 2), _mm256_castps256_ps128(_reg._vf));
            }
        }
    }
    else {
        decx::dsp::fft::CPUK::store_entire_row_transpose_cplxf_fp32(_double_buffer, dst_head_ptr + _start_dex * _pitch_dst,
            _tiles, _load_len_v4, _pitch_dst, _load_H);
    }
}




static void decx::dsp::fft::CPUK::
store_entire_row_transpose_cplxd_fp64_zip(decx::utils::double_buffer_manager* __restrict _double_buffer, 
                                          double* __restrict dst_head_ptr,
                                          const decx::dsp::fft::FKT1D* _tiles,
                                          const uint32_t _load_len_v4, 
                                          const uint32_t _pitch_dst, 
                                          const decx::utils::unpitched_frac_mapping<uint32_t>* zip_info,
                                          const uint32_t                                 _start_dex, 
                                          const uint8_t _load_H)
{
    if (zip_info->_is_zipped)
    {
        // Directly store back after transposed back to vecDist
        _tiles->_inblock_transpose_vecAdj_2_VecDist_cplxd(_double_buffer);

        const de::CPd* _src_row_ptr = NULL;
        double* _dst_row_ptr = NULL;

        decx::utils::simd::xmm256_reg _reg;

        for (uint8_t i = 0; i < _load_H; ++i) 
        {
            _src_row_ptr = _double_buffer->get_leading_ptr<de::CPd>() + i * _tiles->_tile_row_pitch;
            const uint32_t _phyaddr_H = zip_info->_is_Level2 ? zip_info->get_phyaddr_L2(_start_dex + i) :
                                                               zip_info->get_phyaddr_L1(_start_dex + i);

            _dst_row_ptr = dst_head_ptr + _phyaddr_H * _pitch_dst;

            for (uint32_t j = 0; j < _load_len_v4; ++j)
            {
                _reg._vd = _mm256_load_pd((double*)(_src_row_ptr + (j << 1)));
                _reg._vd = _mm256_permute4x64_pd(_reg._vd, 0b11011000);
                
                _mm_store_pd(_dst_row_ptr + (j << 1), _mm256_castpd256_pd128(_reg._vd));
            }
        }
    }
    else {
        decx::dsp::fft::CPUK::store_entire_row_transpose_cplxd_fp64(_double_buffer, 
            dst_head_ptr + _start_dex * _pitch_dst, _tiles, _load_len_v4, _pitch_dst, _load_H);
    }
}



static void decx::dsp::fft::CPUK::
store_entire_row_transpose_cplxf_u8_zip(decx::utils::double_buffer_manager* __restrict _double_buffer, 
                                          uint8_t* __restrict dst_head_ptr,
                                          const decx::dsp::fft::FKT1D* _tiles,
                                          const uint32_t _load_len_v4, 
                                          const uint32_t _pitch_dst, 
                                          const decx::utils::unpitched_frac_mapping<uint32_t>* zip_info,
                                          const uint32_t                                 _start_dex, 
                                          const uint8_t _load_H)
{
    if (zip_info->_is_zipped)
    {
        // Directly store back after transposed back to vecDist
        _tiles->_inblock_transpose_vecAdj_2_VecDist_cplxf(_double_buffer);

        const double* _src_row_ptr = NULL;
        int32_t* _dst_row_ptr = NULL;

        decx::utils::simd::xmm256_reg _reg;
        decx::utils::simd::xmm128_reg _reg1;

        for (uint8_t i = 0; i < _load_H; ++i) 
        {
            _src_row_ptr = _double_buffer->get_leading_ptr<double>() + i * _tiles->_tile_row_pitch;
            const uint32_t _phyaddr_H = zip_info->_is_Level2 ? zip_info->get_phyaddr_L2(_start_dex + i) :
                                                               zip_info->get_phyaddr_L1(_start_dex + i);

            _dst_row_ptr = (int32_t*)(dst_head_ptr + _phyaddr_H * _pitch_dst);

            for (uint32_t j = 0; j < _load_len_v4; ++j)
            {
                _reg._vd = _mm256_load_pd(_src_row_ptr + (j << 2));
                _reg._vf = _mm256_permutevar8x32_ps(_reg._vf, _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7));

                _reg1._vf = _mm256_castps256_ps128(_reg._vf);
                _reg1._vi = _mm_cvtps_epi32(_reg1._vf);
                _reg1._vi = _mm_shuffle_epi8(_reg1._vi, _mm_set1_epi32(0x0C080400));

                _dst_row_ptr[j] = _reg1._arri[0];
            }
        }
    }
    else {
        decx::dsp::fft::CPUK::store_entire_row_transpose_cplxf_u8(_double_buffer, 
            (int32_t*)(dst_head_ptr + _start_dex * _pitch_dst), _tiles, _load_len_v4, _pitch_dst, _load_H);
    }
}




static void decx::dsp::fft::CPUK::
store_entire_row_transpose_cplxd_u8_zip(decx::utils::double_buffer_manager* __restrict _double_buffer, 
                                          uint8_t* __restrict dst_head_ptr,
                                          const decx::dsp::fft::FKT1D* _tiles,
                                          const uint32_t _load_len_v2, 
                                          const uint32_t _pitch_dst, 
                                          const decx::utils::unpitched_frac_mapping<uint32_t>* zip_info,
                                          const uint32_t                                 _start_dex, 
                                          const uint8_t _load_H)
{
    if (zip_info->_is_zipped)
    {
        // Directly store back after transposed back to vecDist
        _tiles->_inblock_transpose_vecAdj_2_VecDist_cplxd(_double_buffer);

        const de::CPd* _src_row_ptr = NULL;
        int16_t* _dst_row_ptr = NULL;

        decx::utils::simd::xmm256_reg _reg;
        decx::utils::simd::xmm128_reg _reg1;

        for (uint8_t i = 0; i < _load_H; ++i) 
        {
            _src_row_ptr = _double_buffer->get_leading_ptr<de::CPd>() + i * _tiles->_tile_row_pitch;
            const uint32_t _phyaddr_H = zip_info->_is_Level2 ? zip_info->get_phyaddr_L2(_start_dex + i) :
                                                               zip_info->get_phyaddr_L1(_start_dex + i);

            _dst_row_ptr = (int16_t*)(dst_head_ptr + _phyaddr_H * _pitch_dst);

            for (uint32_t j = 0; j < _load_len_v2; ++j)
            {
                _reg._vd = _mm256_load_pd((double*)(_src_row_ptr + (j << 1)));
                _reg._vd = _mm256_permute4x64_pd(_reg._vd, 0b11011000);

                _reg1._vd = _mm256_castpd256_pd128(_reg._vd);
                _reg1._vi = _mm_cvtpd_epi32(_reg1._vd);
                _reg1._vi = _mm_shuffle_epi8(_reg1._vi, _mm_set1_epi32(0x0C080400));

                _dst_row_ptr[j] = _reg1._arrs[0];
            }
        }
    }
    else {
        decx::dsp::fft::CPUK::store_entire_row_transpose_cplxd_u8(_double_buffer, 
            (int16_t*)(dst_head_ptr + _start_dex * _pitch_dst), _tiles, _load_len_v2, _pitch_dst, _load_H);
    }
}



#endif
