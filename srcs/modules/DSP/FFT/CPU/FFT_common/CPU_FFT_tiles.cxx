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


#include "CPU_FFT_tiles.h"


template <typename _data_type>
void decx::dsp::fft::_FFT1D_kernel_tile::allocate_tile(const uint32_t tile_frag_len, de::DH* handle)
{
    constexpr uint32_t _alignment = 256 / (sizeof(_data_type) * 8 * 2);
    this->_tile_row_pitch = decx::utils::align<uint32_t>(tile_frag_len, _alignment);
    this->_tile_len = this->_tile_row_pitch * _alignment;        // vec(x)

    this->_total_size = this->_tile_len * 2 * (sizeof(_data_type) * 2);

    if (decx::alloc::_host_virtual_page_malloc(&this->_tmp_ptr, this->_total_size)) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_ALLOCATION,
            ALLOC_FAIL);
        return;
    }
}

template void decx::dsp::fft::_FFT1D_kernel_tile::allocate_tile<float>(const uint32_t, de::DH*);
template void decx::dsp::fft::_FFT1D_kernel_tile::allocate_tile<double>(const uint32_t, de::DH*);


void decx::dsp::fft::_FFT1D_kernel_tile::release()
{
    decx::alloc::_host_virtual_page_dealloc(&this->_tmp_ptr);
}


void decx::dsp::fft::_FFT1D_kernel_tile::flush() const
{
    memset(this->_tmp_ptr.ptr, 0, this->_total_size);
}


void decx::dsp::fft::_FFT1D_kernel_tile::
_inblock_transpose_vecAdj_2_VecDist_cplxf(decx::utils::double_buffer_manager* __restrict _double_buffer) const
{
    const double* src = _double_buffer->get_leading_ptr<double>();
    double* dst = _double_buffer->get_lagging_ptr<double>();

    // In-block transpose
    __m128d _reg[2], _transposed[2];

    const uint32_t frag_len_v2 = this->_tile_row_pitch / 2;

    for (uint32_t z = 0; z < 2; ++z) {
        for (uint32_t i = 0; i < frag_len_v2; ++i) {
            _reg[0] = _mm_load_pd(src + (i << 3));
            _reg[1] = _mm_load_pd(src + (i << 3) + 4);

            _AVX_MM128_TRANSPOSE_2X2_(_reg, _transposed);

            _mm_store_pd(dst + (i << 1), _transposed[0]);
            _mm_store_pd(dst + (i << 1) + this->_tile_row_pitch, _transposed[1]);
        }
        src += 2;
        dst += 2 * this->_tile_row_pitch;
    }
    _double_buffer->update_states();
}



void decx::dsp::fft::_FFT1D_kernel_tile::
_inblock_transpose_vecAdj_2_VecDist_cplxd(decx::utils::double_buffer_manager* __restrict _double_buffer) const
{
    const de::CPd* src = _double_buffer->get_leading_ptr<de::CPd>();
    de::CPd* dst = _double_buffer->get_lagging_ptr<de::CPd>();

    // In-block transpose
    __m256d _reg[2], _transposed[2];

    const uint32_t frag_len_v2 = this->_tile_row_pitch / 2;

    for (uint32_t i = 0; i < frag_len_v2; ++i) {
        _reg[0] = _mm256_load_pd((double*)(src + (i << 2)));
        _reg[1] = _mm256_load_pd((double*)(src + (i << 2) + 2));

        _AVX_MM256_TRANSPOSE_2X2_(_reg, _transposed);

        _mm256_store_pd((double*)(dst + (i << 1)), _transposed[0]);
        _mm256_store_pd((double*)(dst + (i << 1) + this->_tile_row_pitch), _transposed[1]);
    }

    _double_buffer->update_states();
}



void decx::dsp::fft::_FFT1D_kernel_tile::
_inblock_transpose_vecDist_2_VecAdj_fp32(decx::utils::double_buffer_manager* __restrict _double_buffer) const
{
    __m128 _reg[4], _store[4];

    const float* src = _double_buffer->get_leading_ptr<float>();
    float* dst = _double_buffer->get_lagging_ptr<float>();

    for (uint32_t i = 0; i < this->_tile_row_pitch / 4; ++i) {
        _reg[0] = _mm_load_ps(src + (i << 2));
        _reg[1] = _mm_load_ps(src + (i << 2) + this->_tile_row_pitch);
        _reg[2] = _mm_load_ps(src + (i << 2) + this->_tile_row_pitch * 2);
        _reg[3] = _mm_load_ps(src + (i << 2) + this->_tile_row_pitch * 3);

        _AVX_MM128_TRANSPOSE_4X4_(_reg, _store);

        _mm_store_ps(dst + (i << 4), _reg[0]);
        _mm_store_ps(dst + (i << 4) + 4, _reg[1]);
        _mm_store_ps(dst + (i << 4) + 8, _reg[2]);
        _mm_store_ps(dst + (i << 4) + 12, _reg[3]);
    }
    _double_buffer->update_states();
}



void decx::dsp::fft::_FFT1D_kernel_tile::
_inblock_transpose_vecDist_2_VecAdj_fp64(decx::utils::double_buffer_manager* __restrict _double_buffer) const
{
    const double* src = _double_buffer->get_leading_ptr<double>();
    double* dst = _double_buffer->get_lagging_ptr<double>();

    // In-block transpose
    __m128d _reg[2], _transposed[2];

    const uint32_t frag_len_v2 = this->_tile_row_pitch / 2;

    for (uint32_t i = 0; i < frag_len_v2; ++i) {
        _reg[0] = _mm_load_pd(src + (i << 1));
        _reg[1] = _mm_load_pd(src + (i << 1) + this->_tile_row_pitch);

        _AVX_MM128_TRANSPOSE_2X2_(_reg, _transposed);

        _mm_store_pd(dst + (i << 2), _transposed[0]);
        _mm_store_pd(dst + (i << 2) + 2, _transposed[1]);
    }
    _double_buffer->update_states();
}



void decx::dsp::fft::_FFT1D_kernel_tile::
_inblock_transpose_vecDist_2_VecAdj_cplxf(decx::utils::double_buffer_manager* __restrict _double_buffer) const
{
    const double* src = _double_buffer->get_leading_ptr<double>();
    double* dst = _double_buffer->get_lagging_ptr<double>();

    // In-block transpose
    __m128d _reg[2], _transposed[2];

    const uint32_t frag_len_v2 = this->_tile_row_pitch / 2;

    for (uint32_t i = 0; i < frag_len_v2; ++i) {
        for (uint8_t j = 0; j < 2; ++j) {
            _reg[0] = _mm_load_pd(src + (i << 1) + j * (this->_tile_row_pitch << 1));
            _reg[1] = _mm_load_pd(src + (i << 1) + j * (this->_tile_row_pitch << 1) + this->_tile_row_pitch);

            _AVX_MM128_TRANSPOSE_2X2_(_reg, _transposed);

            _mm_store_pd(dst + (i << 3) + (j << 1), _transposed[0]);
            _mm_store_pd(dst + (i << 3) + (j << 1) + 4, _transposed[1]);
        }
    }
    _double_buffer->update_states();
}



void decx::dsp::fft::_FFT1D_kernel_tile::
_inblock_transpose_vecDist_2_VecAdj_cplxd(decx::utils::double_buffer_manager* __restrict _double_buffer) const
{
    const de::CPd* src = _double_buffer->get_leading_ptr<de::CPd>();
    de::CPd* dst = _double_buffer->get_lagging_ptr<de::CPd>();

    // In-block transpose
    __m256d _reg[2], _transposed[2];

    const uint32_t frag_len_v2 = this->_tile_row_pitch / 2;

    for (uint32_t i = 0; i < frag_len_v2; ++i) {
        _reg[0] = _mm256_load_pd((double*)(src + (i << 1)));
        _reg[1] = _mm256_load_pd((double*)(src + (i << 1) + this->_tile_row_pitch));

        _AVX_MM256_TRANSPOSE_2X2_(_reg, _transposed);

        _mm256_store_pd((double*)(dst + (i << 2)), _transposed[0]);
        _mm256_store_pd((double*)(dst + (i << 2) + 2), _transposed[1]);
    }
    _double_buffer->update_states();
}
