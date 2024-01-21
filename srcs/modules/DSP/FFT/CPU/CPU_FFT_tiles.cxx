/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#include "CPU_FFT_tiles.h"


//
//uint64_t decx::dsp::fft::FKI1D::get_warp_num() const
//{
//    return this->_signal_len / this->_warp_proc_len;
//}



void decx::dsp::fft::_FFT1D_kernel_tile_fp32::allocate_tile(const uint32_t tile_frag_len, de::DH* handle)
{
    this->_tile_row_pitch = tile_frag_len;
    this->_tile_len = tile_frag_len * 4;        // vec4

    if (decx::alloc::_host_virtual_page_malloc(&this->_tmp_ptr, this->_tile_len * 2 * sizeof(de::CPf))) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_ALLOCATION,
            ALLOC_FAIL);
        return;
    }
}

void decx::dsp::fft::_FFT1D_kernel_tile_fp32::release()
{
    decx::alloc::_host_virtual_page_dealloc(&this->_tmp_ptr);
}



void decx::dsp::fft::_FFT1D_kernel_tile_fp32::
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



void decx::dsp::fft::_FFT1D_kernel_tile_fp32::
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


void decx::dsp::fft::_FFT1D_kernel_tile_fp32::
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