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


#include "../extend_reflect_exec.h"


namespace decx
{
namespace bp {
namespace CPUK 
{
    _THREAD_CALL_ inline void
    _rev_load_reflection_region_b32_v4(const float* __restrict src, float* __restrict dst, const uint32_t reflect_depth_v4)
    {
        float32x4_t _reg;
        for (uint32_t i = 0; i < reflect_depth_v4; ++i) {
            _reg = vld1q_f32(src + (reflect_depth_v4 - i - 1) * 4);
            _reg = vrev64q_f32(_reg);
            _reg = vextq_f32(_reg, _reg, 2);
            vst1q_f32(dst + i * 4, _reg);
        }
    }



    _THREAD_CALL_ inline void
    _rev_load_reflection_region_b64_v2(const double* __restrict src, double* __restrict dst, const uint32_t reflect_depth_v2)
    {
        float64x2_t _reg;
        for (uint32_t i = 0; i < reflect_depth_v2; ++i) {
            _reg = vld1q_f64(src + (reflect_depth_v2 - i - 1) * 2);
            _reg = vextq_f64(_reg, _reg, 1);
            vst1q_f64(dst + i * 2, _reg);
        }
    }


    _THREAD_CALL_ inline void
    _rev_load_reflection_region_b8_v16(const uint8_t* __restrict src, uint8_t* __restrict dst, const uint32_t reflect_depth_v16)
    {
        uint8x16_t _reg;
        for (uint32_t i = 0; i < reflect_depth_v16; ++i) {
            _reg = vld1q_u8(src + (reflect_depth_v16 - i - 1) * 16);
            _reg = vrev64q_u8(_reg);
            _reg = vextq_u8(_reg, _reg, 8);
            vst1q_u8(dst + i * 16, _reg);
        }
    }

    

    _THREAD_CALL_ inline void
    _rev_load_reflection_region_b16_v8(const uint16_t* __restrict src, uint16_t* __restrict dst, const uint32_t reflect_depth_v8)
    {
        static uint8_t _idx_rev_u16[16] = {14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1};
        const uint8x16_t _idx = vld1q_u8(_idx_rev_u16);

        uint16x8_t _reg, stg;
        for (uint32_t i = 0; i < reflect_depth_v8; ++i) {
            _reg = vld1q_u16(src + (reflect_depth_v8 - i - 1) * 8);
            stg = vreinterpretq_u16_u8(vqtbl1q_u8(vreinterpretq_u8_u16(_reg), _idx));
            vst1q_u16(dst + i * 8, stg);
        }
    }


    _THREAD_CALL_ inline void
    _store_buf_to_dst_b32_v4(const float* __restrict src, float* __restrict dst, const uint32_t store_num_v4)
    {
        for (uint32_t i = 0; i < store_num_v4; ++i) {
            vst1q_f32(dst + i * 4, vld1q_f32(src + i * 4));
        }
    }


    _THREAD_CALL_ inline void
    _store_buf_to_dst_b64_v2(const double* __restrict src, double* __restrict dst, const uint32_t store_num_v2)
    {
        for (uint32_t i = 0; i < store_num_v2; ++i) {
            vst1q_f64(dst + i * 2, vld1q_f64(src + i * 2));
        }
    }


    _THREAD_CALL_ inline void
    _store_buf_to_dst_b8_v16(const uint8_t* __restrict src, uint8_t* __restrict dst, const uint32_t store_num_v16)
    {
        for (uint32_t i = 0; i < store_num_v16; ++i) {
            vst1q_u8(dst + i * 16, vld1q_u8(src + i * 16));
        }
    }


    _THREAD_CALL_ inline void
    _store_buf_to_dst_b16_v8(const uint16_t* __restrict src, uint16_t* __restrict dst, const uint32_t store_num_v8)
    {
        for (uint32_t i = 0; i < store_num_v8 / 2; ++i) {
            vst1q_u16_x2(dst + i * 16, vld1q_u16_x2(src + i * 16));
        }
        if (store_num_v8 & 1){
            vst1q_u16(dst + ((store_num_v8 >> 1) << 4), vld1q_u16(src + ((store_num_v8 >> 1) << 4)));
        }
        
        // for (uint32_t i = 0; i < store_num_v8; ++i) {
        //     vst1q_u16(dst + i * 8, vld1q_u16(src + i * 8));
        // }
    }
}
}
}



_THREAD_FUNCTION_ void
decx::bp::CPUK::_extend_reflect1D_b32(const float* __restrict   src,
                                      float* __restrict         buffer,
                                      float* __restrict         dst,
                                      const decx::bp::extend_reflect_exec_params* b_rfct,
                                      const size_t              _actual_w_v1,
                                      const size_t              _original_w_v4)
{
    const uint8x16_t _idx = decx::bp::e_rfct_exep_get_tbl_b32(b_rfct);

    decx::utils::simd::xmm256_reg reg;
    decx::utils::simd::xmm128_reg store;

    decx::bp::CPUK::_rev_load_reflection_region_b32_v4(src, buffer, b_rfct->_actual_load_num_L / 4);
    decx::bp::CPUK::_store_buf_to_dst_b32_v4(buffer + b_rfct->_L_v8_reflectL, dst, b_rfct->_actual_load_num_L / 4);
    
    reg._vf.val[1] = vld1q_f32(src);
    vst1q_f32(dst + b_rfct->_left, reg._vf.val[1]);

    for (uint32_t i = 1; i < _original_w_v4; ++i) {
        reg._vf.val[0] = reg._vf.val[1];
        reg._vf.val[1] = vld1q_f32(src + i * 4);
        store._vuc = vqtbl2q_u8(reg._vuc, _idx);
        vst1q_f32(dst + b_rfct->_actual_load_num_L + i * 4 - 4, store._vf);
    }
    reg._vf.val[0] = reg._vf.val[1];
    reg._vui.val[1] = veorq_u32(reg._vui.val[1], reg._vui.val[1]);
    store._vuc = vqtbl2q_u8(reg._vuc, _idx);
    vst1q_f32(dst + b_rfct->_actual_load_num_L + _original_w_v4 * 4 - 4, store._vf);

    decx::bp::CPUK::_rev_load_reflection_region_b32_v4(src + (_original_w_v4) * 4 - b_rfct->_actual_load_num_R, 
                                                       buffer, 
                                                       b_rfct->_actual_load_num_R / 4);
    memcpy(dst + _actual_w_v1 + b_rfct->_left, 
           buffer + b_rfct->_rightmost_0num_src + 1, 
           b_rfct->_right * sizeof(float));
}




_THREAD_FUNCTION_ void
decx::bp::CPUK::_extend_H_reflect2D_b32(const float* __restrict   src,
                                      float* __restrict         buffer,
                                      float* __restrict         dst,
                                      const decx::bp::extend_reflect_exec_params* b_rfct,
                                      const uint32_t            Wsrc,
                                      const uint32_t            Wdst,
                                      const uint32_t            _actual_w_v1,
                                      const uint2               _original_dims_v4)
{
    const uint8x16_t _idx = decx::bp::e_rfct_exep_get_tbl_b32(b_rfct);

    decx::utils::simd::xmm256_reg reg;
    decx::utils::simd::xmm128_reg store;

    uint64_t dex_src = 0, dex_dst = 0;

    for (uint32_t i = 0; i < _original_dims_v4.y; ++i) 
    {
        dex_src = i * Wsrc;
        dex_dst = i * Wdst;
        _rev_load_reflection_region_b32_v4(src + dex_src, buffer, b_rfct->_actual_load_num_L / 4);
        _store_buf_to_dst_b32_v4(buffer + b_rfct->_L_v8_reflectL, dst + dex_dst, b_rfct->_actual_load_num_L / 4);
        reg._vf.val[1] = vld1q_f32(src + dex_src);
        vst1q_f32(dst + dex_dst + b_rfct->_left, reg._vf.val[1]);

        dex_src += 4;
        dex_dst += b_rfct->_actual_load_num_L;

        for (uint32_t j = 1; j < _original_dims_v4.x; ++j) {
            reg._vf.val[0] = reg._vf.val[1];
            reg._vf.val[1] = vld1q_f32(src + dex_src);
            store._vuc = vqtbl2q_u8(reg._vuc, _idx);
            vst1q_f32(dst + dex_dst, store._vf);

            dex_src += 4;
            dex_dst += 4;
        }

        reg._vf.val[0] = reg._vf.val[1];
        reg._vui.val[1] = veorq_u32(reg._vui.val[1], reg._vui.val[1]);
        store._vuc = vqtbl2q_u8(reg._vuc, _idx);
        vst1q_f32(dst + dex_dst, store._vf);

        decx::bp::CPUK::_rev_load_reflection_region_b32_v4(src + dex_src - b_rfct->_actual_load_num_R, 
                                                           buffer, 
                                                           b_rfct->_actual_load_num_R / 4);
        memcpy(dst + i * Wdst + b_rfct->_left + _actual_w_v1, 
               buffer + b_rfct->_rightmost_0num_src + 1, 
               b_rfct->_right * sizeof(float));
    }
}



// --------------------------------------------- 8-bit ------------------------------------------------



_THREAD_FUNCTION_ void decx::bp::CPUK::
_extend_reflect1D_b8(const uint8_t* src,
                     uint8_t* buffer, 
                     uint8_t* dst,
                     const decx::bp::extend_reflect_exec_params* b_rfct, 
                     const uint64_t _actual_w_v1, 
                     const uint64_t _original_w_v8)
{
    const uint8x16_t _idx = decx::bp::e_rfct_exep_get_tbl_b8(b_rfct);

    decx::utils::simd::xmm256_reg reg;
    decx::utils::simd::xmm128_reg store;

    decx::bp::CPUK::_rev_load_reflection_region_b8_v16(src, buffer, b_rfct->_actual_load_num_L / 16);
    decx::bp::CPUK::_store_buf_to_dst_b8_v16(buffer + b_rfct->_L_v8_reflectL, dst, b_rfct->_actual_load_num_L / 16);
    reg._vuc.val[1] = vld1q_u8(src);
    vst1q_u8(dst + b_rfct->_left, reg._vuc.val[1]);

    for (uint32_t i = 1; i < _original_w_v8; ++i) {
        reg._vuc.val[0] = reg._vuc.val[1];
        reg._vuc.val[1] = vld1q_u8(src + i * 16);
        store._vuc = vqtbl2q_u8(reg._vuc, _idx);
        vst1q_u8(dst + b_rfct->_actual_load_num_L + i * 16 - 16, store._vuc);
    }
    reg._vuc.val[0] = reg._vuc.val[1];
    reg._vui.val[1] = veorq_u32(reg._vui.val[1], reg._vui.val[1]);
    store._vuc = vqtbl2q_u8(reg._vuc, _idx);
    vst1q_u8(dst + b_rfct->_actual_load_num_L + _original_w_v8 * 16 - 16, store._vuc);

    decx::bp::CPUK::_rev_load_reflection_region_b8_v16(src + (_original_w_v8) * 16 - b_rfct->_actual_load_num_R, 
                                                       buffer, 
                                                       b_rfct->_actual_load_num_R / 16);
    memcpy(dst + _actual_w_v1 + b_rfct->_left, 
           buffer + b_rfct->_rightmost_0num_src + 1, 
           b_rfct->_right * sizeof(uint8_t));
}



_THREAD_FUNCTION_ void
decx::bp::CPUK::_extend_H_reflect2D_b8(const uint8_t* __restrict   src,
                                      uint8_t* __restrict         buffer,
                                      uint8_t* __restrict         dst,
                                      const decx::bp::extend_reflect_exec_params* b_rfct,
                                      const uint32_t            Wsrc,
                                      const uint32_t            Wdst,
                                      const uint32_t            _actual_w_v1,
                                      const uint2               _original_dims_v16)
{
    const uint8x16_t _idx = decx::bp::e_rfct_exep_get_tbl_b8(b_rfct);

    decx::utils::simd::xmm256_reg reg;
    decx::utils::simd::xmm128_reg store;

    uint64_t dex_src = 0, dex_dst = 0;

    for (uint32_t i = 0; i < _original_dims_v16.y; ++i) 
    {
        dex_src = i * Wsrc;
        dex_dst = i * Wdst;
        _rev_load_reflection_region_b8_v16(src + dex_src, buffer, b_rfct->_actual_load_num_L / 16);
        _store_buf_to_dst_b8_v16(buffer + b_rfct->_L_v8_reflectL, dst + dex_dst, b_rfct->_actual_load_num_L / 16);
        reg._vuc.val[1] = vld1q_u8(src + dex_src);
        vst1q_u8(dst + dex_dst + b_rfct->_left, reg._vuc.val[1]);

        dex_src += 16;
        dex_dst += b_rfct->_actual_load_num_L;

        for (uint32_t j = 1; j < _original_dims_v16.x; ++j) {
            reg._vuc.val[0] = reg._vuc.val[1];
            reg._vf.val[1] = vld1q_u8(src + dex_src);
            store._vuc = vqtbl2q_u8(reg._vuc, _idx);
            vst1q_u8(dst + dex_dst, store._vuc);

            dex_src += 16;
            dex_dst += 16;
        }

        reg._vuc.val[0] = reg._vuc.val[1];
        reg._vui.val[1] = veorq_u32(reg._vui.val[1], reg._vui.val[1]);
        store._vuc = vqtbl2q_u8(reg._vuc, _idx);
        vst1q_u8(dst + dex_dst, store._vuc);

        decx::bp::CPUK::_rev_load_reflection_region_b8_v16(src + dex_src - b_rfct->_actual_load_num_R, 
                                                           buffer, 
                                                           b_rfct->_actual_load_num_R / 16);
        memcpy(dst + i * Wdst + b_rfct->_left + _actual_w_v1, 
               buffer + b_rfct->_rightmost_0num_src + 1, 
               b_rfct->_right * sizeof(uint8_t));
    }
}



_THREAD_FUNCTION_ void 
decx::bp::CPUK::_extend_V_reflect2D_m128(float* __restrict   dst,        // point A
                                        const uint32_t      _top, 
                                        const uint32_t      _bottom,
                                        const uint32_t      Hsrc, 
                                        const uint32_t      Wdst)       // in float
{
    float32x4_t recv;
    uint64_t _take = 0, _put = 0;

    for (uint32_t i = 0, _i = _top * 2; i < _top; ++i, --_i) 
    {
        _take = _i * Wdst;
        _put = i * Wdst;
        for (uint32_t j = 0; j < Wdst / 4; ++j) {
            recv = vld1q_f32(dst + _take);
            vst1q_f32(dst + _put, recv);
            _take += 4;
            _put += 4;
        }
    }

    for (uint32_t i = Hsrc + _top, _i = Hsrc + _top - 2; i < Hsrc + _top + _bottom; ++i, --_i) {
        _take = _i * Wdst;
        _put = i * Wdst;
        for (uint32_t j = 0; j < Wdst / 4; ++j) {
            recv = vld1q_f32(dst + _take);
            vst1q_f32(dst + _put, recv);
            _take += 4;
            _put += 4;
        }
    }
}



// -------------------------------------------------------- 16 bit ----------------------------------------------------------------


_THREAD_FUNCTION_ void decx::bp::CPUK::
_extend_reflect1D_b16(const uint16_t* __restrict                    src, 
                      uint16_t* __restrict                          buffer, 
                      uint16_t* __restrict                          dst,
                      const decx::bp::extend_reflect_exec_params*   b_rfct, 
                      const uint64_t                                _actual_w_v1, 
                      const uint64_t                                _original_w_v8)
{
    const uint8x16_t _idx = decx::bp::e_rfct_exep_get_tbl_b16(b_rfct);

    decx::utils::simd::xmm256_reg reg;
    decx::utils::simd::xmm128_reg store;

    decx::bp::CPUK::_rev_load_reflection_region_b16_v8(src, buffer, b_rfct->_actual_load_num_L / 8);
    decx::bp::CPUK::_store_buf_to_dst_b16_v8(buffer + b_rfct->_L_v8_reflectL, dst, b_rfct->_actual_load_num_L / 8);
    reg._vus.val[1] = vld1q_u16(src);
    vst1q_u16(dst + b_rfct->_left, reg._vus.val[1]);

    for (uint32_t i = 1; i < _original_w_v8; ++i) {
        reg._vus = vld1q_u16_x2(src + i * 8 - 8);
        store._vuc = vqtbl2q_u8(reg._vuc, _idx);
        vst1q_u16(dst + b_rfct->_actual_load_num_L + i * 8 - 8, store._vus);
    }

    reg._vus.val[0] = reg._vus.val[1];
    reg._vui.val[1] = veorq_u32(reg._vui.val[1], reg._vui.val[1]);
    store._vuc = vqtbl2q_u8(reg._vuc, _idx);
    vst1q_u16(dst + b_rfct->_actual_load_num_L + _original_w_v8 * 8 - 8, store._vus);

    decx::bp::CPUK::_rev_load_reflection_region_b16_v8(src + (_original_w_v8) * 8 - b_rfct->_actual_load_num_R, 
                                                       buffer, 
                                                       b_rfct->_actual_load_num_R / 8);
    memcpy(dst + _actual_w_v1 + b_rfct->_left, 
           buffer + b_rfct->_rightmost_0num_src + 1, 
           b_rfct->_right * sizeof(uint16_t));
}



_THREAD_FUNCTION_ void
decx::bp::CPUK::_extend_H_reflect2D_b16(const uint16_t* __restrict   src,
                                       uint16_t* __restrict         buffer,
                                       uint16_t* __restrict         dst,
                                       const decx::bp::extend_reflect_exec_params* b_rfct,
                                       const uint32_t            Wsrc,
                                       const uint32_t            Wdst,
                                       const uint32_t            _actual_w_v1,
                                       const uint2               _original_dims_v8)
{
    const uint8x16_t _idx = decx::bp::e_rfct_exep_get_tbl_b16(b_rfct);

    decx::utils::simd::xmm256_reg reg;
    decx::utils::simd::xmm128_reg store;

    uint64_t dex_src = 0, dex_dst = 0;

    for (uint32_t i = 0; i < _original_dims_v8.y; ++i) 
    {
        dex_src = i * Wsrc;
        dex_dst = i * Wdst;
        _rev_load_reflection_region_b16_v8(src + dex_src, buffer, b_rfct->_actual_load_num_L / 8);
        _store_buf_to_dst_b16_v8(buffer + b_rfct->_L_v8_reflectL, dst + dex_dst, b_rfct->_actual_load_num_L / 8);
        reg._vus.val[1] = vld1q_u16(src + dex_src);
        vst1q_u16(dst + dex_dst + b_rfct->_left, reg._vus.val[1]);

        dex_dst += b_rfct->_actual_load_num_L;

        for (uint32_t j = 1; j < _original_dims_v8.x; ++j) {
            reg._vus = vld1q_u16_x2(src + dex_src);
            store._vuc = vqtbl2q_u8(reg._vuc, _idx);
            vst1q_u16(dst + dex_dst, store._vus);

            dex_src += 8;
            dex_dst += 8;
        }

        reg._vus.val[0] = reg._vus.val[1];
        reg._vui.val[1] = veorq_u32(reg._vui.val[1], reg._vui.val[1]);
        store._vuc = vqtbl2q_u8(reg._vuc, _idx);
        vst1q_u16(dst + dex_dst, store._vus);

        dex_src += 8;

        decx::bp::CPUK::_rev_load_reflection_region_b16_v8(src + dex_src - b_rfct->_actual_load_num_R, 
                                                           buffer, 
                                                           b_rfct->_actual_load_num_R / 8);
        memcpy(dst + i * Wdst + b_rfct->_left + _actual_w_v1, 
               buffer + b_rfct->_rightmost_0num_src + 1, 
               b_rfct->_right * sizeof(uint16_t));
    }
}


// --------------------------------------------------------- 64 bit ----------------------------------------------------------------


_THREAD_FUNCTION_ void
decx::bp::CPUK::_extend_reflect1D_b64(const double* __restrict   src,
                                      double* __restrict         buffer,
                                      double* __restrict         dst,
                                      const decx::bp::extend_reflect_exec_params* b_rfct,
                                      const uint64_t             _actual_w_v1,
                                      const uint64_t             _original_w_v2)
{
    const uint8x16_t _idx = decx::bp::e_rfct_exep_get_tbl_b64(b_rfct);

    decx::utils::simd::xmm256_reg reg;
    decx::utils::simd::xmm128_reg store;

    decx::bp::CPUK::_rev_load_reflection_region_b64_v2(src, buffer, b_rfct->_actual_load_num_L / 2);
    decx::bp::CPUK::_store_buf_to_dst_b64_v2(buffer + b_rfct->_L_v8_reflectL, dst, b_rfct->_actual_load_num_L / 2);
    reg._vd.val[1] = vld1q_f64(src);
    vst1q_f64(dst + b_rfct->_left, reg._vd.val[1]);

    for (uint32_t i = 1; i < _original_w_v2; ++i) {
        reg._vd.val[0] = reg._vd.val[1];
        reg._vd.val[1] = vld1q_f64(src + i * 2);
        store._vuc = vqtbl2q_u8(reg._vuc, _idx);
        vst1q_f64(dst + b_rfct->_actual_load_num_L + i * 2 - 2, store._vd);
    }
    reg._vf.val[0] = reg._vf.val[1];
    reg._vui.val[1] = veorq_u32(reg._vui.val[1], reg._vui.val[1]);
    store._vuc = vqtbl2q_u8(reg._vuc, _idx);
    vst1q_f64(dst + b_rfct->_actual_load_num_L + _original_w_v2 * 2 - 2, store._vd);

    decx::bp::CPUK::_rev_load_reflection_region_b64_v2(src + (_original_w_v2) * 2 - b_rfct->_actual_load_num_R, 
                                                       buffer, 
                                                       b_rfct->_actual_load_num_R / 2);
    memcpy(dst + _actual_w_v1 + b_rfct->_left, 
           buffer + b_rfct->_rightmost_0num_src + 1, 
           b_rfct->_right * sizeof(double));
}



_THREAD_FUNCTION_ void
decx::bp::CPUK::_extend_H_reflect2D_b64(const double* __restrict   src,
                                        double* __restrict         buffer,
                                        double* __restrict         dst,
                                        const decx::bp::extend_reflect_exec_params* b_rfct,
                                        const uint32_t            Wsrc,
                                        const uint32_t            Wdst,
                                        const uint32_t            _actual_w_v1,
                                        const uint2               _original_dims_v2)
{
    const uint8x16_t _idx = decx::bp::e_rfct_exep_get_tbl_b64(b_rfct);

    decx::utils::simd::xmm256_reg reg;
    decx::utils::simd::xmm128_reg store;

    uint64_t dex_src = 0, dex_dst = 0;

    for (uint32_t i = 0; i < _original_dims_v2.y; ++i) 
    {
        dex_src = i * Wsrc;
        dex_dst = i * Wdst;
        _rev_load_reflection_region_b64_v2(src + dex_src, buffer, b_rfct->_actual_load_num_L / 2);
        _store_buf_to_dst_b64_v2(buffer + b_rfct->_L_v8_reflectL, dst + dex_dst, b_rfct->_actual_load_num_L / 2);
        reg._vd.val[1] = vld1q_f64(src + dex_src);
        vst1q_f64(dst + dex_dst + b_rfct->_left, reg._vd.val[1]);

        dex_src += 2;
        dex_dst += b_rfct->_actual_load_num_L;

        for (uint32_t j = 1; j < _original_dims_v2.x; ++j) {
            reg._vf.val[0] = reg._vf.val[1];
            reg._vd.val[1] = vld1q_f64(src + dex_src);
            store._vuc = vqtbl2q_u8(reg._vuc, _idx);
            vst1q_f64(dst + dex_dst, store._vd);

            dex_src += 2;
            dex_dst += 2;
        }

        reg._vf.val[0] = reg._vf.val[1];
        reg._vui.val[1] = veorq_u32(reg._vui.val[1], reg._vui.val[1]);
        store._vuc = vqtbl2q_u8(reg._vuc, _idx);
        vst1q_f64(dst + dex_dst, store._vd);

        decx::bp::CPUK::_rev_load_reflection_region_b64_v2(src + dex_src - b_rfct->_actual_load_num_R, 
                                                           buffer, 
                                                           b_rfct->_actual_load_num_R / 2);
        memcpy(dst + i * Wdst + b_rfct->_left + _actual_w_v1, 
               buffer + b_rfct->_rightmost_0num_src + 1, 
               b_rfct->_right * sizeof(double));
    }
}