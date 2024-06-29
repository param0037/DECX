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


#include "extend_constant.h"
#include "extend_reflect_exec_params.h"



void decx::bp::_extend_constant_b32_1D(const float* src, float* dst, const float _val,
    const uint32_t _left, const uint32_t _right, const size_t _length_src, const size_t _actual_Lsrc, de::DH* handle)
{
    decx::bp::extend_reflect_exec_params b_rfct;
    decx::bp::e_rfct_exep_gen_b32(&b_rfct, _left, _right, _actual_Lsrc, _length_src / 8);

    decx::bp::CPUK::_extend_constant1D_b32(src, dst, _val, &b_rfct, _actual_Lsrc, _length_src / 8);
}


void decx::bp::_extend_constant_b64_1D(const double* src, double* dst, const double _val, const uint32_t _left, const uint32_t _right,
    const size_t _length_src, const size_t _actual_Lsrc, de::DH* handle)
{
    decx::bp::extend_reflect_exec_params b_rfct;
    decx::bp::e_rfct_exep_gen_b64(&b_rfct, _left, _right, _actual_Lsrc, _length_src / 4);

    decx::bp::CPUK::_extend_constant1D_b64(src, dst, _val, &b_rfct, _actual_Lsrc, _length_src / 4);
}



void decx::bp::_extend_constant_b8_1D(const uint8_t* src, uint8_t* dst, const uint8_t val, const uint32_t _left, const uint32_t _right,
    const size_t _length_src, const size_t _actual_Lsrc, de::DH* handle)
{
    decx::bp::extend_reflect_exec_params b_rfct;
    decx::bp::e_rfct_exep_gen_b8(&b_rfct, _left, _right, _actual_Lsrc, _length_src / 16);

    decx::bp::CPUK::_extend_constant1D_b8(src, dst, val, &b_rfct, _actual_Lsrc, _length_src / 16);
}


void decx::bp::_extend_constant_b16_1D(const uint16_t* src, uint16_t* dst, const uint16_t val, const uint32_t _left, const uint32_t _right,
    const size_t _length_src, const size_t _actual_Lsrc, de::DH* handle)
{
    decx::bp::extend_reflect_exec_params b_rfct;
    decx::bp::e_rfct_exep_gen_b8(&b_rfct, _left, _right, _actual_Lsrc, _length_src / 8);

    decx::bp::CPUK::_extend_constant1D_b16(src, dst, val, &b_rfct, _actual_Lsrc, _length_src / 8);
}



void decx::bp::_extend_constant_b32_2D(const float* src, float* dst, const float _val, const uint4 _ext,
    const uint32_t Wsrc, const uint32_t Wdst, const uint32_t _actual_Wsrc, const uint32_t Hsrc, de::DH* handle)
{
    decx::bp::extend_reflect_exec_params b_rfct;
    decx::bp::e_rfct_exep_gen_b32(&b_rfct, _ext.x, _ext.y, _actual_Wsrc, Wsrc / 8);

    decx::bp::CPUK::_extend_H_constant2D_b32(src,
        decx::utils::ptr_shift_xy<float, float>(dst, _ext.z, 0, Wdst),
        _val,
        &b_rfct,
        Wsrc, Wdst, _actual_Wsrc, make_uint2(Wsrc / 8, Hsrc));

    const __m256 _v_val = _mm256_set1_ps(_val);
    decx::bp::CPUK::_extend_V_constant2D_m256(dst, _v_val, _ext.z, _ext.w, Hsrc, Wdst);
}



void decx::bp::_extend_constant_b64_2D(const double* src, double* dst, const double _val, const uint4 _ext,
    const uint32_t Wsrc, const uint32_t Wdst, const uint32_t _actual_Wsrc, const uint32_t Hsrc, de::DH* handle)
{
    decx::bp::extend_reflect_exec_params b_rfct;
    decx::bp::e_rfct_exep_gen_b64(&b_rfct, _ext.x, _ext.y, _actual_Wsrc, Wsrc / 4);

    decx::bp::CPUK::_extend_H_constant2D_b64(src, 
        decx::utils::ptr_shift_xy<double, double>(dst, _ext.z, 0, Wdst),
        _val,
        &b_rfct, Wsrc, Wdst, _actual_Wsrc, make_uint2(Wsrc / 4, Hsrc));

    const __m256d _v_val = _mm256_set1_pd(_val);
    decx::bp::CPUK::_extend_V_constant2D_m256((float*)dst, _mm256_castpd_ps(_v_val), _ext.z, _ext.w, Hsrc, Wdst * 2);
}



void decx::bp::_extend_constant_b8_2D(const uint8_t* src, uint8_t* dst, const uint8_t _val, const uint4 _ext,
    const uint32_t Wsrc, const uint32_t Wdst, const uint32_t _actual_Wsrc, const uint32_t Hsrc, de::DH* handle)
{
    decx::bp::extend_reflect_exec_params b_rfct;
    decx::bp::e_rfct_exep_gen_b8(&b_rfct, _ext.x, _ext.y, _actual_Wsrc, Wsrc / 16);

    decx::bp::CPUK::_extend_H_constant2D_b8(src,
        decx::utils::ptr_shift_xy<uint8_t, uint8_t>(dst, _ext.z, 0, Wdst),
        _val,
        &b_rfct, Wsrc, Wdst, _actual_Wsrc,
        make_uint2(decx::utils::ceil<uint32_t>(_actual_Wsrc, 16), Hsrc));

    const __m256i _v_val = _mm256_set1_epi8(_val);
    decx::bp::CPUK::_extend_V_constant2D_m256((float*)dst, _mm256_castsi256_ps(_v_val), _ext.z, _ext.w, Hsrc, Wdst / 4);
}


void decx::bp::_extend_constant_b16_2D(const uint16_t* src, uint16_t* dst, const uint16_t _val, const uint4 _ext,
    const uint32_t Wsrc, const uint32_t Wdst, const uint32_t _actual_Wsrc, const uint32_t Hsrc, de::DH* handle)
{
    decx::bp::extend_reflect_exec_params b_rfct;
    decx::bp::e_rfct_exep_gen_b16(&b_rfct, _ext.x, _ext.y, _actual_Wsrc, Wsrc / 8);

    decx::bp::CPUK::_extend_H_constant2D_b16(src,
        decx::utils::ptr_shift_xy<uint16_t, uint16_t>(dst, _ext.z, 0, Wdst),
        _val,
        &b_rfct, Wsrc, Wdst, _actual_Wsrc,
        make_uint2(decx::utils::ceil<uint32_t>(_actual_Wsrc, 8), Hsrc));

    const __m256i _v_val = _mm256_set1_epi16(_val);
    decx::bp::CPUK::_extend_V_constant2D_m256((float*)dst, _mm256_castsi256_ps(_v_val), _ext.z, _ext.w, Hsrc, Wdst / 2);
}
