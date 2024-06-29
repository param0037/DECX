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


#include "extend_reflect_exec_params.h"



namespace decx
{
    namespace bp {
        inline uint32_t _shufflevar_param_i32_gen_front(const uint32_t _index_v8, const uint32_t _L_v8)
        {
            return (((uint32_t)(_index_v8 < (8 - _L_v8))) * (_L_v8 + _index_v8));
        }


        inline uint32_t _shufflevar_param_i64_gen_front(const uint32_t _index_v8, const uint32_t _L_v8)
        {
            return (((uint32_t)(_index_v8 < (4 - _L_v8))) * (_L_v8 + _index_v8));
        }


        inline uint8_t _shufflevar_param_u8_gen_front(const uint8_t _index_v8, const uint8_t _L_v8)
        {
            return (((uint8_t)(_index_v8 < (16 - _L_v8))) * (_L_v8 + _index_v8));
        }


        inline uint32_t _shufflevar_param_i32_gen_back(const uint32_t _index_v8, const uint32_t _L_v8)
        {
            return (((uint32_t)(_index_v8 > (7 - _L_v8))) * (_index_v8 - (8 - _L_v8)));
        }

        inline uint32_t _shufflevar_param_i64_gen_back(const uint32_t _index_v8, const uint32_t _L_v8)
        {
            return (((uint32_t)(_index_v8 > (3 - _L_v8))) * (_index_v8 - (4 - _L_v8)));
        }


        inline uint8_t _shufflevar_param_u8_gen_back(const uint8_t _index_v8, const uint8_t _L_v8)
        {
            return (((uint8_t)(_index_v8 > (15 - _L_v8))) * (_index_v8 - (16 - _L_v8)));
        }
    }
}



void decx::bp::e_rfct_exep_gen_b32(decx::bp::extend_reflect_exec_params* _src, const uint32_t _left,
    const uint32_t _right, const size_t _actual_w_v1, const size_t _Wsrc_v8)
{
    _src->_left = _left;
    _src->_right = _right;
    _src->_actual_load_num_L = decx::utils::ceil<uint32_t>(_left + 1, 8) * 8;
    _src->_rightmost_0num_src = _Wsrc_v8 * 8 - _actual_w_v1;
    _src->_actual_load_num_R = decx::utils::ceil<uint32_t>(_right + _src->_rightmost_0num_src + 1, 8) * 8;
    uint32_t raw_L_v8_reflectL = 8 - ((_left + 1) % 8);
    _src->_L_v8_reflectL = raw_L_v8_reflectL == 8 ? 0 : raw_L_v8_reflectL;
    _src->_L_v8_L = 8 - (_left % 8);
}



void decx::bp::e_rfct_exep_gen_b8(decx::bp::extend_reflect_exec_params* _src, const uint32_t _left,
    const uint32_t _right, const size_t _actual_w_v1, const size_t _Wsrc_v8)
{
    _src->_left = _left;
    _src->_right = _right;
    _src->_actual_load_num_L = decx::utils::ceil<uint32_t>(_left + 1, 16) * 16;
    _src->_rightmost_0num_src = decx::utils::ceil<uint32_t>(_actual_w_v1, 16) * 16 - _actual_w_v1;
    _src->_actual_load_num_R = decx::utils::ceil<uint32_t>(_right + _src->_rightmost_0num_src + 1, 16) * 16;
    uint32_t raw_L_v16_reflectL = 16 - ((_left + 1) % 16);
    _src->_L_v8_reflectL = raw_L_v16_reflectL == 16 ? 0 : raw_L_v16_reflectL;
    _src->_L_v8_L = 16 - (_left % 16);
}



void decx::bp::e_rfct_exep_gen_b16(decx::bp::extend_reflect_exec_params* _src, const uint32_t _left,
    const uint32_t _right, const size_t _actual_w_v1, const size_t _Wsrc_v8)
{
    _src->_left = _left;
    _src->_right = _right;
    _src->_actual_load_num_L = decx::utils::ceil<uint32_t>(_left + 1, 8) * 8;
    _src->_rightmost_0num_src = decx::utils::ceil<uint32_t>(_actual_w_v1, 8) * 8 - _actual_w_v1;
    _src->_actual_load_num_R = decx::utils::ceil<uint32_t>(_right + _src->_rightmost_0num_src + 1, 8) * 8;
    uint32_t raw_L_v16_reflectL = 8 - ((_left + 1) % 8);
    _src->_L_v8_reflectL = raw_L_v16_reflectL == 8 ? 0 : raw_L_v16_reflectL;
    _src->_L_v8_L = 8 - (_left % 8);
}





void decx::bp::e_rfct_exep_gen_b64(decx::bp::extend_reflect_exec_params* _src, const uint32_t _left,
    const uint32_t _right, const size_t _actual_w_v1, const size_t _Wsrc_v8)
{
    _src->_left = _left;
    _src->_right = _right;
    _src->_actual_load_num_L = decx::utils::ceil<uint32_t>(_left + 1, 4) * 4;
    _src->_rightmost_0num_src = _Wsrc_v8 * 4 - _actual_w_v1;
    _src->_actual_load_num_R = decx::utils::ceil<uint32_t>(_right + _src->_rightmost_0num_src + 1, 4) * 4;
    uint32_t raw_L_v8_reflectL = 4 - ((_left + 1) % 4);
    _src->_L_v8_reflectL = raw_L_v8_reflectL == 4 ? 0 : raw_L_v8_reflectL;
    _src->_L_v8_L = 4 - (_left % 4);
}



uint32_t decx::bp::e_rfct_exep_get_buffer_len(const decx::bp::extend_reflect_exec_params* _src)
{
    return max(_src->_actual_load_num_L, _src->_actual_load_num_R);
}




__m256i decx::bp::e_rfct_exep_get_shufflevar_f_b32(const decx::bp::extend_reflect_exec_params* _src)
{
    __m256i f = _mm256_setr_epi32(decx::bp::_shufflevar_param_i32_gen_front(0, _src->_L_v8_L),
        decx::bp::_shufflevar_param_i32_gen_front(1, _src->_L_v8_L),
        decx::bp::_shufflevar_param_i32_gen_front(2, _src->_L_v8_L),
        decx::bp::_shufflevar_param_i32_gen_front(3, _src->_L_v8_L),
        decx::bp::_shufflevar_param_i32_gen_front(4, _src->_L_v8_L),
        decx::bp::_shufflevar_param_i32_gen_front(5, _src->_L_v8_L),
        decx::bp::_shufflevar_param_i32_gen_front(6, _src->_L_v8_L),
        decx::bp::_shufflevar_param_i32_gen_front(7, _src->_L_v8_L));

    return f;
}




__m256i decx::bp::e_rfct_exep_get_shufflevar_b_b32(const decx::bp::extend_reflect_exec_params* _src)
{
    __m256i b = _mm256_setr_epi32(decx::bp::_shufflevar_param_i32_gen_back(0, _src->_L_v8_L),
        decx::bp::_shufflevar_param_i32_gen_back(1, _src->_L_v8_L),
        decx::bp::_shufflevar_param_i32_gen_back(2, _src->_L_v8_L),
        decx::bp::_shufflevar_param_i32_gen_back(3, _src->_L_v8_L),
        decx::bp::_shufflevar_param_i32_gen_back(4, _src->_L_v8_L),
        decx::bp::_shufflevar_param_i32_gen_back(5, _src->_L_v8_L),
        decx::bp::_shufflevar_param_i32_gen_back(6, _src->_L_v8_L),
        decx::bp::_shufflevar_param_i32_gen_back(7, _src->_L_v8_L));

    return b;
}


__m256i decx::bp::e_rfct_exep_get_blend_b32(const decx::bp::extend_reflect_exec_params* _src)
{
    __m256i blend = _mm256_setzero_si256();
    for (int i = 8 - _src->_L_v8_L; i < 8; ++i) {
#ifdef _MSC_VER
        blend.m256i_i32[i] = 0xFFFFFFFFU;
#endif
#ifdef __GNUC__
        ((uint32_t*)&blend)[i] = 0xFFFFFFFFU;
#endif
    }

    return blend;
}




__m256i decx::bp::e_rfct_exep_get_shufflevar_f_b64(const decx::bp::extend_reflect_exec_params* _src)
{
    __m256i f = _mm256_setr_epi32(decx::bp::_shufflevar_param_i64_gen_front(0, _src->_L_v8_L) * 2,
        decx::bp::_shufflevar_param_i64_gen_front(0, _src->_L_v8_L) * 2 + 1,
        decx::bp::_shufflevar_param_i64_gen_front(1, _src->_L_v8_L) * 2,
        decx::bp::_shufflevar_param_i64_gen_front(1, _src->_L_v8_L) * 2 + 1,
        decx::bp::_shufflevar_param_i64_gen_front(2, _src->_L_v8_L) * 2,
        decx::bp::_shufflevar_param_i64_gen_front(2, _src->_L_v8_L) * 2 + 1,
        decx::bp::_shufflevar_param_i64_gen_front(3, _src->_L_v8_L) * 2,
        decx::bp::_shufflevar_param_i64_gen_front(3, _src->_L_v8_L) * 2 + 1);

    return f;
}




__m256i decx::bp::e_rfct_exep_get_shufflevar_b_b64(const decx::bp::extend_reflect_exec_params* _src)
{
    __m256i b = _mm256_setr_epi32(decx::bp::_shufflevar_param_i64_gen_back(0, _src->_L_v8_L) * 2,
        decx::bp::_shufflevar_param_i64_gen_back(0, _src->_L_v8_L) * 2 + 1,
        decx::bp::_shufflevar_param_i64_gen_back(1, _src->_L_v8_L) * 2,
        decx::bp::_shufflevar_param_i64_gen_back(1, _src->_L_v8_L) * 2 + 1,
        decx::bp::_shufflevar_param_i64_gen_back(2, _src->_L_v8_L) * 2,
        decx::bp::_shufflevar_param_i64_gen_back(2, _src->_L_v8_L) * 2 + 1,
        decx::bp::_shufflevar_param_i64_gen_back(3, _src->_L_v8_L) * 2,
        decx::bp::_shufflevar_param_i64_gen_back(3, _src->_L_v8_L) * 2 + 1);

    return b;
}


__m256i decx::bp::e_rfct_exep_get_blend_b64(const decx::bp::extend_reflect_exec_params* _src)
{
    __m256i blend = _mm256_setzero_si256();
    for (int i = 4 - _src->_L_v8_L; i < 4; ++i) {
#ifdef _MSC_VER
        blend.m256i_i64[i] = 0xFFFFFFFFFFFFFFFFU;
#endif
#ifdef __GNUC__
        ((uint64_t*)&blend)[i] = 0xFFFFFFFFFFFFFFFFU;
#endif
    }

    return blend;
}



__m128i decx::bp::e_rfct_exep_get_shufflevar_f_b8(const decx::bp::extend_reflect_exec_params* _src)
{
    __m128i f = _mm_setr_epi8(decx::bp::_shufflevar_param_u8_gen_front(0, _src->_L_v8_L),
        decx::bp::_shufflevar_param_u8_gen_front(1, _src->_L_v8_L),
        decx::bp::_shufflevar_param_u8_gen_front(2, _src->_L_v8_L),
        decx::bp::_shufflevar_param_u8_gen_front(3, _src->_L_v8_L),
        decx::bp::_shufflevar_param_u8_gen_front(4, _src->_L_v8_L),
        decx::bp::_shufflevar_param_u8_gen_front(5, _src->_L_v8_L),
        decx::bp::_shufflevar_param_u8_gen_front(6, _src->_L_v8_L),
        decx::bp::_shufflevar_param_u8_gen_front(7, _src->_L_v8_L),
        decx::bp::_shufflevar_param_u8_gen_front(8, _src->_L_v8_L),
        decx::bp::_shufflevar_param_u8_gen_front(9, _src->_L_v8_L),
        decx::bp::_shufflevar_param_u8_gen_front(10, _src->_L_v8_L),
        decx::bp::_shufflevar_param_u8_gen_front(11, _src->_L_v8_L),
        decx::bp::_shufflevar_param_u8_gen_front(12, _src->_L_v8_L),
        decx::bp::_shufflevar_param_u8_gen_front(13, _src->_L_v8_L),
        decx::bp::_shufflevar_param_u8_gen_front(14, _src->_L_v8_L),
        decx::bp::_shufflevar_param_u8_gen_front(15, _src->_L_v8_L));

    return f;
}




__m128i decx::bp::e_rfct_exep_get_shufflevar_f_b16(const decx::bp::extend_reflect_exec_params* _src)
{
    __m128i f = _mm_setr_epi8(decx::bp::_shufflevar_param_i32_gen_front(0, _src->_L_v8_L) * 2,
        decx::bp::_shufflevar_param_i32_gen_front(0, _src->_L_v8_L) * 2 + 1,
        decx::bp::_shufflevar_param_i32_gen_front(1, _src->_L_v8_L) * 2,
        decx::bp::_shufflevar_param_i32_gen_front(1, _src->_L_v8_L) * 2 + 1,
        decx::bp::_shufflevar_param_i32_gen_front(2, _src->_L_v8_L) * 2,
        decx::bp::_shufflevar_param_i32_gen_front(2, _src->_L_v8_L) * 2 + 1,
        decx::bp::_shufflevar_param_i32_gen_front(3, _src->_L_v8_L) * 2,
        decx::bp::_shufflevar_param_i32_gen_front(3, _src->_L_v8_L) * 2 + 1,
        decx::bp::_shufflevar_param_i32_gen_front(4, _src->_L_v8_L) * 2,
        decx::bp::_shufflevar_param_i32_gen_front(4, _src->_L_v8_L) * 2 + 1,
        decx::bp::_shufflevar_param_i32_gen_front(5, _src->_L_v8_L) * 2,
        decx::bp::_shufflevar_param_i32_gen_front(5, _src->_L_v8_L) * 2 + 1,
        decx::bp::_shufflevar_param_i32_gen_front(6, _src->_L_v8_L) * 2,
        decx::bp::_shufflevar_param_i32_gen_front(6, _src->_L_v8_L) * 2 + 1,
        decx::bp::_shufflevar_param_i32_gen_front(7, _src->_L_v8_L) * 2,
        decx::bp::_shufflevar_param_i32_gen_front(7, _src->_L_v8_L) * 2 + 1);

    return f;
}




__m128i decx::bp::e_rfct_exep_get_shufflevar_b_b8(const decx::bp::extend_reflect_exec_params* _src)
{
    __m128i f = _mm_setr_epi8(decx::bp::_shufflevar_param_u8_gen_back(0, _src->_L_v8_L),
        decx::bp::_shufflevar_param_u8_gen_back(1, _src->_L_v8_L),
        decx::bp::_shufflevar_param_u8_gen_back(2, _src->_L_v8_L),
        decx::bp::_shufflevar_param_u8_gen_back(3, _src->_L_v8_L),
        decx::bp::_shufflevar_param_u8_gen_back(4, _src->_L_v8_L),
        decx::bp::_shufflevar_param_u8_gen_back(5, _src->_L_v8_L),
        decx::bp::_shufflevar_param_u8_gen_back(6, _src->_L_v8_L),
        decx::bp::_shufflevar_param_u8_gen_back(7, _src->_L_v8_L),
        decx::bp::_shufflevar_param_u8_gen_back(8, _src->_L_v8_L),
        decx::bp::_shufflevar_param_u8_gen_back(9, _src->_L_v8_L),
        decx::bp::_shufflevar_param_u8_gen_back(10, _src->_L_v8_L),
        decx::bp::_shufflevar_param_u8_gen_back(11, _src->_L_v8_L),
        decx::bp::_shufflevar_param_u8_gen_back(12, _src->_L_v8_L),
        decx::bp::_shufflevar_param_u8_gen_back(13, _src->_L_v8_L),
        decx::bp::_shufflevar_param_u8_gen_back(14, _src->_L_v8_L),
        decx::bp::_shufflevar_param_u8_gen_back(15, _src->_L_v8_L));

    return f;
}



__m128i decx::bp::e_rfct_exep_get_shufflevar_b_b16(const decx::bp::extend_reflect_exec_params* _src)
{
    __m128i f = _mm_setr_epi8(decx::bp::_shufflevar_param_i32_gen_back(0, _src->_L_v8_L) * 2,
        decx::bp::_shufflevar_param_i32_gen_back(0, _src->_L_v8_L) * 2 + 1,
        decx::bp::_shufflevar_param_i32_gen_back(1, _src->_L_v8_L) * 2,
        decx::bp::_shufflevar_param_i32_gen_back(1, _src->_L_v8_L) * 2 + 1,
        decx::bp::_shufflevar_param_i32_gen_back(2, _src->_L_v8_L) * 2,
        decx::bp::_shufflevar_param_i32_gen_back(2, _src->_L_v8_L) * 2 + 1,
        decx::bp::_shufflevar_param_i32_gen_back(3, _src->_L_v8_L) * 2,
        decx::bp::_shufflevar_param_i32_gen_back(3, _src->_L_v8_L) * 2 + 1,
        decx::bp::_shufflevar_param_i32_gen_back(4, _src->_L_v8_L) * 2,
        decx::bp::_shufflevar_param_i32_gen_back(4, _src->_L_v8_L) * 2 + 1,
        decx::bp::_shufflevar_param_i32_gen_back(5, _src->_L_v8_L) * 2,
        decx::bp::_shufflevar_param_i32_gen_back(5, _src->_L_v8_L) * 2 + 1,
        decx::bp::_shufflevar_param_i32_gen_back(6, _src->_L_v8_L) * 2,
        decx::bp::_shufflevar_param_i32_gen_back(6, _src->_L_v8_L) * 2 + 1,
        decx::bp::_shufflevar_param_i32_gen_back(7, _src->_L_v8_L) * 2,
        decx::bp::_shufflevar_param_i32_gen_back(7, _src->_L_v8_L) * 2 + 1);

    return f;
}


__m128i decx::bp::e_rfct_exep_get_blend_b8(const decx::bp::extend_reflect_exec_params* _src)
{
    __m128i blend = _mm_setzero_si128();
    for (int i = 16 - _src->_L_v8_L; i < 16; ++i) {
#ifdef _MSC_VER
        blend.m128i_i8[i] = 0xFF;
#endif
#ifdef __GNUC__
        ((uint8_t*)&blend)[i] = 0xFF;
#endif
    }

    return blend;
}



__m128i decx::bp::e_rfct_exep_get_blend_b16(const decx::bp::extend_reflect_exec_params* _src)
{
    __m128i blend = _mm_setzero_si128();
    for (int i = 8 - _src->_L_v8_L; i < 8; ++i) {
#ifdef _MSC_VER
        blend.m128i_i8[i * 2] = 0xFF;
        blend.m128i_i8[i * 2 + 1] = 0xFF;
#endif
#ifdef __GNUC__
        ((uint8_t*)&blend)[i * 2] = 0xFF;
        ((uint8_t*)&blend)[i * 2 + 1] = 0xFF;
#endif
    }

    return blend;
}