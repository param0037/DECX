/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/


#ifndef _LEFTOVERS_H_
#define _LEFTOVERS_H_

#include "../basic.h"
#include "decx_utils_functions.h"


namespace decx
{
    namespace utils
    {
        struct _left_8;
        struct _left_4;
        struct _left_16;
    }
}


struct decx::utils::_left_8
{
    size_t _bound;
    size_t _unit_bound;
    uchar _occupied_len;

#ifdef _DECX_CUDA_CODES_
    __host__ __device__ _left_8() {
        this->_bound = 0;
        this->_occupied_len = 0;
        this->_unit_bound = 0;
    }


    __host__ __device__ _left_8(const size_t bd, const uchar _occ) {
        this->_bound = bd;
        this->_occupied_len = _occ;
        this->_unit_bound = decx::utils::ceil<size_t>(bd, 8);
    }
#endif

#ifdef _DECX_CPU_CODES_
    _left_8() {
        this->_bound = 0;
        this->_occupied_len = 0;
        this->_unit_bound = 0;
    }


    _left_8(const size_t bd, const uchar _occ) {
        this->_bound = bd;
        this->_occupied_len = _occ;
        this->_unit_bound = decx::utils::ceil<size_t>(bd, 8);
    }

    inline __m256 get_filled_fragment_fp32(const float __x) {
        __m256 res = _mm256_set1_ps(0);
        for (uchar i = 0; i < this->_occupied_len; ++i) {
#ifdef Windows
            res.m256_f32[i] = __x;
#endif
#ifdef Linux
            ((float*)&res)[i] = __x;
#endif
        }

        return res;
    }


    inline __m256i get_filled_fragment_int(const int __x) {
        __m256i res = _mm256_set1_epi32(0);
        for (uchar i = 0; i < this->_occupied_len; ++i) {
#ifdef Windows
            res.m256i_i32[i] = __x;
#endif
#ifdef Linux
            ((int*)&res)[i] = __x;
#endif
        }

        return res;
    }
#endif
};



struct decx::utils::_left_4
{
    size_t _bound;
    uint _occupied_len;

#ifdef _DECX_CUDA_CODES_
    __host__ __device__ _left_4(const size_t bd, const uint _occ) {
        this->_bound = bd;
        this->_occupied_len = _occ;
    }

    __host__ __device__ _left_4() {
        this->_bound = 0;
        this->_occupied_len = 0;
    }

    __host__ __device__ __inline__ float4
        get_filled_fragment_fp32(const float __x) {
        float4 res = make_float4(0, 0, 0, 0);
        for (int i = 0; i < this->_occupied_len; ++i) {
            ((float*)&res)[i] = __x;
        }
        return res;
    }

    /*
    * get a [a, b, __x, __x]<float4> (assuming that the occupied length is 2)
    * Fill the blank with @param __x : the element filled in the blank
    */
    __host__ __device__ __inline__ void
        get_blank_filled_fragment_fp32(float4 *src, const float __x) {
        for (int i = 4 - this->_occupied_len; i < 4; ++i) {
            ((float*)src)[i] = __x;
        }
}
#endif

#ifdef _DECX_CPU_CODES_
    _left_4(const size_t bd, const uint _occ) {
        this->_bound = bd;
        this->_occupied_len = _occ;
    }
#endif
};



struct decx::utils::_left_16
{
    size_t _bound;
    uint _occupied_len;

#ifdef _DECX_CUDA_CODES_
    __host__ __device__ _left_16(const size_t bd, const uint _occ) {
        this->_bound = bd;
        this->_occupied_len = _occ;
    }
#endif

#ifdef _DECX_CPU_CODES_
    _left_16(const size_t bd, const uint _occ) {
        this->_bound = bd;
        this->_occupied_len = _occ;
    }
#endif
};



namespace decx
{
    namespace utils
    {
        static void _left_4_advisor(_left_4* _in, size_t req_len);


        static void _left_8_advisor(_left_8* _in, size_t req_len);
    }
}


static void decx::utils::_left_4_advisor(_left_4* _in, size_t req_len)
{
    _in->_bound = req_len;
    _in->_occupied_len = decx::utils::ceil<size_t>(req_len, 4) * 4 - req_len;
}



static void decx::utils::_left_8_advisor(_left_8* _in, size_t req_len)
{
    _in->_bound = req_len;
    _in->_occupied_len = decx::utils::ceil<size_t>(req_len, 8) * 8 - req_len;
}


#endif