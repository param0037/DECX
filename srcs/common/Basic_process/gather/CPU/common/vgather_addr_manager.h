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

#ifndef _VGATHER_ADDR_MGR_H_
#define _VGATHER_ADDR_MGR_H_

#include "../../../../basic.h"
#include "../../../../../modules/core/thread_management/thread_pool.h"
#include "../../../../SIMD/x86_64/simd_fast_math_avx2.h"

/**
 * 0 -------------------> X
 * |  _____________
 * | |      |      |
 * | | base |  1   |
 * | |______|______|
 * | |      |      |
 * | |  2   |  3   |
 * | |______|______|
 * |
 * v
 * Y 
*/

namespace decx
{
namespace CPUK
{
    static __m256i g_permute_mapXY_v8 = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);

    class VGT_addr_mgr;
}
}

class decx::CPUK::VGT_addr_mgr
{
private:
    decx::utils::simd::
    xmm256_reg _Wsrc_v8,
                _Hsrc_v8,
                _offset_flatten_base_v8,
                _dist_down_X_v8,
                _dist_down_Y_v8,
                _offset_flat_within_v8;

public:
    _THREAD_CALL_ void plan(__m256 lane1, __m256 lane2)
    {
        __m256 tmp_lane1 = _mm256_permutevar8x32_ps(lane1, decx::CPUK::g_permute_mapXY_v8);
        __m256 tmp_lane2 = _mm256_permutevar8x32_ps(lane2, decx::CPUK::g_permute_mapXY_v8);

        __m256 _X_v8_fp32 = _mm256_permute2f128_ps(tmp_lane1, tmp_lane2, 0x20);
        __m256 _Y_v8_fp32 = _mm256_permute2f128_ps(tmp_lane1, tmp_lane2, 0x31);

        decx::utils::simd::xmm256_reg _X_v8, _Y_v8;

        _X_v8._vf = _mm256_round_ps(_X_v8_fp32, _MM_FROUND_FLOOR);
        this->_dist_down_X_v8._vf = _mm256_sub_ps(_X_v8_fp32, _X_v8._vf);
        _X_v8._vi = _mm256_cvtps_epi32(_X_v8._vf);

        // Boundary check, clamp all overflow address to 0
        // Target : X(base) | 0 <= X && X+1 < boundary
        // This is an inverse mask
        this->_offset_flat_within_v8._vi = _mm256_cmpgt_epi32(_mm256_setzero_si256(), _X_v8._vi);
        // Inverse map, if X >= boundary_X, X + 1 must larger than that as well
        this->_offset_flat_within_v8._vi = _mm256_or_si256(_mm256_cmpgt_epi32(_X_v8._vi, this->_Wsrc_v8._vi),
                                                        this->_offset_flat_within_v8._vi);

        _Y_v8._vf = _mm256_round_ps(_Y_v8_fp32, _MM_FROUND_FLOOR);
        this->_dist_down_Y_v8._vf = _mm256_sub_ps(_Y_v8_fp32, _Y_v8._vf);
        _Y_v8._vi = _mm256_cvtps_epi32(_Y_v8._vf);

        this->_offset_flat_within_v8._vi = _mm256_or_si256(this->_offset_flat_within_v8._vi,
                                                        _mm256_cmpgt_epi32(_mm256_setzero_si256(), _Y_v8._vi));
        // Inverse map, if X >= boundary_X, X + 1 must larger than that as well
        this->_offset_flat_within_v8._vi = _mm256_or_si256(_mm256_cmpgt_epi32(_Y_v8._vi, this->_Hsrc_v8._vi),
                                                        this->_offset_flat_within_v8._vi);


        this->_offset_flatten_base_v8._vi = _mm256_add_epi32(_mm256_mullo_epi32(_Y_v8._vi, this->_Wsrc_v8._vi), _X_v8._vi);
        this->_offset_flatten_base_v8._vi = _mm256_andnot_si256(this->_offset_flat_within_v8._vi, this->_offset_flatten_base_v8._vi);
    }


    _THREAD_CALL_ __m256 get_current_inbound_fp32() const{
        return this->_offset_flat_within_v8._vf;
    }


    _THREAD_CALL_ void set_src_dims(const uint2 src_dims_v1){
        this->_Wsrc_v8._vf = _mm256_broadcast_ss((float*)(&src_dims_v1.x));
        this->_Hsrc_v8._vf = _mm256_broadcast_ss((float*)(&src_dims_v1.y));
    }


    _THREAD_CALL_ __m256i get_addr0() const{
        return this->_offset_flatten_base_v8._vi;
    }
    _THREAD_CALL_ __m256i get_addr1() const{
        return _mm256_add_epi32(this->_offset_flatten_base_v8._vi, _mm256_set1_epi32(1));
    }
    _THREAD_CALL_ __m256i get_addr2() const{
        return _mm256_add_epi32(this->_offset_flatten_base_v8._vi, this->_Wsrc_v8._vi);
    }
    _THREAD_CALL_ __m256i get_addr3() const{
        return _mm256_add_epi32(this->_offset_flatten_base_v8._vi, 
               _mm256_add_epi32(this->_Wsrc_v8._vi, _mm256_set1_epi32(1)));
    }

    _THREAD_CALL_ __m256 get_dist_down_X() const{
        return this->_dist_down_X_v8._vf;
    }
    _THREAD_CALL_ __m256 get_dist_down_Y() const{
        return this->_dist_down_Y_v8._vf;
    }
    _THREAD_CALL_ __m256 get_dist_up_X() const{
        return _mm256_sub_ps(_mm256_set1_ps(1.f), this->_dist_down_X_v8._vf);
    }
    _THREAD_CALL_ __m256 get_dist_up_Y() const{
        return _mm256_sub_ps(_mm256_set1_ps(1.f), this->_dist_down_Y_v8._vf);
    }
};

#endif
