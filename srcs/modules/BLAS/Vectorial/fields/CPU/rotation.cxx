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


#include "rotation.h"



static inline _THREAD_CALL_ __m128
decx::fields::CPUK::rotcalc_internal_shuffle(decx::utils::simd::xmm128_reg* _px, decx::utils::simd::xmm128_reg* _py,
    decx::utils::simd::xmm128_reg* _pz)
{
    __m128 tmp1, tmp2;
    tmp1 = _mm_blend_ps(_mm_permute_ps(_py->_vf, _MM_SHUFFLE(2, 2, 2, 2)),
        _mm_permute_ps(_pz->_vf, _MM_SHUFFLE(0, 0, 0, 0)), 0b0010);
    tmp1 = _mm_blend_ps(tmp1, _mm_permute_ps(_px->_vf, _MM_SHUFFLE(1, 1, 1, 1)), 0b0100);

    tmp2 = _mm_blend_ps(_mm_permute_ps(_pz->_vf, _MM_SHUFFLE(1, 1, 1, 1)),
        _mm_permute_ps(_px->_vf, _MM_SHUFFLE(2, 2, 2, 2)), 0b0010);
    tmp2 = _mm_blend_ps(tmp2, _mm_permute_ps(_py->_vf, _MM_SHUFFLE(0, 0, 0, 0)), 0b0100);

    return _mm_sub_ps(tmp1, tmp2);
}




_THREAD_FUNCTION_ void
decx::fields::CPUK::rotation_field3D_fp32(const float* __restrict     src,
                                          float* __restrict           dst,
                                          const uint3                 _proc_dims,
                                          const uint                  dp_x_wp,
                                          const uint                  d_pitch)
{
    size_t loc_dex = 0;
    __m128 _reg, tmp1, tmp2;
    decx::utils::simd::xmm128_reg _pFX_px, _pFY_py, _pFZ_pz;

    for (int i = 0; i < _proc_dims.x; ++i) 
    {
        for (int j = 0; j < _proc_dims.y; ++j) 
        {
            loc_dex = i * dp_x_wp + j * d_pitch;
            for (int k = 0; k < _proc_dims.z; ++k) 
            {
                _reg = _mm_load_ps(src + loc_dex);
                _pFX_px._vf = (i == _proc_dims.x - 1) ? _reg : _mm_sub_ps(_reg, _mm_load_ps(src + loc_dex + dp_x_wp));
                _pFY_py._vf = (j == _proc_dims.y - 1) ? _reg : _mm_sub_ps(_reg, _mm_load_ps(src + loc_dex + d_pitch));
                _pFZ_pz._vf = (k < _proc_dims.z - 1) ? _reg : _mm_sub_ps(_reg, _mm_load_ps(src + loc_dex + 4));

                tmp1 = _mm_blend_ps(_mm_permute_ps(_pFY_py._vf, _MM_SHUFFLE(2, 2, 2, 2)),
                    _mm_permute_ps(_pFZ_pz._vf, _MM_SHUFFLE(0, 0, 0, 0)), 0b0010);
                tmp1 = _mm_blend_ps(tmp1, _mm_permute_ps(_pFX_px._vf, _MM_SHUFFLE(1, 1, 1, 1)), 0b0100);

                tmp2 = _mm_blend_ps(_mm_permute_ps(_pFZ_pz._vf, _MM_SHUFFLE(1, 1, 1, 1)),
                    _mm_permute_ps(_pFX_px._vf, _MM_SHUFFLE(2, 2, 2, 2)), 0b0010);
                tmp2 = _mm_blend_ps(tmp2, _mm_permute_ps(_pFY_py._vf, _MM_SHUFFLE(0, 0, 0, 0)), 0b0100);

                _mm_store_ps(dst + loc_dex, _reg);
                loc_dex += 4;
            }
        }
    }
}