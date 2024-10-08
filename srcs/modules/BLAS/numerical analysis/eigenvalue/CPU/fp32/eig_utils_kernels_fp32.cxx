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

#include "../eig_utils_kernels.h"


template <bool _is_last>
_THREAD_FUNCTION_ static void decx::blas::CPUK::
extract_diag_fp32(const float* __restrict mat, 
                  float* __restrict diag, 
                  float* __restrict off_diag, 
                  const uint32_t proc_diag_len, 
                  const uint32_t pitchmat)
{
    uint64_t LDG_dex = 0;

    const uint32_t loop_time = _is_last ? proc_diag_len - 1 : proc_diag_len;

    for (int32_t i = 0; i < loop_time; ++i){
        diag[i] = mat[LDG_dex];
        off_diag[i] = mat[LDG_dex + 1];
        LDG_dex += (pitchmat + 1);
    }
    if (_is_last) {
        diag[proc_diag_len - 1] = mat[LDG_dex];
        off_diag[proc_diag_len] = 0;        // Manually set to zero for further security.
    }
}

template void decx::blas::CPUK::extract_diag_fp32<true>(const float* __restrict, 
    float* __restrict, float* __restrict, const uint32_t, const uint32_t);

template void decx::blas::CPUK::extract_diag_fp32<false>(const float* __restrict, 
    float* __restrict, float* __restrict, const uint32_t, const uint32_t);


_THREAD_FUNCTION_ void decx::blas::CPUK::
Gerschgorin_bound_fp32(const float* __restrict diag, 
                       const float* __restrict off_diag,
                       float* __restrict u, 
                       float* __restrict l, 
                       const uint32_t proc_len)
{
    uint32_t LDG_dex = 0;

    float _l = INFINITY, _u = -INFINITY;

    for (int32_t i = 0; i < proc_len; ++i){
        float r = off_diag[i] + off_diag[i + 1];
        float current_u = diag[i] + r;
        float current_l = diag[i] - r;

        if (current_l < _l) _l = current_l;
        if (current_u > _u) _u = current_u;
    }
    *u = _u;
    *l = _l;
}


_THREAD_FUNCTION_ void decx::blas::CPUK::
count_eigv_fp32(const float* __restrict diag, 
                const float* __restrict off_diag, 
                uint32_t* __restrict count, 
                const float x,
                const uint32_t N)
{
    uint32_t _count = 0;
    float d = 1;
    
    for (int32_t i = 0; i < N; ++i){
        d = diag[i] - x - off_diag[i] * off_diag[i] / d;
        if (d < 0) ++_count;
    }
    *count = _count;
}
