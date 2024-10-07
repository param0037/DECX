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

#ifndef _FILL_KERNELS_H_
#define _FILL_KERNELS_H_

#include "../../../basic.h"
#include "../../../../modules/core/thread_management/thread_pool.h"


namespace decx
{
namespace CPUK {
    _THREAD_FUNCTION_ void fill1D_constant_fp32(float* dst, const uint64_t proc_len_v, const float _constant);
    _THREAD_FUNCTION_ void fill1D_constant_int32(int32_t* dst, const uint64_t proc_len_v, const int32_t _constant);
    _THREAD_FUNCTION_ void fill1D_constant_fp64(double* dst, const uint64_t proc_len_v, const double _constant);

    _THREAD_FUNCTION_ void fill2D_constant_fp32(float* dst, const uint2 proc_dims_v1, const uint32_t pitch_v1, const float _constant);
    _THREAD_FUNCTION_ void fill2D_constant_int32(int32_t* dst, const uint2 proc_dims_v1, const uint32_t pitch_v1, const int32_t _constant);
    _THREAD_FUNCTION_ void fill2D_constant_fp64(double* dst, const uint2 proc_dims_v1, const uint32_t pitch_v1, const double _constant);

    template <typename _data_type>
    using fill1D_const_kernel = void(_data_type*, const uint64_t, const _data_type);

    template <typename _data_type>
    using fill2D_const_kernel = void(_data_type*, const uint2, const uint32_t, const _data_type);
}

namespace CPUK {
    _THREAD_FUNCTION_ void fill1D_rand_fp32(float* dst, const uint64_t proc_len_v, const float min, const float max);
    _THREAD_FUNCTION_ void fill1D_rand_int32(int32_t* dst, const uint64_t proc_len_v, const int32_t min, const int32_t max);
    _THREAD_FUNCTION_ void fill1D_rand_fp64(double* dst, const uint64_t proc_len_v, const double min, const double max);

    _THREAD_FUNCTION_ void fill2D_rand_fp32(float* dst, const uint2 proc_dims_v1, const uint32_t pitch_v1, const float min, const float max);
    _THREAD_FUNCTION_ void fill2D_rand_int32(int32_t* dst, const uint2 proc_dims_v1, const uint32_t pitch_v1, const int32_t min, const int32_t max);
    _THREAD_FUNCTION_ void fill2D_rand_fp64(double* dst, const uint2 proc_dims_v1, const uint32_t pitch_v1, const double min, const double max);

    template <typename _data_type>
    using fill1D_rand_kernel = void(_data_type*, const uint64_t);

    template <typename _data_type>
    using fill2D_rand_kernel = void(_data_type*, const uint2, const uint32_t);
}
}


#endif
