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

#ifndef _HOUSEHOLDER_REFLECTOR_H_
#define _HOUSEHOLDER_REFLECTOR_H_

#include <basic.h>
#include "../../../../core/thread_management/thread_pool.h"


namespace decx
{
namespace blas{
    void householder_calc_v8_fp32(const float* __restrict p_col, float* __restrict p_V, const uint32_t proc_len_v1,
        const uint32_t local_col_id);


    namespace CPUK
    {
        _THREAD_CALL_ static uint32_t calc_proc_len_v(const uint32_t local_col_id, const uint8_t alignment, const uint32_t proc_len_v1);
    }
}
}


_THREAD_CALL_ static uint32_t 
decx::blas::CPUK::calc_proc_len_v(const uint32_t    local_col_id, 
                                  const uint8_t     alignment, 
                                  const uint32_t    proc_len_v1)
{
    uint32_t left = local_col_id % (uint32_t)alignment;
    uint32_t is_left = left == 0 ? 0 : 1;
    uint32_t post_length = proc_len_v1 - is_left;
    return decx::utils::ceil<uint32_t>(post_length, alignment) + is_left;
}

#endif