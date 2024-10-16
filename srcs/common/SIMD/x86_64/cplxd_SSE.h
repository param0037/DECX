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


#ifndef _CPLXD_SSE_H_
#define _CPLXD_SSE_H_

#include "../../basic.h"
#include "../../../modules/core/thread_management/thread_pool.h"
#include "../../Classes/classes_util.h"


namespace decx
{
namespace dsp
{
namespace CPUK {
    inline _THREAD_CALL_
    __m128d _cp_mul_cp_fp64(const __m128d __x, const __m128d __y)
    {
        __m128d rr_ii = _mm_mul_pd(__x, __y);
        __m128d ri_ir = _mm_mul_pd(__x, _mm_permute_pd(__y, 1));
        __m128d res = _mm_unpacklo_pd(rr_ii, ri_ir);
        res = _mm_addsub_pd(res, _mm_unpackhi_pd(rr_ii, ri_ir));
        return res;
    }


    inline _THREAD_CALL_
    __m128d _cp_fma_cp_fp64(const __m128d __x, const __m128d __y, const __m128d __z)
    {
        __m128d res = decx::dsp::CPUK::_cp_mul_cp_fp64(__x, __y);
        return _mm_add_pd(res, __z);
    }
}
}
}


#endif
