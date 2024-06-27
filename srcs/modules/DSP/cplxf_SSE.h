/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _CPLXF_SSE_H_
#define _CPLXF_SSE_H_

#include "../core/basic.h"
#include "../core/thread_management/thread_pool.h"
#include "../classes/classes_util.h"


#define _expand_CPf32_MM128_(__real, __image) _mm_setr_ps(__real, __image,       \
                                                          __real, __image)       \



namespace decx
{
namespace dsp
{
namespace CPUK {
    inline _THREAD_CALL_
    __m128 _cp2_mul_cp2_fp32(const __m128 __x, const __m128 __y)
    {
        __m128 rr_ii = _mm_mul_ps(__x, __y);
        __m128 ri_ir = _mm_mul_ps(__x, _mm_permute_ps(__y, 0b10110001));
        // real, real | image, image
        rr_ii = _mm_permute_ps(rr_ii, 0b11011000);
        ri_ir = _mm_permute_ps(ri_ir, 0b11011000);
        __m128 res = _mm_unpacklo_ps(rr_ii, ri_ir);
        res = _mm_addsub_ps(res, _mm_unpackhi_ps(rr_ii, ri_ir));
        return res;
    }


    inline _THREAD_CALL_
    __m128 _cp2_fma_cp2_fp32(const __m128 __x, const __m128 __y, const __m128 __z) {
        __m128 res = decx::dsp::CPUK::_cp2_mul_cp2_fp32(__x, __y);
        return _mm_add_ps(res, __z);
    }
}
}
}


#endif
