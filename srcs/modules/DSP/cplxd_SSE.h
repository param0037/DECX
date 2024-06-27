/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _CPLXD_SSE_H_
#define _CPLXD_SSE_H_

#include "../core/basic.h"
#include "../core/thread_management/thread_pool.h"
#include "../classes/classes_util.h"


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
