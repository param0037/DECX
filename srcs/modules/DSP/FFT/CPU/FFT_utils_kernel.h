/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _CPU_FFT_UTILS_KERNEL_H_
#define _CPU_FFT_UTILS_KERNEL_H_


#include "../../../core/basic.h"
#include "../../../core/thread_management/thread_pool.h"
#include "../../../core/utils/fragment_arrangment.h"
#include "../../../classes/classes_util.h"
#include "../fft_utils.h"
#include "../../CPU_cpf32_avx.h"
#include "../../../classes/Matrix.h"


#ifdef _MSC_VER
#define _STORE_RES_R2C_1ST_ {   \
    dst[dex + dex_w] = _mm256_castps_pd(res).m256d_f64[0];      dex_w += procW;     \
    dst[dex + dex_w] = _mm256_castps_pd(res).m256d_f64[1];      dex_w += procW;     \
    dst[dex + dex_w] = _mm256_castps_pd(res).m256d_f64[2];      dex_w += procW;     \
    dst[dex + dex_w] = _mm256_castps_pd(res).m256d_f64[3];                          \
}
#endif
#ifdef __GNUC__
#define _STORE_RES_R2C_1ST_ {   \
    dst[dex + dex_w] = ((double*)&res)[0];      dex_w += procW;     \
    dst[dex + dex_w] = ((double*)&res)[1];      dex_w += procW;     \
    dst[dex + dex_w] = ((double*)&res)[2];      dex_w += procW;     \
    dst[dex + dex_w] = ((double*)&res)[3];      dex_w += procW;     \
}
#endif

#ifdef _MSC_VER
#define _STORE_RES_C2C_MID_ {   \
    dst[dex + dex_w] = res.m256d_f64[0];      dex_w += procW;     \
    dst[dex + dex_w] = res.m256d_f64[1];      dex_w += procW;     \
    dst[dex + dex_w] = res.m256d_f64[2];      dex_w += procW;     \
    dst[dex + dex_w] = res.m256d_f64[3];                          \
}
#endif
#ifdef __GNUC__
#define _STORE_RES_C2C_MID_ {   \
    dst[dex + dex_w] = ((double*)&res)[0];      dex_w += procW;     \
    dst[dex + dex_w] = ((double*)&res)[1];      dex_w += procW;     \
    dst[dex + dex_w] = ((double*)&res)[2];      dex_w += procW;     \
    dst[dex + dex_w] = ((double*)&res)[3];                          \
}
#endif


#define _STORE_RES_C2R_LAST_ {      \
    O_buffer = _mm_setr_ps(_mm256_castpd_ps(res).m256_f32[0],       \
                           _mm256_castpd_ps(res).m256_f32[2],       \
                           _mm256_castpd_ps(res).m256_f32[4],       \
                           _mm256_castpd_ps(res).m256_f32[6]);      \
    _mm_store_ps(dst + dex_w * procH + i * 4, _mm_div_ps(O_buffer, _mm_set1_ps((float)signal_W)));  \
}


namespace decx
{
    namespace signal 
    {
        namespace CPUK {
            /*
            * @param Wsrc : Width of src, in vec4
            * @param Wdst : Width of dst, in vec4
            * @param proc_dim : ~.x -> width; ~.y -> height (BOTH IN VEC4)
            */
            _THREAD_FUNCTION_ void
            _FFT2D_transpose_C(const double* __restrict src, double* __restrict dst,
                const uint Wsrc, const uint Wdst, const uint2 proc_dim, const uint _L4);


            _THREAD_FUNCTION_ void
            /*
            * @param proc_len : in vec4
            */
            _FFT1D_cpy_cvtcp_f32(const double* __restrict src, float* __restrict dst, const size_t proc_len);


            _THREAD_FUNCTION_ void
                _FFT2D_transpose_C_and_divide(const double* __restrict src, double* __restrict dst,
                    const uint Wsrc, const uint Wdst, const uint2 proc_dim, const float _denominator, const uint _L4);


            _THREAD_FUNCTION_ void
                _FFT2D_transpose_C2R_and_divide(const double* __restrict src, float* __restrict dst,
                    const uint Wsrc, const uint Wdst, const uint2 proc_dim, const float _denominator, const uint _L4);
        }
    }
}






#endif