/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#include "distributions_exec.h"
#include "../../../DSP/FFT/fft_utils.h"      // include PI


namespace decx
{
    namespace gen {
        namespace CPUK {
            _THREAD_CALL_ static float uniform_01_fp32(const uint32_t _precision);
        }
    }
}


_THREAD_CALL_ float decx::gen::CPUK::uniform_01_fp32(const uint32_t _precision)
{
    int raw_i32 = rand() % _precision;
    return (float)raw_i32 / (float)_precision;
}


_THREAD_FUNCTION_ void 
decx::gen::CPUK::_gaussian2D_fp32(float* __restrict     _target, 
                                  const float           expect, 
                                  const float           diveation,
                                  const uint2           proc_dims, 
                                  const uint32_t        Wsrc, 
                                  const float2          clip_range, 
                                  const __m256          blend_var,
                                  const uint32_t        resolution)
{
    srand(time(NULL));

    __m256 uniform_v1, uniform_v2;
    __m256 rand_res;

    const __m256 mean_v8 = _mm256_set1_ps(expect);
    const __m256 diveation_v8 = _mm256_set1_ps(diveation);
    const __m256 upper_range_v8 = _mm256_set1_ps(clip_range.y);
    const __m256 lower_range_v8 = _mm256_set1_ps(clip_range.x);
    __m256 crit_tmp;
    
    uint64_t dex = 0;
    for (int i = 0; i < proc_dims.y; ++i) {
        dex = i * Wsrc;
        for (int j = 0; j < proc_dims.x; ++j) 
        {
            uniform_v1 = _mm256_set_ps(uniform_01_fp32(resolution), uniform_01_fp32(resolution),
                uniform_01_fp32(resolution), uniform_01_fp32(resolution),
                uniform_01_fp32(resolution), uniform_01_fp32(resolution),
                uniform_01_fp32(resolution), uniform_01_fp32(resolution));

            uniform_v2 = _mm256_set_ps(uniform_01_fp32(resolution), uniform_01_fp32(resolution),
                uniform_01_fp32(resolution), uniform_01_fp32(resolution),
                uniform_01_fp32(resolution), uniform_01_fp32(resolution),
                uniform_01_fp32(resolution), uniform_01_fp32(resolution));

            rand_res = _mm256_mul_ps(_mm256_log_ps(uniform_v1), _mm256_set1_ps(-2));
            rand_res = _mm256_sqrt_ps(rand_res);
            uniform_v2 = _mm256_mul_ps(uniform_v2, _mm256_set1_ps(Two_Pi));
            uniform_v2 = _mm256_cos_ps(uniform_v2);
            rand_res = _mm256_mul_ps(rand_res, uniform_v2);

            rand_res = _mm256_fmadd_ps(rand_res, diveation_v8, mean_v8);

            // clip min
            crit_tmp = _mm256_cmp_ps(rand_res, lower_range_v8, _CMP_LT_OS);
            rand_res = _mm256_blendv_ps(rand_res, lower_range_v8, crit_tmp);
            // clip max
            crit_tmp = _mm256_cmp_ps(rand_res, upper_range_v8, _CMP_GT_OS);
            rand_res = _mm256_blendv_ps(rand_res, upper_range_v8, crit_tmp);
            
            if (j == proc_dims.x - 1) {
                rand_res = _mm256_blendv_ps(rand_res, _mm256_setzero_ps(), blend_var);
            }

            _mm256_store_ps(_target + dex, rand_res);
            dex += 8;
        }
    }
}