/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/

#include "FFT1D_radix_4_kernel.h"


_THREAD_FUNCTION_
void decx::signal::CPUK::_FFT1D_R4_fp32_R2C_first_ST(const float* __restrict src, 
                                               double* __restrict dst, 
                                               const size_t signal_length,
                                               const uint2 b_op_dex_range)
{
    float recv[4];
    de::CPf res[4];

    size_t dex = 0;

    const size_t total_Bcalc_num = signal_length / 4;

    for (int i = b_op_dex_range.x; i < b_op_dex_range.y; ++i) {
        dex = i;
        recv[0] = src[dex];
        recv[1] = src[dex + total_Bcalc_num];
        recv[2] = src[dex + total_Bcalc_num * 2];
        recv[3] = src[dex + total_Bcalc_num * 3];

        res[0].real = recv[0] + recv[1] + recv[2] + recv[3];
        res[0].image = 0;

        res[1].real = recv[0] - recv[2];
        res[1].image = recv[1] - recv[3];

        res[2].real = recv[0] - recv[1] + recv[2] - recv[3];
        res[2].image = 0;

        res[3].real = recv[0] - recv[2];
        res[3].image = recv[3] - recv[1];

        dex = i * 4;
        dst[dex] = *((double*)&res[0]);
        dst[dex + 1] = *((double*)&res[1]);
        dst[dex + 2] = *((double*)&res[2]);
        dst[dex + 3] = *((double*)&res[3]);
    }
}



_THREAD_FUNCTION_
void decx::signal::CPUK::_IFFT1D_R4_fp32_C2C_first_ST(const double* __restrict src, 
                                                double* __restrict dst, 
                                                const size_t signal_length,
                                                const uint2 b_op_dex_range)
{
    de::CPf recv[4], tmp;
    de::CPf res[4];

    size_t dex = 0;

    const size_t total_Bcalc_num = signal_length / 4;

    for (int i = b_op_dex_range.x; i < b_op_dex_range.y; ++i) {
        dex = i;
        *((double*)&recv[0]) = src[dex];
        *((double*)&recv[1]) = src[dex + total_Bcalc_num];
        *((double*)&recv[2]) = src[dex + total_Bcalc_num * 2];
        *((double*)&recv[3]) = src[dex + total_Bcalc_num * 3];

        *((__m256*)&recv[0]) = _mm256_mul_ps(*((__m256*)&recv[0]), _mm256_setr_ps(1, -1, 1, -1, 1, -1, 1,-1));
        *((__m256*)&recv[0]) = _mm256_div_ps(*((__m256*)&recv[0]), _mm256_set1_ps((float)signal_length));

        res[0].real = recv[0].real + recv[1].real + recv[2].real + recv[3].real;
        res[0].image = recv[0].image + recv[1].image + recv[2].image + recv[3].image;

        tmp = recv[0];
        tmp.real -= recv[1].image;
        tmp.image += recv[1].real;
        tmp.real -= recv[2].real;
        tmp.image -= recv[2].image;
        res[1].real = tmp.real + recv[3].image;
        res[1].image = tmp.image - recv[3].real;

        tmp = recv[0];
        tmp.real -= recv[1].real;
        tmp.image -= recv[1].image;
        tmp.real += recv[2].real;
        tmp.image += recv[2].image;
        res[2].real = tmp.real - recv[3].real;
        res[2].image = tmp.image - recv[3].image;

        tmp = recv[0];
        tmp.real += recv[1].image;
        tmp.image -= recv[1].real;
        tmp.real -= recv[2].real;
        tmp.image -= recv[2].image;
        res[3].real = tmp.real - recv[3].image;
        res[3].image = tmp.image + recv[3].real;

        dex = i * 4;
        dst[dex] = *((double*)&res[0]);
        dst[dex + 1] = *((double*)&res[1]);
        dst[dex + 2] = *((double*)&res[2]);
        dst[dex + 3] = *((double*)&res[3]);
    }
}



_THREAD_FUNCTION_
void decx::signal::CPUK::_FFT1D_R4_fp32_C2C_first_ST(const double* __restrict src, 
                                                double* __restrict dst, 
                                                const size_t signal_length,
                                                const uint2 b_op_dex_range)
{
    de::CPf recv[4], tmp;
    de::CPf res[4];

    size_t dex = 0;

    const size_t total_Bcalc_num = signal_length / 4;

    for (int i = b_op_dex_range.x; i < b_op_dex_range.y; ++i) {
        dex = i;
        *((double*)&recv[0]) = src[dex];
        *((double*)&recv[1]) = src[dex + total_Bcalc_num];
        *((double*)&recv[2]) = src[dex + total_Bcalc_num * 2];
        *((double*)&recv[3]) = src[dex + total_Bcalc_num * 3];

        res[0].real = recv[0].real + recv[1].real + recv[2].real + recv[3].real;
        res[0].image = recv[0].image + recv[1].image + recv[2].image + recv[3].image;

        tmp = recv[0];
        tmp.real -= recv[1].image;
        tmp.image += recv[1].real;
        tmp.real -= recv[2].real;
        tmp.image -= recv[2].image;
        res[1].real = tmp.real + recv[3].image;
        res[1].image = tmp.image - recv[3].real;

        tmp = recv[0];
        tmp.real -= recv[1].real;
        tmp.image -= recv[1].image;
        tmp.real += recv[2].real;
        tmp.image += recv[2].image;
        res[2].real = tmp.real - recv[3].real;
        res[2].image = tmp.image - recv[3].image;

        tmp = recv[0];
        tmp.real += recv[1].image;
        tmp.image -= recv[1].real;
        tmp.real -= recv[2].real;
        tmp.image -= recv[2].image;
        res[3].real = tmp.real - recv[3].image;
        res[3].image = tmp.image + recv[3].real;

        dex = i * 4;
        dst[dex] = *((double*)&res[0]);
        dst[dex + 1] = *((double*)&res[1]);
        dst[dex + 2] = *((double*)&res[2]);
        dst[dex + 3] = *((double*)&res[3]);
    }
}



_THREAD_FUNCTION_
void decx::signal::CPUK::_FFT1D_R4_fp32_R2C_first_ST_vec4(const float* __restrict src,
                                                    double* __restrict dst,
                                                    const size_t signal_length,
                                                    const uint2 b_op_dex_range)
{
    __m128 recv[4], tmp[4];
    __m256 res;
    size_t dex = 0;
    const size_t half_signal_len = signal_length / 4;

    for (uint i = b_op_dex_range.x; i < b_op_dex_range.y; ++i) {
        recv[0] = _mm_load_ps(src + (i << 2));
        recv[1] = _mm_load_ps(src + (i << 2) + half_signal_len);
        recv[2] = _mm_load_ps(src + (i << 2) + (half_signal_len << 1));
        recv[3] = _mm_load_ps(src + (i << 2) + half_signal_len * 3);

        // res[0].real
        tmp[0] = _mm_add_ps(recv[0], recv[1]);
        tmp[0] = _mm_add_ps(_mm_add_ps(recv[2], recv[3]), tmp[0]);
        // res[1].real
        tmp[1] = _mm_sub_ps(recv[0], recv[2]);
        // res[1].image
        tmp[2] = _mm_sub_ps(recv[1], recv[3]);
        // res[2].real
        tmp[3] = _mm_sub_ps(recv[0], recv[1]);
        tmp[3] = _mm_add_ps(_mm_sub_ps(recv[2], recv[3]), tmp[3]);
#ifdef _MSC_VER
        dex = i * 16;
        res = _mm256_setr_ps(tmp[0].m128_f32[0], 0, 
                             tmp[1].m128_f32[0], tmp[2].m128_f32[0],
                             tmp[3].m128_f32[0], 0, 
                             tmp[1].m128_f32[0], -tmp[2].m128_f32[0]);
        _mm256_store_pd(dst + dex, _mm256_castps_pd(res));

        res = _mm256_setr_ps(tmp[0].m128_f32[1], 0, 
                             tmp[1].m128_f32[1], tmp[2].m128_f32[1],
                             tmp[3].m128_f32[1], 0, 
                             tmp[1].m128_f32[1], -tmp[2].m128_f32[1]);
        _mm256_store_pd(dst + dex + 4, _mm256_castps_pd(res));

        res = _mm256_setr_ps(tmp[0].m128_f32[2], 0, 
                             tmp[1].m128_f32[2], tmp[2].m128_f32[2],
                             tmp[3].m128_f32[2], 0, 
                             tmp[1].m128_f32[2], -tmp[2].m128_f32[2]);
        _mm256_store_pd(dst + dex + 8, _mm256_castps_pd(res));

        res = _mm256_setr_ps(tmp[0].m128_f32[3], 0, 
                             tmp[1].m128_f32[3], tmp[2].m128_f32[3],
                             tmp[3].m128_f32[3], 0, 
                             tmp[1].m128_f32[3], -tmp[2].m128_f32[3]);
        _mm256_store_pd(dst + dex + 12, _mm256_castps_pd(res));
#endif

#ifdef __GNUC__
        res = _mm256_setr_ps(((float*)&tmp[0])[0], 0, 
                             ((float*)&tmp[1])[0], ((float*)&tmp[2])[0],
                             ((float*)&tmp[3])[0], 0, 
                             ((float*)&tmp[1])[0], -((float*)&tmp[2])[0]);
        _mm256_store_pd(dst + dex, _mm256_castps_pd(res));

        res = _mm256_setr_ps(((float*)&tmp[0])[1], 0, 
                             ((float*)&tmp[1])[1], ((float*)&tmp[2])[1],
                             ((float*)&tmp[3])[1], 0, 
                             ((float*)&tmp[1])[1], -((float*)&tmp[2])[1]);
        _mm256_store_pd(dst + dex + 4, _mm256_castps_pd(res));

        res = _mm256_setr_ps(((float*)&tmp[0])[2], 0, 
                             ((float*)&tmp[1])[2], ((float*)&tmp[2])[2],
                             ((float*)&tmp[3])[2], 0, 
                             ((float*)&tmp[1])[2], -((float*)&tmp[2])[2]);
        _mm256_store_pd(dst + dex + 8, _mm256_castps_pd(res));

        res = _mm256_setr_ps(((float*)&tmp[0])[3], 0, 
                             ((float*)&tmp[1])[3], ((float*)&tmp[2])[3],
                             ((float*)&tmp[3])[3], 0, 
                             ((float*)&tmp[1])[3], -((float*)&tmp[2])[3]);
        _mm256_store_pd(dst + dex + 12, _mm256_castps_pd(res));
#endif
    }
}



_THREAD_FUNCTION_
void decx::signal::CPUK::_IFFT1D_R4_fp32_C2C_first_ST_vec4(const double* __restrict src,
                                                    double* __restrict dst,
                                                    const size_t signal_length,
                                                    const uint2 b_op_dex_range)
{
    __m256 recv[4], tmp[4];
    __m256 res;
    size_t dex = 0;
    const size_t half_signal_len = signal_length / 4;

    for (uint i = b_op_dex_range.x; i < b_op_dex_range.y; ++i) {
        recv[0] = _mm256_castpd_ps(_mm256_load_pd(src + (i << 2)));
        recv[1] = _mm256_castpd_ps(_mm256_load_pd(src + (i << 2) + half_signal_len));
        recv[2] = _mm256_castpd_ps(_mm256_load_pd(src + (i << 2) + (half_signal_len << 1)));
        recv[3] = _mm256_castpd_ps(_mm256_load_pd(src + (i << 2) + half_signal_len * 3));

        recv[0] = _mm256_div_ps(recv[0], _mm256_set1_ps((float)signal_length));
        recv[1] = _mm256_div_ps(recv[1], _mm256_set1_ps((float)signal_length));
        recv[2] = _mm256_div_ps(recv[2], _mm256_set1_ps((float)signal_length));
        recv[3] = _mm256_div_ps(recv[3], _mm256_set1_ps((float)signal_length));

        recv[0] = _mm256_mul_ps(recv[0], _mm256_setr_ps(1, -1, 1, -1, 1, -1, 1, -1));
        recv[1] = _mm256_mul_ps(recv[1], _mm256_setr_ps(1, -1, 1, -1, 1, -1, 1, -1));
        recv[2] = _mm256_mul_ps(recv[2], _mm256_setr_ps(1, -1, 1, -1, 1, -1, 1, -1));
        recv[3] = _mm256_mul_ps(recv[3], _mm256_setr_ps(1, -1, 1, -1, 1, -1, 1, -1));

        // res 0
        tmp[0] = _mm256_add_ps(recv[0], recv[1]);
        tmp[0] = _mm256_add_ps(tmp[0], _mm256_add_ps(recv[2], recv[3]));
        // res 1
        tmp[1] = _mm256_addsub_ps(recv[0], _mm256_permute_ps(recv[1], 0b10110001));
        tmp[1] = _mm256_sub_ps(tmp[1], recv[2]);
        tmp[1] = _mm256_permute_ps(_mm256_addsub_ps(_mm256_permute_ps(tmp[1], 0b10110001), recv[3]), 0b10110001);
        // res 2
        tmp[2] = _mm256_sub_ps(recv[0], recv[1]);
        tmp[2] = _mm256_add_ps(tmp[2], recv[2]);
        tmp[2] = _mm256_sub_ps(tmp[2], recv[3]);

        tmp[3] = _mm256_permute_ps(_mm256_addsub_ps(_mm256_permute_ps(recv[0], 0b10110001), recv[1]), 0b10110001);
        tmp[3] = _mm256_sub_ps(tmp[3], recv[2]);
        tmp[3] = _mm256_addsub_ps(tmp[3], _mm256_permute_ps(recv[3], 0b10110001));

#ifdef _MSC_VER
        dex = i * 16;
        res = _mm256_setr_ps(tmp[0].m256_f32[0], tmp[0].m256_f32[1],
                             tmp[1].m256_f32[0], tmp[1].m256_f32[1],
                             tmp[2].m256_f32[0], tmp[2].m256_f32[1],
                             tmp[3].m256_f32[0], tmp[3].m256_f32[1]);
        _mm256_store_pd(dst + dex, _mm256_castps_pd(res));

        res = _mm256_setr_ps(tmp[0].m256_f32[2], tmp[0].m256_f32[3],
                             tmp[1].m256_f32[2], tmp[1].m256_f32[3],
                             tmp[2].m256_f32[2], tmp[2].m256_f32[3],
                             tmp[3].m256_f32[2], tmp[3].m256_f32[3]);
        _mm256_store_pd(dst + dex + 4, _mm256_castps_pd(res));

        res = _mm256_setr_ps(tmp[0].m256_f32[4], tmp[0].m256_f32[5],
                             tmp[1].m256_f32[4], tmp[1].m256_f32[5],
                             tmp[2].m256_f32[4], tmp[2].m256_f32[5],
                             tmp[3].m256_f32[4], tmp[3].m256_f32[5]);
        _mm256_store_pd(dst + dex + 8, _mm256_castps_pd(res));

        res = _mm256_setr_ps(tmp[0].m256_f32[6], tmp[0].m256_f32[7],
                             tmp[1].m256_f32[6], tmp[1].m256_f32[7],
                             tmp[2].m256_f32[6], tmp[2].m256_f32[7],
                             tmp[3].m256_f32[6], tmp[3].m256_f32[7]);
        _mm256_store_pd(dst + dex + 12, _mm256_castps_pd(res));
#endif

#ifdef __GNUC__
        res = _mm256_setr_ps(((float*)&tmp[0])[0], 0, 
                             ((float*)&tmp[1])[0], ((float*)&tmp[2])[0],
                             ((float*)&tmp[3])[0], 0, 
                             ((float*)&tmp[1])[0], -((float*)&tmp[2])[0]);
        _mm256_store_pd(dst + dex, _mm256_castps_pd(res));

        res = _mm256_setr_ps(((float*)&tmp[0])[1], 0, 
                             ((float*)&tmp[1])[1], ((float*)&tmp[2])[1],
                             ((float*)&tmp[3])[1], 0, 
                             ((float*)&tmp[1])[1], -((float*)&tmp[2])[1]);
        _mm256_store_pd(dst + dex + 4, _mm256_castps_pd(res));

        res = _mm256_setr_ps(((float*)&tmp[0])[2], 0, 
                             ((float*)&tmp[1])[2], ((float*)&tmp[2])[2],
                             ((float*)&tmp[3])[2], 0, 
                             ((float*)&tmp[1])[2], -((float*)&tmp[2])[2]);
        _mm256_store_pd(dst + dex + 8, _mm256_castps_pd(res));

        res = _mm256_setr_ps(((float*)&tmp[0])[3], 0, 
                             ((float*)&tmp[1])[3], ((float*)&tmp[2])[3],
                             ((float*)&tmp[3])[3], 0, 
                             ((float*)&tmp[1])[3], -((float*)&tmp[2])[3]);
        _mm256_store_pd(dst + dex + 12, _mm256_castps_pd(res));
#endif
    }
}



_THREAD_FUNCTION_
void decx::signal::CPUK::_FFT1D_R4_fp32_C2C_first_ST_vec4(const double* __restrict src,
                                                    double* __restrict dst,
                                                    const size_t signal_length,
                                                    const uint2 b_op_dex_range)
{
    __m256 recv[4], tmp[4];
    __m256 res;
    size_t dex = 0;
    const size_t half_signal_len = signal_length / 4;

    for (uint i = b_op_dex_range.x; i < b_op_dex_range.y; ++i) {
        recv[0] = _mm256_castpd_ps(_mm256_load_pd(src + (i << 2)));
        recv[1] = _mm256_castpd_ps(_mm256_load_pd(src + (i << 2) + half_signal_len));
        recv[2] = _mm256_castpd_ps(_mm256_load_pd(src + (i << 2) + (half_signal_len << 1)));
        recv[3] = _mm256_castpd_ps(_mm256_load_pd(src + (i << 2) + half_signal_len * 3));

        // res 0
        tmp[0] = _mm256_add_ps(recv[0], recv[1]);
        tmp[0] = _mm256_add_ps(tmp[0], _mm256_add_ps(recv[2], recv[3]));
        // res 1
        tmp[1] = _mm256_addsub_ps(recv[0], _mm256_permute_ps(recv[1], 0b10110001));
        tmp[1] = _mm256_sub_ps(tmp[1], recv[2]);
        tmp[1] = _mm256_permute_ps(_mm256_addsub_ps(_mm256_permute_ps(tmp[1], 0b10110001), recv[3]), 0b10110001);
        // res 2
        tmp[2] = _mm256_sub_ps(recv[0], recv[1]);
        tmp[2] = _mm256_add_ps(tmp[2], recv[2]);
        tmp[2] = _mm256_sub_ps(tmp[2], recv[3]);

        tmp[3] = _mm256_permute_ps(_mm256_addsub_ps(_mm256_permute_ps(recv[0], 0b10110001), recv[1]), 0b10110001);
        tmp[3] = _mm256_sub_ps(tmp[3], recv[2]);
        tmp[3] = _mm256_addsub_ps(tmp[3], _mm256_permute_ps(recv[3], 0b10110001));

#ifdef _MSC_VER
        dex = i * 16;
        res = _mm256_setr_ps(tmp[0].m256_f32[0], tmp[0].m256_f32[1],
                             tmp[1].m256_f32[0], tmp[1].m256_f32[1],
                             tmp[2].m256_f32[0], tmp[2].m256_f32[1],
                             tmp[3].m256_f32[0], tmp[3].m256_f32[1]);
        _mm256_store_pd(dst + dex, _mm256_castps_pd(res));

        res = _mm256_setr_ps(tmp[0].m256_f32[2], tmp[0].m256_f32[3],
                             tmp[1].m256_f32[2], tmp[1].m256_f32[3],
                             tmp[2].m256_f32[2], tmp[2].m256_f32[3],
                             tmp[3].m256_f32[2], tmp[3].m256_f32[3]);
        _mm256_store_pd(dst + dex + 4, _mm256_castps_pd(res));

        res = _mm256_setr_ps(tmp[0].m256_f32[4], tmp[0].m256_f32[5],
                             tmp[1].m256_f32[4], tmp[1].m256_f32[5],
                             tmp[2].m256_f32[4], tmp[2].m256_f32[5],
                             tmp[3].m256_f32[4], tmp[3].m256_f32[5]);
        _mm256_store_pd(dst + dex + 8, _mm256_castps_pd(res));

        res = _mm256_setr_ps(tmp[0].m256_f32[6], tmp[0].m256_f32[7],
                             tmp[1].m256_f32[6], tmp[1].m256_f32[7],
                             tmp[2].m256_f32[6], tmp[2].m256_f32[7],
                             tmp[3].m256_f32[6], tmp[3].m256_f32[7]);
        _mm256_store_pd(dst + dex + 12, _mm256_castps_pd(res));
#endif

#ifdef __GNUC__
        res = _mm256_setr_ps(((float*)&tmp[0])[0], 0, 
                             ((float*)&tmp[1])[0], ((float*)&tmp[2])[0],
                             ((float*)&tmp[3])[0], 0, 
                             ((float*)&tmp[1])[0], -((float*)&tmp[2])[0]);
        _mm256_store_pd(dst + dex, _mm256_castps_pd(res));

        res = _mm256_setr_ps(((float*)&tmp[0])[1], 0, 
                             ((float*)&tmp[1])[1], ((float*)&tmp[2])[1],
                             ((float*)&tmp[3])[1], 0, 
                             ((float*)&tmp[1])[1], -((float*)&tmp[2])[1]);
        _mm256_store_pd(dst + dex + 4, _mm256_castps_pd(res));

        res = _mm256_setr_ps(((float*)&tmp[0])[2], 0, 
                             ((float*)&tmp[1])[2], ((float*)&tmp[2])[2],
                             ((float*)&tmp[3])[2], 0, 
                             ((float*)&tmp[1])[2], -((float*)&tmp[2])[2]);
        _mm256_store_pd(dst + dex + 8, _mm256_castps_pd(res));

        res = _mm256_setr_ps(((float*)&tmp[0])[3], 0, 
                             ((float*)&tmp[1])[3], ((float*)&tmp[2])[3],
                             ((float*)&tmp[3])[3], 0, 
                             ((float*)&tmp[1])[3], -((float*)&tmp[2])[3]);
        _mm256_store_pd(dst + dex + 12, _mm256_castps_pd(res));
#endif
    }
}



_THREAD_FUNCTION_
void decx::signal::CPUK::_FFT1D_R4_fp32_C2C_ST(const double* __restrict src, 
                                         double* __restrict dst, 
                                         const size_t signal_length, 
                                         const size_t warp_proc_len,
                                         const uint2 b_op_dex_range)
{
    de::CPf recv[4], res[4], W, tmp;

    size_t dex = 0;
    uint num_of_Bcalc_in_warp = warp_proc_len / 4, warp_loc_id;

    const size_t total_Bcalc_num = signal_length / 4;

    for (int i = b_op_dex_range.x; i < b_op_dex_range.y; ++i) {
        dex = i;
        *((double*)&recv[0]) = src[dex];
        *((double*)&recv[1]) = src[dex + total_Bcalc_num];
        *((double*)&recv[2]) = src[dex + total_Bcalc_num * 2];
        *((double*)&recv[3]) = src[dex + total_Bcalc_num * 3];

        warp_loc_id = i % num_of_Bcalc_in_warp;

        W.construct_with_phase(2 * Pi * (float)warp_loc_id / (float)warp_proc_len);
        recv[1] = _complex_mul(recv[1], W);
        W.construct_with_phase(2 * Pi * (float)(warp_loc_id * 2) / (float)warp_proc_len);
        recv[2] = _complex_mul(recv[2], W);
        W.construct_with_phase(2 * Pi * (float)(warp_loc_id * 3) / (float)warp_proc_len);
        recv[3] = _complex_mul(recv[3], W);

        res[0].real = recv[0].real + recv[1].real + recv[2].real + recv[3].real;
        res[0].image = recv[0].image + recv[1].image + recv[2].image + recv[3].image;

        tmp = recv[0];
        //tmp = _complex_fma(recv[1], de::CPf(0, 1), tmp);
        tmp.real -= recv[1].image;
        tmp.image += recv[1].real;
        //tmp = _complex_fma(recv[2], de::CPf(-1, 0), tmp);
        tmp.real -= recv[2].real;
        tmp.image -= recv[2].image;
        //res[1] = _complex_fma(recv[3], de::CPf(0, -1), tmp);
        res[1].real = tmp.real + recv[3].image;
        res[1].image = tmp.image - recv[3].real;

        tmp = recv[0];
        //tmp = _complex_fma(recv[1], de::CPf(-1, 0), tmp);
        tmp.real -= recv[1].real;
        tmp.image -= recv[1].image;
        //tmp = _complex_fma(recv[2], de::CPf(1, 0), tmp);
        tmp.real += recv[2].real;
        tmp.image += recv[2].image;
        //res[2] = _complex_fma(recv[3], de::CPf(-1, 0), tmp);
        res[2].real = tmp.real - recv[3].real;
        res[2].image = tmp.image - recv[3].image;

        tmp = recv[0];
        //tmp = _complex_fma(recv[1], de::CPf(0, -1), tmp);
        tmp.real += recv[1].image;
        tmp.image -= recv[1].real;
        //tmp = _complex_fma(recv[2], de::CPf(-1, 0), tmp);
        tmp.real -= recv[2].real;
        tmp.image -= recv[2].image;
        //res[3] = _complex_fma(recv[3], de::CPf(0, 1), tmp);
        res[3].real = tmp.real - recv[3].image;
        res[3].image = tmp.image + recv[3].real;

        dex = (i / num_of_Bcalc_in_warp) * warp_proc_len + warp_loc_id;
        dst[dex] = *((double*)&res[0]);
        dst[dex + num_of_Bcalc_in_warp] = *((double*)&res[1]);
        dst[dex + num_of_Bcalc_in_warp * 2] = *((double*)&res[2]);
        dst[dex + num_of_Bcalc_in_warp * 3] = *((double*)&res[3]);
    }
}



_THREAD_FUNCTION_
void decx::signal::CPUK::_FFT1D_R4_fp32_C2C_ST_vec4(const double* __restrict      src, 
                                              double* __restrict            dst, 
                                              const size_t                  signal_length,      // in element
                                              const size_t                  warp_proc_len,      // in element
                                              const uint2                   b_op_dex_range)     // in vec4
{
    __m256 recv[4];
    __m256 res;

    size_t dex = 0;     // in vec4
    uint num_of_Bcalc_in_warp = warp_proc_len / 4 / 4,  // in vec4
        warp_loc_id;        // in vec4

    const size_t total_Bcalc_num = signal_length / 4;

    for (int i = b_op_dex_range.x; i < b_op_dex_range.y; ++i) {
        recv[0] = _mm256_castpd_ps(_mm256_load_pd(src + (i << 2)));
        recv[1] = _mm256_castpd_ps(_mm256_load_pd(src + (i << 2) + total_Bcalc_num));
        recv[2] = _mm256_castpd_ps(_mm256_load_pd(src + (i << 2) + total_Bcalc_num * 2));
        recv[3] = _mm256_castpd_ps(_mm256_load_pd(src + (i << 2) + total_Bcalc_num * 3));

        warp_loc_id = i % num_of_Bcalc_in_warp;
        dex = (i / num_of_Bcalc_in_warp) * warp_proc_len / 4 + warp_loc_id;
        
        recv[1] = decx::signal::CPUK::_cp4_mul_cp4_fp32(recv[1], 
            _mm256_setr_ps(cosf(Two_Pi * (warp_loc_id * 4) / warp_proc_len), sinf(Two_Pi * ((warp_loc_id << 2)) / warp_proc_len),
                cosf(Two_Pi * ((warp_loc_id << 2) + 1) / warp_proc_len), sinf(Two_Pi * ((warp_loc_id << 2) + 1) / warp_proc_len),
                cosf(Two_Pi * ((warp_loc_id << 2) + 2) / warp_proc_len), sinf(Two_Pi * ((warp_loc_id << 2) + 2) / warp_proc_len),
                cosf(Two_Pi * ((warp_loc_id << 2) + 3) / warp_proc_len), sinf(Two_Pi * ((warp_loc_id << 2) + 3) / warp_proc_len)));
        
        recv[2] = decx::signal::CPUK::_cp4_mul_cp4_fp32(recv[2],
            _mm256_setr_ps(cosf(Two_Pi * (warp_loc_id * 4) * 2 / warp_proc_len), sinf(Two_Pi * ((warp_loc_id << 2)) * 2 / warp_proc_len),
                cosf(Two_Pi * ((warp_loc_id << 2) + 1) * 2 / warp_proc_len), sinf(Two_Pi * ((warp_loc_id << 2) + 1) * 2 / warp_proc_len),
                cosf(Two_Pi * ((warp_loc_id << 2) + 2) * 2 / warp_proc_len), sinf(Two_Pi * ((warp_loc_id << 2) + 2) * 2 / warp_proc_len),
                cosf(Two_Pi * ((warp_loc_id << 2) + 3) * 2 / warp_proc_len), sinf(Two_Pi * ((warp_loc_id << 2) + 3) * 2 / warp_proc_len)));

        recv[3] = decx::signal::CPUK::_cp4_mul_cp4_fp32(recv[3],
            _mm256_setr_ps(cosf(Two_Pi * (warp_loc_id * 4) * 3 / warp_proc_len), sinf(Two_Pi * ((warp_loc_id << 2)) * 3 / warp_proc_len),
                cosf(Two_Pi * ((warp_loc_id << 2) + 1) * 3 / warp_proc_len), sinf(Two_Pi * ((warp_loc_id << 2) + 1) * 3 / warp_proc_len),
                cosf(Two_Pi * ((warp_loc_id << 2) + 2) * 3 / warp_proc_len), sinf(Two_Pi * ((warp_loc_id << 2) + 2) * 3 / warp_proc_len),
                cosf(Two_Pi * ((warp_loc_id << 2) + 3) * 3 / warp_proc_len), sinf(Two_Pi * ((warp_loc_id << 2) + 3) * 3 / warp_proc_len)));

        // res 0
        res = _mm256_add_ps(recv[0], recv[1]);
        res = _mm256_add_ps(res, _mm256_add_ps(recv[2], recv[3]));
        _mm256_store_pd(dst + dex * 4, _mm256_castps_pd(res));
        // res 1
        dex += num_of_Bcalc_in_warp;
        res = _mm256_addsub_ps(recv[0], _mm256_permute_ps(recv[1], 0b10110001));
        res = _mm256_sub_ps(res, recv[2]);
        res = _mm256_permute_ps(_mm256_addsub_ps(_mm256_permute_ps(res, 0b10110001), recv[3]), 0b10110001);
        _mm256_store_pd(dst + dex * 4, _mm256_castps_pd(res));
        // res 2
        dex += num_of_Bcalc_in_warp;
        res = _mm256_sub_ps(recv[0], recv[1]);
        res = _mm256_add_ps(res, recv[2]);
        res = _mm256_sub_ps(res, recv[3]);
        _mm256_store_pd(dst + dex * 4, _mm256_castps_pd(res));

        dex += num_of_Bcalc_in_warp;
        res = _mm256_permute_ps(_mm256_addsub_ps(_mm256_permute_ps(recv[0], 0b10110001), recv[1]), 0b10110001);
        res = _mm256_sub_ps(res, recv[2]);
        res = _mm256_addsub_ps(res, _mm256_permute_ps(recv[3], 0b10110001));
        _mm256_store_pd(dst + dex * 4, _mm256_castps_pd(res));
    }
}



void decx::signal::cpu::FFT_R4_R2C_assign_task_1D_first(decx::utils::_thr_1D* t1D, decx::utils::frag_manager* f_mgr, 
    const float* __restrict src, double* __restrict dst, const size_t signal_length)
{
    const size_t _total_Bop_num = signal_length / 4;
    
    decx::fft::__called_ST_func_first _kernel_ptr = NULL;
    if (_total_Bop_num % 4) {
        _kernel_ptr = decx::signal::CPUK::_FFT1D_R4_fp32_R2C_first_ST;
        decx::utils::frag_manager_gen(f_mgr, _total_Bop_num, t1D->total_thread);
    }
    else {
        _kernel_ptr = decx::signal::CPUK::_FFT1D_R4_fp32_R2C_first_ST_vec4;
        decx::utils::frag_manager_gen(f_mgr, _total_Bop_num / 4, t1D->total_thread);
    }

    uint2 start_end = make_uint2(0, 0);
    if (f_mgr->is_left) {
        for (int i = 0; i < t1D->total_thread - 1; ++i) {
            start_end.y += f_mgr->frag_len;
            t1D->_async_thread[i] = decx::cpu::register_task(&decx::thread_pool, _kernel_ptr, src, dst, signal_length, start_end);
            start_end.x += f_mgr->frag_len;
        }
        start_end.y += f_mgr->frag_left_over;
        t1D->_async_thread[t1D->total_thread - 1] = decx::cpu::register_task(&decx::thread_pool, _kernel_ptr, src, dst, signal_length, start_end);
    }
    else {
        for (int i = 0; i < t1D->total_thread; ++i) {
            start_end.y += f_mgr->frag_len;
            t1D->_async_thread[i] = decx::cpu::register_task(&decx::thread_pool, _kernel_ptr, src, dst, signal_length, start_end);
            start_end.x += f_mgr->frag_len;
        }
    }
    t1D->__sync_all_threads();
}



void decx::signal::cpu::IFFT_R4_C2C_assign_task_1D_first(decx::utils::_thr_1D* t1D, decx::utils::frag_manager* f_mgr, 
    const double* __restrict src, double* __restrict dst, const size_t signal_length)
{
    const size_t _total_Bop_num = signal_length / 4;
    
    decx::fft::__called_ST_func_IFFT_first _kernel_ptr = NULL;
    if (_total_Bop_num % 4) {
        _kernel_ptr = decx::signal::CPUK::_IFFT1D_R4_fp32_C2C_first_ST;
        decx::utils::frag_manager_gen(f_mgr, _total_Bop_num, t1D->total_thread);
    }
    else {
        _kernel_ptr = decx::signal::CPUK::_IFFT1D_R4_fp32_C2C_first_ST_vec4;
        decx::utils::frag_manager_gen(f_mgr, _total_Bop_num / 4, t1D->total_thread);
    }
    
    uint2 start_end = make_uint2(0, 0);
    if (f_mgr->is_left) {
        for (int i = 0; i < t1D->total_thread - 1; ++i) {
            start_end.y += f_mgr->frag_len;
            t1D->_async_thread[i] = decx::cpu::register_task(&decx::thread_pool, _kernel_ptr, src, dst, signal_length, start_end);
            start_end.x += f_mgr->frag_len;
        }
        start_end.y += f_mgr->frag_left_over;
        t1D->_async_thread[t1D->total_thread - 1] = decx::cpu::register_task(&decx::thread_pool, _kernel_ptr, src, dst, signal_length, start_end);
    }
    else {
        for (int i = 0; i < t1D->total_thread; ++i) {
            start_end.y += f_mgr->frag_len;
            t1D->_async_thread[i] = decx::cpu::register_task(&decx::thread_pool, _kernel_ptr, src, dst, signal_length, start_end);
            start_end.x += f_mgr->frag_len;
        }
    }
    t1D->__sync_all_threads();
}


void decx::signal::cpu::FFT_R4_C2C_assign_task_1D_first(decx::utils::_thr_1D* t1D, decx::utils::frag_manager* f_mgr,
    const double* __restrict src, double* __restrict dst, const size_t signal_length)
{
    const size_t _total_Bop_num = signal_length / 4;

    decx::fft::__called_ST_func_IFFT_first _kernel_ptr = NULL;
    if (_total_Bop_num % 4) {
        _kernel_ptr = decx::signal::CPUK::_FFT1D_R4_fp32_C2C_first_ST;
        decx::utils::frag_manager_gen(f_mgr, _total_Bop_num, t1D->total_thread);
    }
    else {
        _kernel_ptr = decx::signal::CPUK::_FFT1D_R4_fp32_C2C_first_ST_vec4;
        decx::utils::frag_manager_gen(f_mgr, _total_Bop_num / 4, t1D->total_thread);
    }

    uint2 start_end = make_uint2(0, 0);
    if (f_mgr->is_left) {
        for (int i = 0; i < t1D->total_thread - 1; ++i) {
            start_end.y += f_mgr->frag_len;
            t1D->_async_thread[i] = decx::cpu::register_task(&decx::thread_pool, _kernel_ptr, src, dst, signal_length, start_end);
            start_end.x += f_mgr->frag_len;
        }
        start_end.y += f_mgr->frag_left_over;
        t1D->_async_thread[t1D->total_thread - 1] = decx::cpu::register_task(&decx::thread_pool, _kernel_ptr, src, dst, signal_length, start_end);
    }
    else {
        for (int i = 0; i < t1D->total_thread; ++i) {
            start_end.y += f_mgr->frag_len;
            t1D->_async_thread[i] = decx::cpu::register_task(&decx::thread_pool, _kernel_ptr, src, dst, signal_length, start_end);
            start_end.x += f_mgr->frag_len;
        }
    }
    t1D->__sync_all_threads();
}



void decx::signal::cpu::FFT_R4_C2C_assign_task_1D(decx::utils::_thr_1D* t1D, decx::utils::frag_manager* f_mgr,
    decx::alloc::MIF<double>* MIF_0, decx::alloc::MIF<double>* MIF_1, const size_t signal_length,
    const size_t warp_proc_len)
{
    decx::fft::__called_ST_func kernel_ptr = NULL;

    const size_t _total_Bop_num = signal_length / 4;

    if ((warp_proc_len / 4) % 4) {
        kernel_ptr = decx::signal::CPUK::_FFT1D_R4_fp32_C2C_ST;
        decx::utils::frag_manager_gen(f_mgr, _total_Bop_num, t1D->total_thread);
    }
    else {
        kernel_ptr = decx::signal::CPUK::_FFT1D_R4_fp32_C2C_ST_vec4;
        decx::utils::frag_manager_gen(f_mgr, _total_Bop_num / 4, t1D->total_thread);
    }
    uint2 start_end = make_uint2(0, 0);

    double* read_ptr = NULL, * write_ptr = NULL;
    if (MIF_0->leading) {
        read_ptr = MIF_0->mem;          write_ptr = MIF_1->mem;
        decx::utils::set_mutex_memory_state<double, double>(MIF_1, MIF_0);
    }
    else {
        read_ptr = MIF_1->mem;          write_ptr = MIF_0->mem;
        decx::utils::set_mutex_memory_state<double, double>(MIF_0, MIF_1);
    }

    if (f_mgr->is_left) {
        for (int i = 0; i < t1D->total_thread - 1; ++i) {
            start_end.y += f_mgr->frag_len;
            t1D->_async_thread[i] = decx::cpu::register_task(&decx::thread_pool, kernel_ptr, read_ptr, write_ptr, signal_length, warp_proc_len, start_end);
            start_end.x += f_mgr->frag_len;
        }
        start_end.y += f_mgr->frag_left_over;
        t1D->_async_thread[t1D->total_thread - 1] = decx::cpu::register_task(&decx::thread_pool, kernel_ptr, read_ptr, write_ptr, signal_length, warp_proc_len, start_end);
    }
    else {
        for (int i = 0; i < t1D->total_thread; ++i) {
            start_end.y += f_mgr->frag_len;
            t1D->_async_thread[i] = decx::cpu::register_task(&decx::thread_pool, kernel_ptr, read_ptr, write_ptr, signal_length, warp_proc_len, start_end);
            start_end.x += f_mgr->frag_len;
        }
    }
    t1D->__sync_all_threads();
}

