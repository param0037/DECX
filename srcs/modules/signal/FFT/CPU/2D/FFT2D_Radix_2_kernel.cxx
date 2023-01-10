/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/

#include "FFT2D_Radix_2_kernel.h"



_THREAD_CALL_ void 
decx::signal::CPUK::_FFT2D_R2_fp32_R2C_first_ST_vec4col(const float* __restrict src,
                                                  double* __restrict dst, 
                                                  const uint signal_W,
                                                  const uint Wsrc,
                                                  const uint procW,
                                                  const uint procH)
{
    float recv[2][8];
    __m256 res[2];
    __m256d O_buffer[2];

    size_t dex = 0, dex_src = 0;
    const uint b_op_num = signal_W / 2;
    uint dex_w = 0;

    for (int i = 0; i < procH; ++i) {
        for (int j = 0; j < b_op_num; ++j) {
            dex_w = j;
            recv[0][0] = src[dex_src + dex_w];      recv[1][0] = src[dex_src + dex_w + b_op_num];
            dex_w += Wsrc;                                              
            recv[0][2] = src[dex_src + dex_w];      recv[1][2] = src[dex_src + dex_w + b_op_num];
            dex_w += Wsrc;                                              
            recv[0][4] = src[dex_src + dex_w];      recv[1][4] = src[dex_src + dex_w + b_op_num];
            dex_w += Wsrc;                                              
            recv[0][6] = src[dex_src + dex_w];      recv[1][6] = src[dex_src + dex_w + b_op_num];

            res[0] = _mm256_add_ps(_mm256_load_ps(recv[0]), _mm256_load_ps(recv[1]));
            res[1] = _mm256_sub_ps(_mm256_load_ps(recv[0]), _mm256_load_ps(recv[1]));

            O_buffer[0] = _mm256_unpacklo_pd(_mm256_castps_pd(res[0]), _mm256_castps_pd(res[1]));
            O_buffer[1] = _mm256_unpackhi_pd(_mm256_castps_pd(res[0]), _mm256_castps_pd(res[1]));

            dex_w = j * 2;
            _mm_store_pd(dst + dex + dex_w, ((__m128d*)&O_buffer[0])[0]);
            dex_w += procW;
            _mm_store_pd(dst + dex + dex_w, ((__m128d*)&O_buffer[1])[0]);
            dex_w += procW;
            _mm_store_pd(dst + dex + dex_w, ((__m128d*)&O_buffer[0])[1]);
            dex_w += procW;
            _mm_store_pd(dst + dex + dex_w, ((__m128d*)&O_buffer[1])[1]);
        }
        dex += 4 * procW;
        dex_src += 4 * Wsrc;
    }
}



_THREAD_CALL_ void 
decx::signal::CPUK::_IFFT2D_R2_fp32_C2C_first_ST_vec4col(const double* __restrict src,
                                                   double* __restrict dst, 
                                                   const uint signal_W,
                                                   const uint procW,
                                                   const uint procH)
{
    double recv[2][4];
    __m256 res[2];
    __m256d O_buffer[2];

    size_t dex = 0;
    const uint b_op_num = signal_W / 2;
    uint dex_w = 0;

    for (int i = 0; i < procH; ++i) {
        for (int j = 0; j < b_op_num; ++j) {
            dex_w = j;
            recv[0][0] = src[dex + dex_w];      recv[1][0] = src[dex + dex_w + b_op_num];
            dex_w += procW;
            recv[0][1] = src[dex + dex_w];      recv[1][1] = src[dex + dex_w + b_op_num];
            dex_w += procW;
            recv[0][2] = src[dex + dex_w];      recv[1][2] = src[dex + dex_w + b_op_num];
            dex_w += procW;
            recv[0][3] = src[dex + dex_w];      recv[1][3] = src[dex + dex_w + b_op_num];

            // conj
            _mm256_store_pd(recv[0], _mm256_castps_pd(_mm256_mul_ps(_mm256_castpd_ps(_mm256_load_pd(recv[0])), _mm256_setr_ps(1, -1, 1, -1, 1, -1, 1, -1))));
            _mm256_store_pd(recv[1], _mm256_castps_pd(_mm256_mul_ps(_mm256_castpd_ps(_mm256_load_pd(recv[1])), _mm256_setr_ps(1, -1, 1, -1, 1, -1, 1, -1))));
            // * /= signal_w
            _mm256_store_pd(recv[0], _mm256_castps_pd(_mm256_div_ps(_mm256_castpd_ps(_mm256_load_pd(recv[0])), _mm256_set1_ps((float)signal_W))));
            _mm256_store_pd(recv[1], _mm256_castps_pd(_mm256_div_ps(_mm256_castpd_ps(_mm256_load_pd(recv[1])), _mm256_set1_ps((float)signal_W))));

            res[0] = _mm256_add_ps(_mm256_castpd_ps(_mm256_load_pd(recv[0])), _mm256_castpd_ps(_mm256_load_pd(recv[1])));
            res[1] = _mm256_sub_ps(_mm256_castpd_ps(_mm256_load_pd(recv[0])), _mm256_castpd_ps(_mm256_load_pd(recv[1])));

            O_buffer[0] = _mm256_unpacklo_pd(_mm256_castps_pd(res[0]), _mm256_castps_pd(res[1]));
            O_buffer[1] = _mm256_unpackhi_pd(_mm256_castps_pd(res[0]), _mm256_castps_pd(res[1]));

            dex_w = j * 2;
            _mm_store_pd(dst + dex + dex_w, ((__m128d*)&O_buffer[0])[0]);
            dex_w += procW;
            _mm_store_pd(dst + dex + dex_w, ((__m128d*)&O_buffer[1])[0]);
            dex_w += procW;
            _mm_store_pd(dst + dex + dex_w, ((__m128d*)&O_buffer[0])[1]);
            dex_w += procW;
            _mm_store_pd(dst + dex + dex_w, ((__m128d*)&O_buffer[1])[1]);
        }
        dex += 4 * procW;
    }
}



_THREAD_CALL_ void 
decx::signal::CPUK::_FFT2D_R2_fp32_C2C_first_ST_vec4col(const double* __restrict src,
                                                   double* __restrict dst, 
                                                   const uint signal_W,
                                                   const uint procW,
                                                   const uint procH)
{
    double recv[2][4];
    __m256 res[2];
    __m256d O_buffer[2];

    size_t dex = 0;
    const uint b_op_num = signal_W / 2;
    uint dex_w = 0;

    for (int i = 0; i < procH; ++i) {
        for (int j = 0; j < b_op_num; ++j) {
            dex_w = j;
            recv[0][0] = src[dex + dex_w];      recv[1][0] = src[dex + dex_w + b_op_num];
            dex_w += procW;
            recv[0][1] = src[dex + dex_w];      recv[1][1] = src[dex + dex_w + b_op_num];
            dex_w += procW;
            recv[0][2] = src[dex + dex_w];      recv[1][2] = src[dex + dex_w + b_op_num];
            dex_w += procW;
            recv[0][3] = src[dex + dex_w];      recv[1][3] = src[dex + dex_w + b_op_num];

            res[0] = _mm256_add_ps(_mm256_castpd_ps(_mm256_load_pd(recv[0])), _mm256_castpd_ps(_mm256_load_pd(recv[1])));
            res[1] = _mm256_sub_ps(_mm256_castpd_ps(_mm256_load_pd(recv[0])), _mm256_castpd_ps(_mm256_load_pd(recv[1])));

            O_buffer[0] = _mm256_unpacklo_pd(_mm256_castps_pd(res[0]), _mm256_castps_pd(res[1]));
            O_buffer[1] = _mm256_unpackhi_pd(_mm256_castps_pd(res[0]), _mm256_castps_pd(res[1]));

            dex_w = j * 2;
            _mm_store_pd(dst + dex + dex_w, ((__m128d*)&O_buffer[0])[0]);
            dex_w += procW;
            _mm_store_pd(dst + dex + dex_w, ((__m128d*)&O_buffer[1])[0]);
            dex_w += procW;
            _mm_store_pd(dst + dex + dex_w, ((__m128d*)&O_buffer[0])[1]);
            dex_w += procW;
            _mm_store_pd(dst + dex + dex_w, ((__m128d*)&O_buffer[1])[1]);
        }
        dex += 4 * procW;
    }
}



_THREAD_CALL_ void 
decx::signal::CPUK::_IFFT2D_R2_fp32_C2C_first_ST_vec4col_L4(const double* __restrict src,
                                                      double* __restrict dst, 
                                                      const uint signal_W,
                                                      const uint procW,
                                                      const uint procH,
                                                      const uint _Left)
{
    double recv[2][4];
    __m256 res[2];
    __m256d O_buffer[2];

    size_t dex = 0;
    const uint b_op_num = signal_W / 2;
    uint dex_w = 0;

    for (int i = 0; i < procH; ++i) {
        for (int j = 0; j < b_op_num; ++j) {
            dex_w = j;
            recv[0][0] = src[dex + dex_w];      recv[1][0] = src[dex + dex_w + b_op_num];
            dex_w += procW;
            recv[0][1] = src[dex + dex_w];      recv[1][1] = src[dex + dex_w + b_op_num];
            dex_w += procW;
            recv[0][2] = src[dex + dex_w];      recv[1][2] = src[dex + dex_w + b_op_num];
            dex_w += procW;
            recv[0][3] = src[dex + dex_w];      recv[1][3] = src[dex + dex_w + b_op_num];

            // conj
            _mm256_store_pd(recv[0], _mm256_castps_pd(_mm256_mul_ps(_mm256_castpd_ps(_mm256_load_pd(recv[0])), _mm256_setr_ps(1, -1, 1, -1, 1, -1, 1, -1))));
            _mm256_store_pd(recv[1], _mm256_castps_pd(_mm256_mul_ps(_mm256_castpd_ps(_mm256_load_pd(recv[1])), _mm256_setr_ps(1, -1, 1, -1, 1, -1, 1, -1))));
            // * /= signal_w
            _mm256_store_pd(recv[0], _mm256_castps_pd(_mm256_div_ps(_mm256_castpd_ps(_mm256_load_pd(recv[0])), _mm256_set1_ps((float)signal_W))));
            _mm256_store_pd(recv[1], _mm256_castps_pd(_mm256_div_ps(_mm256_castpd_ps(_mm256_load_pd(recv[1])), _mm256_set1_ps((float)signal_W))));

            res[0] = _mm256_add_ps(_mm256_castpd_ps(_mm256_load_pd(recv[0])), _mm256_castpd_ps(_mm256_load_pd(recv[1])));
            res[1] = _mm256_sub_ps(_mm256_castpd_ps(_mm256_load_pd(recv[0])), _mm256_castpd_ps(_mm256_load_pd(recv[1])));

            O_buffer[0] = _mm256_unpacklo_pd(_mm256_castps_pd(res[0]), _mm256_castps_pd(res[1]));
            O_buffer[1] = _mm256_unpackhi_pd(_mm256_castps_pd(res[0]), _mm256_castps_pd(res[1]));

            dex_w = j * 2;
            _mm_store_pd(dst + dex + dex_w, ((__m128d*)&O_buffer[0])[0]);
            dex_w += procW;
            _mm_store_pd(dst + dex + dex_w, ((__m128d*)&O_buffer[1])[0]);
            dex_w += procW;
            _mm_store_pd(dst + dex + dex_w, ((__m128d*)&O_buffer[0])[1]);
            dex_w += procW;
            _mm_store_pd(dst + dex + dex_w, ((__m128d*)&O_buffer[1])[1]);
        }
        dex += 4 * procW;
    }

    for (int j = 0; j < b_op_num; ++j) {
        dex_w = j;
        for (int k = 0; k < _Left; ++k) {
            recv[0][k] = src[dex + dex_w];      recv[1][k] = src[dex + dex_w + b_op_num];
            dex_w += procW;
        }
        
        // conj
        _mm256_store_pd(recv[0], _mm256_castps_pd(_mm256_mul_ps(_mm256_castpd_ps(_mm256_load_pd(recv[0])), _mm256_setr_ps(1, -1, 1, -1, 1, -1, 1, -1))));
        _mm256_store_pd(recv[1], _mm256_castps_pd(_mm256_mul_ps(_mm256_castpd_ps(_mm256_load_pd(recv[1])), _mm256_setr_ps(1, -1, 1, -1, 1, -1, 1, -1))));
        // * /= signal_w
        _mm256_store_pd(recv[0], _mm256_castps_pd(_mm256_div_ps(_mm256_castpd_ps(_mm256_load_pd(recv[0])), _mm256_set1_ps((float)signal_W))));
        _mm256_store_pd(recv[1], _mm256_castps_pd(_mm256_div_ps(_mm256_castpd_ps(_mm256_load_pd(recv[1])), _mm256_set1_ps((float)signal_W))));

        res[0] = _mm256_add_ps(_mm256_castpd_ps(_mm256_load_pd(recv[0])), _mm256_castpd_ps(_mm256_load_pd(recv[1])));
        res[1] = _mm256_sub_ps(_mm256_castpd_ps(_mm256_load_pd(recv[0])), _mm256_castpd_ps(_mm256_load_pd(recv[1])));

        O_buffer[0] = _mm256_unpacklo_pd(_mm256_castps_pd(res[0]), _mm256_castps_pd(res[1]));
        O_buffer[1] = _mm256_unpackhi_pd(_mm256_castps_pd(res[0]), _mm256_castps_pd(res[1]));

        dex_w = j * 2;
        for (int k = 0; k < _Left; ++k) {
            _mm_store_pd(dst + dex + dex_w, ((__m128d*) & O_buffer[k % 2])[k / 2]);
            dex_w += procW;
        }
    }
}



_THREAD_CALL_ void 
decx::signal::CPUK::_FFT2D_R2_fp32_C2C_first_ST_vec4col_L4(const double* __restrict src,
                                                      double* __restrict dst, 
                                                      const uint signal_W,
                                                      const uint procW,
                                                      const uint procH,
                                                      const uint _Left)
{
    double recv[2][4];
    __m256 res[2];
    __m256d O_buffer[2];

    size_t dex = 0;
    const uint b_op_num = signal_W / 2;
    uint dex_w = 0;

    for (int i = 0; i < procH; ++i) {
        for (int j = 0; j < b_op_num; ++j) {
            dex_w = j;
            recv[0][0] = src[dex + dex_w];      recv[1][0] = src[dex + dex_w + b_op_num];
            dex_w += procW;
            recv[0][1] = src[dex + dex_w];      recv[1][1] = src[dex + dex_w + b_op_num];
            dex_w += procW;
            recv[0][2] = src[dex + dex_w];      recv[1][2] = src[dex + dex_w + b_op_num];
            dex_w += procW;
            recv[0][3] = src[dex + dex_w];      recv[1][3] = src[dex + dex_w + b_op_num];

            res[0] = _mm256_add_ps(_mm256_castpd_ps(_mm256_load_pd(recv[0])), _mm256_castpd_ps(_mm256_load_pd(recv[1])));
            res[1] = _mm256_sub_ps(_mm256_castpd_ps(_mm256_load_pd(recv[0])), _mm256_castpd_ps(_mm256_load_pd(recv[1])));

            O_buffer[0] = _mm256_unpacklo_pd(_mm256_castps_pd(res[0]), _mm256_castps_pd(res[1]));
            O_buffer[1] = _mm256_unpackhi_pd(_mm256_castps_pd(res[0]), _mm256_castps_pd(res[1]));

            dex_w = j * 2;
            _mm_store_pd(dst + dex + dex_w, ((__m128d*)&O_buffer[0])[0]);
            dex_w += procW;
            _mm_store_pd(dst + dex + dex_w, ((__m128d*)&O_buffer[1])[0]);
            dex_w += procW;
            _mm_store_pd(dst + dex + dex_w, ((__m128d*)&O_buffer[0])[1]);
            dex_w += procW;
            _mm_store_pd(dst + dex + dex_w, ((__m128d*)&O_buffer[1])[1]);
        }
        dex += 4 * procW;
    }

    for (int j = 0; j < b_op_num; ++j) {
        dex_w = j;
        for (int k = 0; k < _Left; ++k) {
            recv[0][k] = src[dex + dex_w];      recv[1][k] = src[dex + dex_w + b_op_num];
            dex_w += procW;
        }
        
        res[0] = _mm256_add_ps(_mm256_castpd_ps(_mm256_load_pd(recv[0])), _mm256_castpd_ps(_mm256_load_pd(recv[1])));
        res[1] = _mm256_sub_ps(_mm256_castpd_ps(_mm256_load_pd(recv[0])), _mm256_castpd_ps(_mm256_load_pd(recv[1])));

        O_buffer[0] = _mm256_unpacklo_pd(_mm256_castps_pd(res[0]), _mm256_castps_pd(res[1]));
        O_buffer[1] = _mm256_unpackhi_pd(_mm256_castps_pd(res[0]), _mm256_castps_pd(res[1]));

        dex_w = j * 2;
        for (int k = 0; k < _Left; ++k) {
            _mm_store_pd(dst + dex + dex_w, ((__m128d*) & O_buffer[k % 2])[k / 2]);
            dex_w += procW;
        }
    }
}



_THREAD_CALL_ void 
decx::signal::CPUK::_FFT2D_R2_fp32_R2C_first_ST_vec4col_L4(const float* __restrict src,
                                                     double* __restrict dst, 
                                                     const uint signal_W,
                                                     const uint Wsrc,
                                                     const uint procW,
                                                     const uint procH,      // in vec4
                                                     const uint _Left)      // in element
{
    float recv[2][8];
    __m256 res[2];
    __m256d O_buffer[2];

    size_t dex = 0, dex_src = 0;
    const uint b_op_num = signal_W / 2;
    uint dex_w = 0;

    for (int i = 0; i < procH; ++i) {
        for (int j = 0; j < b_op_num; ++j) {
            dex_w = j;
            recv[0][0] = src[dex_src + dex_w];      recv[1][0] = src[dex_src + dex_w + b_op_num];
            dex_w += Wsrc;                                              
            recv[0][2] = src[dex_src + dex_w];      recv[1][2] = src[dex_src + dex_w + b_op_num];
            dex_w += Wsrc;                                              
            recv[0][4] = src[dex_src + dex_w];      recv[1][4] = src[dex_src + dex_w + b_op_num];
            dex_w += Wsrc;                                              
            recv[0][6] = src[dex_src + dex_w];      recv[1][6] = src[dex_src + dex_w + b_op_num];

            res[0] = _mm256_add_ps(_mm256_load_ps(recv[0]), _mm256_load_ps(recv[1]));
            res[1] = _mm256_sub_ps(_mm256_load_ps(recv[0]), _mm256_load_ps(recv[1]));

            O_buffer[0] = _mm256_unpacklo_pd(_mm256_castps_pd(res[0]), _mm256_castps_pd(res[1]));
            O_buffer[1] = _mm256_unpackhi_pd(_mm256_castps_pd(res[0]), _mm256_castps_pd(res[1]));

            dex_w = j * 2;
            _mm_store_pd(dst + dex + dex_w, ((__m128d*)&O_buffer[0])[0]);
            dex_w += procW;
            _mm_store_pd(dst + dex + dex_w, ((__m128d*)&O_buffer[1])[0]);
            dex_w += procW;
            _mm_store_pd(dst + dex + dex_w, ((__m128d*)&O_buffer[0])[1]);
            dex_w += procW;
            _mm_store_pd(dst + dex + dex_w, ((__m128d*)&O_buffer[1])[1]);
        }
        dex += 4 * procW;
        dex_src += 4 * Wsrc;
    }

    for (int j = 0; j < b_op_num; ++j) {
        dex_w = j;
        for (int k = 0; k < _Left; ++k) {
            recv[0][k * 2] = src[dex_src + dex_w];      
            recv[1][k * 2] = src[dex_src + dex_w + b_op_num];
            dex_w += Wsrc;
        }
        
        res[0] = _mm256_add_ps(_mm256_load_ps(recv[0]), _mm256_load_ps(recv[1]));
        res[1] = _mm256_sub_ps(_mm256_load_ps(recv[0]), _mm256_load_ps(recv[1]));

        O_buffer[0] = _mm256_unpacklo_pd(_mm256_castps_pd(res[0]), _mm256_castps_pd(res[1]));
        O_buffer[1] = _mm256_unpackhi_pd(_mm256_castps_pd(res[0]), _mm256_castps_pd(res[1]));

        dex_w = j * 2;
        for (int k = 0; k < _Left; ++k) {
            _mm_store_pd(dst + dex + dex_w, ((__m128d*)&O_buffer[k % 2])[k / 2]);
            dex_w += procW;
        }
    }
}




_THREAD_CALL_ void 
decx::signal::CPUK::_FFT2D_R2_fp32_C2C_ST_vec4col(const double* __restrict src, 
                                            double* __restrict dst,
                                            const uint warp_proc_len,       // in element
                                            const uint signal_W,            // in element
                                            const uint procW,
                                            const uint procH)
{
    __m256d recv[2], tmp, res;
    de::CPf W;
    size_t dex = 0;     // in vec4
    uint dex_w = 0;

    uint num_of_Bcalc_in_warp = warp_proc_len / 2,
        warp_loc_id;
    const size_t num_total_Bops = signal_W / 2;

    for (int i = 0; i < procH; ++i) {
        for (int j = 0; j < num_total_Bops; ++j) {
            dex_w = j;
#ifdef _MSC_VER
            recv[0].m256d_f64[0] = src[dex + dex_w];    recv[1].m256d_f64[0] = src[dex + dex_w + num_total_Bops];
            dex_w += procW;
            recv[0].m256d_f64[1] = src[dex + dex_w];    recv[1].m256d_f64[1] = src[dex + dex_w + num_total_Bops];
            dex_w += procW;
            recv[0].m256d_f64[2] = src[dex + dex_w];    recv[1].m256d_f64[2] = src[dex + dex_w + num_total_Bops];
            dex_w += procW;
            recv[0].m256d_f64[3] = src[dex + dex_w];    recv[1].m256d_f64[3] = src[dex + dex_w + num_total_Bops];
#endif
#ifdef __GNUC__
            ((double*)&recv[0])[0] = src[dex + dex_w];    ((double*)&recv[1])[0] = src[dex + dex_w + num_total_Bops];
            dex_w += procW;
            ((double*)&recv[0])[1] = src[dex + dex_w];    ((double*)&recv[1])[1] = src[dex + dex_w + num_total_Bops];
            dex_w += procW;
            ((double*)&recv[0])[2] = src[dex + dex_w];    ((double*)&recv[1])[2] = src[dex + dex_w + num_total_Bops];
            dex_w += procW;
            ((double*)&recv[0])[3] = src[dex + dex_w];    ((double*)&recv[1])[3] = src[dex + dex_w + num_total_Bops];
#endif
            warp_loc_id = j % num_of_Bcalc_in_warp;

            W.construct_with_phase(Two_Pi * (float)warp_loc_id / (float)warp_proc_len);
            tmp = _mm256_castps_pd(decx::signal::CPUK::_cp4_mul_cp1_fp32(_mm256_castpd_ps(recv[1]), W));

            res = _mm256_castps_pd(_mm256_add_ps(_mm256_castpd_ps(recv[0]), _mm256_castpd_ps(tmp)));
            recv[1] = _mm256_castps_pd(_mm256_sub_ps(_mm256_castpd_ps(recv[0]), _mm256_castpd_ps(tmp)));

            dex_w = (j / num_of_Bcalc_in_warp) * warp_proc_len + warp_loc_id;
#ifdef _MSC_VER
            dst[dex + dex_w] = res.m256d_f64[0];    dst[dex + dex_w + num_of_Bcalc_in_warp] = recv[1].m256d_f64[0];
            dex_w += procW;
            dst[dex + dex_w] = res.m256d_f64[1];    dst[dex + dex_w + num_of_Bcalc_in_warp] = recv[1].m256d_f64[1];
            dex_w += procW;
            dst[dex + dex_w] = res.m256d_f64[2];    dst[dex + dex_w + num_of_Bcalc_in_warp] = recv[1].m256d_f64[2];
            dex_w += procW;
            dst[dex + dex_w] = res.m256d_f64[3];    dst[dex + dex_w + num_of_Bcalc_in_warp] = recv[1].m256d_f64[3];
#endif
#ifdef __GNUC__
            dst[dex + dex_w] = ((double*)&res)[0];    dst[dex + dex_w + num_of_Bcalc_in_warp] = ((double*)&recv[1])[0];
            dex_w += procW;
            dst[dex + dex_w] = ((double*)&res)[1];    dst[dex + dex_w + num_of_Bcalc_in_warp] = ((double*)&recv[1])[1];
            dex_w += procW;
            dst[dex + dex_w] = ((double*)&res)[2];    dst[dex + dex_w + num_of_Bcalc_in_warp] = ((double*)&recv[1])[2];
            dex_w += procW;
            dst[dex + dex_w] = ((double*)&res)[3];    dst[dex + dex_w + num_of_Bcalc_in_warp] = ((double*)&recv[1])[3];
#endif
        }
        dex += 4 * procW;
    }
}




_THREAD_CALL_ void 
decx::signal::CPUK::_FFT2D_R2_fp32_C2C_ST_vec4col_L4(const double* __restrict src, 
                                               double* __restrict dst,
                                               const uint warp_proc_len,       // in element
                                               const uint signal_W,            // in element
                                               const uint procW,
                                               const uint procH,
                                               const uint _Left)
{
    __m256d recv[2], tmp, res;
    de::CPf W;
    size_t dex = 0;     // in vec4
    uint dex_w = 0;

    uint num_of_Bcalc_in_warp = warp_proc_len / 2,
        warp_loc_id;
    const size_t num_total_Bops = signal_W / 2;

    for (int i = 0; i < procH; ++i) {
        for (int j = 0; j < num_total_Bops; ++j) {
            dex_w = j;
#ifdef _MSC_VER
            recv[0].m256d_f64[0] = src[dex + dex_w];    recv[1].m256d_f64[0] = src[dex + dex_w + num_total_Bops];
            dex_w += procW;
            recv[0].m256d_f64[1] = src[dex + dex_w];    recv[1].m256d_f64[1] = src[dex + dex_w + num_total_Bops];
            dex_w += procW;
            recv[0].m256d_f64[2] = src[dex + dex_w];    recv[1].m256d_f64[2] = src[dex + dex_w + num_total_Bops];
            dex_w += procW;
            recv[0].m256d_f64[3] = src[dex + dex_w];    recv[1].m256d_f64[3] = src[dex + dex_w + num_total_Bops];
#endif
#ifdef __GNUC__
            ((double*)&recv[0])[0] = src[dex + dex_w];    ((double*)&recv[1])[0] = src[dex + dex_w + num_total_Bops];
            dex_w += procW;
            ((double*)&recv[0])[1] = src[dex + dex_w];    ((double*)&recv[1])[1] = src[dex + dex_w + num_total_Bops];
            dex_w += procW;
            ((double*)&recv[0])[2] = src[dex + dex_w];    ((double*)&recv[1])[2] = src[dex + dex_w + num_total_Bops];
            dex_w += procW;
            ((double*)&recv[0])[3] = src[dex + dex_w];    ((double*)&recv[1])[3] = src[dex + dex_w + num_total_Bops];
#endif
            warp_loc_id = j % num_of_Bcalc_in_warp;

            W.construct_with_phase(Two_Pi * (float)warp_loc_id / (float)warp_proc_len);
            tmp = _mm256_castps_pd(decx::signal::CPUK::_cp4_mul_cp1_fp32(_mm256_castpd_ps(recv[1]), W));

            res = _mm256_castps_pd(_mm256_add_ps(_mm256_castpd_ps(recv[0]), _mm256_castpd_ps(tmp)));
            recv[1] = _mm256_castps_pd(_mm256_sub_ps(_mm256_castpd_ps(recv[0]), _mm256_castpd_ps(tmp)));

            dex_w = (j / num_of_Bcalc_in_warp) * warp_proc_len + warp_loc_id;
#ifdef _MSC_VER
            dst[dex + dex_w] = res.m256d_f64[0];    dst[dex + dex_w + num_of_Bcalc_in_warp] = recv[1].m256d_f64[0];
            dex_w += procW;
            dst[dex + dex_w] = res.m256d_f64[1];    dst[dex + dex_w + num_of_Bcalc_in_warp] = recv[1].m256d_f64[1];
            dex_w += procW;
            dst[dex + dex_w] = res.m256d_f64[2];    dst[dex + dex_w + num_of_Bcalc_in_warp] = recv[1].m256d_f64[2];
            dex_w += procW;
            dst[dex + dex_w] = res.m256d_f64[3];    dst[dex + dex_w + num_of_Bcalc_in_warp] = recv[1].m256d_f64[3];
#endif
#ifdef __GNUC__
            dst[dex + dex_w] = ((double*)&res)[0];    dst[dex + dex_w + num_of_Bcalc_in_warp] = ((double*)&recv[1])[0];
            dex_w += procW;
            dst[dex + dex_w] = ((double*)&res)[1];    dst[dex + dex_w + num_of_Bcalc_in_warp] = ((double*)&recv[1])[1];
            dex_w += procW;
            dst[dex + dex_w] = ((double*)&res)[2];    dst[dex + dex_w + num_of_Bcalc_in_warp] = ((double*)&recv[1])[2];
            dex_w += procW;
            dst[dex + dex_w] = ((double*)&res)[3];    dst[dex + dex_w + num_of_Bcalc_in_warp] = ((double*)&recv[1])[3];
#endif
        }
        dex += 4 * procW;
    }

    for (int j = 0; j < num_total_Bops; ++j) {
        dex_w = j;
        for (int k = 0; k < _Left; ++k) {
#ifdef _MSC_VER
            recv[0].m256d_f64[k] = src[dex + dex_w];    
            recv[1].m256d_f64[k] = src[dex + dex_w + num_total_Bops];
#endif
#ifdef __GNUC__
            ((double*)&recv[0])[k] = src[dex + dex_w];    
            ((double*)&recv[1])[k] = src[dex + dex_w + num_total_Bops];
#endif
            dex_w += procW;
        }

        warp_loc_id = j % num_of_Bcalc_in_warp;

        W.construct_with_phase(Two_Pi * (float)warp_loc_id / (float)warp_proc_len);
        tmp = _mm256_castps_pd(decx::signal::CPUK::_cp4_mul_cp1_fp32(_mm256_castpd_ps(recv[1]), W));

        res = _mm256_castps_pd(_mm256_add_ps(_mm256_castpd_ps(recv[0]), _mm256_castpd_ps(tmp)));
        recv[1] = _mm256_castps_pd(_mm256_sub_ps(_mm256_castpd_ps(recv[0]), _mm256_castpd_ps(tmp)));

        dex_w = (j / num_of_Bcalc_in_warp) * warp_proc_len + warp_loc_id;
        for (int k = 0; k < _Left; ++k) {
#ifdef _MSC_VER
            dst[dex + dex_w] = res.m256d_f64[k];    
            dst[dex + dex_w + num_of_Bcalc_in_warp] = recv[1].m256d_f64[k];
#endif
#ifdef __GNUC__
            dst[dex + dex_w] = ((double*)&res)[k];    
            dst[dex + dex_w + num_of_Bcalc_in_warp] = ((double*)&recv[1])[k];
#endif
            dex_w += procW;
        }
    }
}
