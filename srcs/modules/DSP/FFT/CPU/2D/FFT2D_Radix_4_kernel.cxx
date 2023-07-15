/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#include "FFT2D_Radix_4_kernel.h"



_THREAD_CALL_ void 
decx::signal::CPUK::_FFT2D_R4_fp32_R2C_first_ST_vec4col(const float* __restrict src,
                                                  double* __restrict dst, 
                                                  const uint signal_W,
                                                  const uint Wsrc,
                                                  const uint procW,
                                                  const uint procH)
{
    __m256 recv[4];
    __m256 res, tmp0, tmp1;

    size_t dex = 0, dex_src = 0;
    const uint b_op_num = signal_W / 4;
    uint dex_w = 0;

    for (int i = 0; i < procH; ++i) {
        for (int j = 0; j < b_op_num; ++j) {
            dex_w = j;
#ifdef _MSC_VER
            recv[0].m256_f32[0] = src[dex_src + dex_w];
            recv[1].m256_f32[0] = src[dex_src + dex_w + b_op_num];
            recv[2].m256_f32[0] = src[dex_src + dex_w + (b_op_num << 1)];
            recv[3].m256_f32[0] = src[dex_src + dex_w + b_op_num * 3];
            dex_w += Wsrc;               
            recv[0].m256_f32[2] = src[dex_src + dex_w];
            recv[1].m256_f32[2] = src[dex_src + dex_w + b_op_num];
            recv[2].m256_f32[2] = src[dex_src + dex_w + (b_op_num << 1)];
            recv[3].m256_f32[2] = src[dex_src + dex_w + b_op_num * 3];
            dex_w += Wsrc;               
            recv[0].m256_f32[4] = src[dex_src + dex_w];
            recv[1].m256_f32[4] = src[dex_src + dex_w + b_op_num];
            recv[2].m256_f32[4] = src[dex_src + dex_w + (b_op_num << 1)];
            recv[3].m256_f32[4] = src[dex_src + dex_w + b_op_num * 3];
            dex_w += Wsrc;               
            recv[0].m256_f32[6] = src[dex_src + dex_w];
            recv[1].m256_f32[6] = src[dex_src + dex_w + b_op_num];
            recv[2].m256_f32[6] = src[dex_src + dex_w + (b_op_num << 1)];
            recv[3].m256_f32[6] = src[dex_src + dex_w + b_op_num * 3];
#endif
#ifdef __GNUC__
            ((double*)&recv[0])[0] = src[dex_src + dex_w];
            ((double*)&recv[1])[0] = src[dex_src + dex_w + b_op_num];
            ((double*)&recv[2])[0] = src[dex_src + dex_w + (b_op_num << 1)];
            ((double*)&recv[3])[0] = src[dex_src + dex_w + b_op_num * 3];
            dex_w += Wsrc;               
            ((double*)&recv[0])[2] = src[dex_src + dex_w];
            ((double*)&recv[1])[2] = src[dex_src + dex_w + b_op_num];
            ((double*)&recv[2])[2] = src[dex_src + dex_w + (b_op_num << 1)];
            ((double*)&recv[3])[2] = src[dex_src + dex_w + b_op_num * 3];
            dex_w += Wsrc;               
            ((double*)&recv[0])[4] = src[dex_src + dex_w];
            ((double*)&recv[1])[4] = src[dex_src + dex_w + b_op_num];
            ((double*)&recv[2])[4] = src[dex_src + dex_w + (b_op_num << 1)];
            ((double*)&recv[3])[4] = src[dex_src + dex_w + b_op_num * 3];
            dex_w += Wsrc;               
            ((double*)&recv[0])[6] = src[dex_src + dex_w];
            ((double*)&recv[1])[6] = src[dex_src + dex_w + b_op_num];
            ((double*)&recv[2])[6] = src[dex_src + dex_w + (b_op_num << 1)];
            ((double*)&recv[3])[6] = src[dex_src + dex_w + b_op_num * 3];
#endif

            // 1st 
            res = _mm256_add_ps(_mm256_add_ps(recv[0], recv[1]), 
                _mm256_add_ps(recv[2], recv[3]));
            dex_w = j * 4;
            _STORE_RES_R2C_1ST_;

            // 2nd
            res = _mm256_sub_ps(recv[0], recv[2]);
            res = _mm256_add_ps(_mm256_sub_ps(_mm256_permute_ps(recv[1], 0b10110001),
                                              _mm256_permute_ps(recv[3], 0b10110001)), res);
            dex_w = j * 4 + 1;
            _STORE_RES_R2C_1ST_;

            // 3rd
            res = _mm256_add_ps(_mm256_sub_ps(recv[0], recv[1]), _mm256_sub_ps(recv[2], recv[3]));
            dex_w = j * 4 + 2;
            _STORE_RES_R2C_1ST_;

            // 4th
            res = _mm256_sub_ps(recv[0], recv[2]);
            res = _mm256_add_ps(_mm256_sub_ps(_mm256_permute_ps(recv[3], 0b10110001),
                                              _mm256_permute_ps(recv[1], 0b10110001)), res);
            dex_w = j * 4 + 3;
            _STORE_RES_R2C_1ST_;
        }
        dex += (procW << 2);
        dex_src += (Wsrc << 2);
    }
}



_THREAD_CALL_ void 
decx::signal::CPUK::_IFFT2D_R4_fp32_C2C_first_ST_vec4col(const double* __restrict src,
                                                  double* __restrict dst, 
                                                  const uint signal_W,
                                                  const uint procW,
                                                  const uint procH)
{
    __m256d recv[4];
    __m256 res, tmp0, tmp1;

    size_t dex = 0;
    const uint b_op_num = signal_W / 4;
    uint dex_w = 0;

    for (int i = 0; i < procH; ++i) {
        for (int j = 0; j < b_op_num; ++j) {
            dex_w = j;
#ifdef _MSC_VER
            recv[0].m256d_f64[0] = src[dex + dex_w];
            recv[1].m256d_f64[0] = src[dex + dex_w + b_op_num];
            recv[2].m256d_f64[0] = src[dex + dex_w + (b_op_num << 1)];
            recv[3].m256d_f64[0] = src[dex + dex_w + b_op_num * 3];
            dex_w += procW;
            recv[0].m256d_f64[1] = src[dex + dex_w];
            recv[1].m256d_f64[1] = src[dex + dex_w + b_op_num];
            recv[2].m256d_f64[1] = src[dex + dex_w + (b_op_num << 1)];
            recv[3].m256d_f64[1] = src[dex + dex_w + b_op_num * 3];
            dex_w += procW;
            recv[0].m256d_f64[2] = src[dex + dex_w];
            recv[1].m256d_f64[2] = src[dex + dex_w + b_op_num];
            recv[2].m256d_f64[2] = src[dex + dex_w + (b_op_num << 1)];
            recv[3].m256d_f64[2] = src[dex + dex_w + b_op_num * 3];
            dex_w += procW;
            recv[0].m256d_f64[3] = src[dex + dex_w];
            recv[1].m256d_f64[3] = src[dex + dex_w + b_op_num];
            recv[2].m256d_f64[3] = src[dex + dex_w + (b_op_num << 1)];
            recv[3].m256d_f64[3] = src[dex + dex_w + b_op_num * 3];
#endif
#ifdef __GNUC__
            ((double*)&recv[0])[0] = src[dex + dex_w];
            ((double*)&recv[1])[0] = src[dex + dex_w + b_op_num];
            ((double*)&recv[2])[0] = src[dex + dex_w + (b_op_num << 1)];
            ((double*)&recv[3])[0] = src[dex + dex_w + b_op_num * 3];
            dex_w += procW;
            ((double*)&recv[0])[1] = src[dex + dex_w];
            ((double*)&recv[1])[1] = src[dex + dex_w + b_op_num];
            ((double*)&recv[2])[1] = src[dex + dex_w + (b_op_num << 1)];
            ((double*)&recv[3])[1] = src[dex + dex_w + b_op_num * 3];
            dex_w += procW;
            ((double*)&recv[0])[2] = src[dex + dex_w];
            ((double*)&recv[1])[2] = src[dex + dex_w + b_op_num];
            ((double*)&recv[2])[2] = src[dex + dex_w + (b_op_num << 1)];
            ((double*)&recv[3])[2] = src[dex + dex_w + b_op_num * 3];
            dex_w += procW;
            ((double*)&recv[0])[3] = src[dex + dex_w];
            ((double*)&recv[1])[3] = src[dex + dex_w + b_op_num];
            ((double*)&recv[2])[3] = src[dex + dex_w + (b_op_num << 1)];
            ((double*)&recv[3])[3] = src[dex + dex_w + b_op_num * 3];
#endif
            // conj
            recv[0] = _mm256_castps_pd(_mm256_mul_ps(_mm256_castpd_ps(recv[0]), _mm256_setr_ps(1, -1, 1, -1, 1, -1, 1, -1)));
            recv[1] = _mm256_castps_pd(_mm256_mul_ps(_mm256_castpd_ps(recv[1]), _mm256_setr_ps(1, -1, 1, -1, 1, -1, 1, -1)));
            recv[2] = _mm256_castps_pd(_mm256_mul_ps(_mm256_castpd_ps(recv[2]), _mm256_setr_ps(1, -1, 1, -1, 1, -1, 1, -1)));
            recv[3] = _mm256_castps_pd(_mm256_mul_ps(_mm256_castpd_ps(recv[3]), _mm256_setr_ps(1, -1, 1, -1, 1, -1, 1, -1)));
            // * /= signal_W
            recv[0] = _mm256_castps_pd(_mm256_div_ps(_mm256_castpd_ps(recv[0]), _mm256_set1_ps((float)signal_W)));
            recv[1] = _mm256_castps_pd(_mm256_div_ps(_mm256_castpd_ps(recv[1]), _mm256_set1_ps((float)signal_W)));
            recv[2] = _mm256_castps_pd(_mm256_div_ps(_mm256_castpd_ps(recv[2]), _mm256_set1_ps((float)signal_W)));
            recv[3] = _mm256_castps_pd(_mm256_div_ps(_mm256_castpd_ps(recv[3]), _mm256_set1_ps((float)signal_W)));

            // 1st 
            res = _mm256_add_ps(_mm256_add_ps(_mm256_castpd_ps(recv[0]), _mm256_castpd_ps(recv[1])),
                                _mm256_add_ps(_mm256_castpd_ps(recv[2]), _mm256_castpd_ps(recv[3])));
            dex_w = j * 4;
            _STORE_RES_R2C_1ST_;

            // 2nd
            res = _mm256_addsub_ps(_mm256_castpd_ps(recv[0]), 
                                   _mm256_permute_ps(_mm256_castpd_ps(recv[1]), 0b10110001) ) ;
            res = _mm256_sub_ps(res, _mm256_castpd_ps(recv[2]));
            res = _mm256_permute_ps(res, 0b10110001);
            res = _mm256_addsub_ps(res, _mm256_castpd_ps(recv[3]));
            res = _mm256_permute_ps(res, 0b10110001);
            dex_w = j * 4 + 1;
            _STORE_RES_R2C_1ST_;

            // 3rd
            res = _mm256_add_ps(_mm256_sub_ps(_mm256_castpd_ps(recv[0]), _mm256_castpd_ps(recv[1])),
                                _mm256_sub_ps(_mm256_castpd_ps(recv[2]), _mm256_castpd_ps(recv[3])));
            dex_w = j * 4 + 2;
            _STORE_RES_R2C_1ST_;

            // 4th
            res = _mm256_sub_ps(_mm256_castpd_ps(recv[0]), _mm256_castpd_ps(recv[2]));
            res = _mm256_permute_ps(res, 0b10110001);
            res = _mm256_addsub_ps(res, _mm256_castpd_ps(recv[1]));
            res = _mm256_permute_ps(res, 0b10110001);
            res = _mm256_addsub_ps(res, _mm256_permute_ps(_mm256_castpd_ps(recv[3]), 0b10110001) );
            dex_w = j * 4 + 3;
            _STORE_RES_R2C_1ST_;
        }
        dex += (procW << 2);
    }
}




_THREAD_CALL_ void 
decx::signal::CPUK::_FFT2D_R4_fp32_C2C_first_ST_vec4col(const double* __restrict src,
                                                  double* __restrict dst, 
                                                  const uint signal_W,
                                                  const uint procW,
                                                  const uint procH)
{
    __m256d recv[4];
    __m256 res, tmp0, tmp1;

    size_t dex = 0;
    const uint b_op_num = signal_W / 4;
    uint dex_w = 0;

    for (int i = 0; i < procH; ++i) {
        for (int j = 0; j < b_op_num; ++j) {
            dex_w = j;
#ifdef _MSC_VER
            recv[0].m256d_f64[0] = src[dex + dex_w];
            recv[1].m256d_f64[0] = src[dex + dex_w + b_op_num];
            recv[2].m256d_f64[0] = src[dex + dex_w + (b_op_num << 1)];
            recv[3].m256d_f64[0] = src[dex + dex_w + b_op_num * 3];
            dex_w += procW;
            recv[0].m256d_f64[1] = src[dex + dex_w];
            recv[1].m256d_f64[1] = src[dex + dex_w + b_op_num];
            recv[2].m256d_f64[1] = src[dex + dex_w + (b_op_num << 1)];
            recv[3].m256d_f64[1] = src[dex + dex_w + b_op_num * 3];
            dex_w += procW;
            recv[0].m256d_f64[2] = src[dex + dex_w];
            recv[1].m256d_f64[2] = src[dex + dex_w + b_op_num];
            recv[2].m256d_f64[2] = src[dex + dex_w + (b_op_num << 1)];
            recv[3].m256d_f64[2] = src[dex + dex_w + b_op_num * 3];
            dex_w += procW;
            recv[0].m256d_f64[3] = src[dex + dex_w];
            recv[1].m256d_f64[3] = src[dex + dex_w + b_op_num];
            recv[2].m256d_f64[3] = src[dex + dex_w + (b_op_num << 1)];
            recv[3].m256d_f64[3] = src[dex + dex_w + b_op_num * 3];
#endif
#ifdef __GNUC__
            ((double*)&recv[0])[0] = src[dex + dex_w];
            ((double*)&recv[1])[0] = src[dex + dex_w + b_op_num];
            ((double*)&recv[2])[0] = src[dex + dex_w + (b_op_num << 1)];
            ((double*)&recv[3])[0] = src[dex + dex_w + b_op_num * 3];
            dex_w += procW;
            ((double*)&recv[0])[1] = src[dex + dex_w];
            ((double*)&recv[1])[1] = src[dex + dex_w + b_op_num];
            ((double*)&recv[2])[1] = src[dex + dex_w + (b_op_num << 1)];
            ((double*)&recv[3])[1] = src[dex + dex_w + b_op_num * 3];
            dex_w += procW;
            ((double*)&recv[0])[2] = src[dex + dex_w];
            ((double*)&recv[1])[2] = src[dex + dex_w + b_op_num];
            ((double*)&recv[2])[2] = src[dex + dex_w + (b_op_num << 1)];
            ((double*)&recv[3])[2] = src[dex + dex_w + b_op_num * 3];
            dex_w += procW;
            ((double*)&recv[0])[3] = src[dex + dex_w];
            ((double*)&recv[1])[3] = src[dex + dex_w + b_op_num];
            ((double*)&recv[2])[3] = src[dex + dex_w + (b_op_num << 1)];
            ((double*)&recv[3])[3] = src[dex + dex_w + b_op_num * 3];
#endif

            // 1st 
            res = _mm256_add_ps(_mm256_add_ps(_mm256_castpd_ps(recv[0]), _mm256_castpd_ps(recv[1])),
                                _mm256_add_ps(_mm256_castpd_ps(recv[2]), _mm256_castpd_ps(recv[3])));
            dex_w = j * 4;
            _STORE_RES_R2C_1ST_;

            // 2nd
            res = _mm256_addsub_ps(_mm256_castpd_ps(recv[0]), 
                                   _mm256_permute_ps(_mm256_castpd_ps(recv[1]), 0b10110001) ) ;
            res = _mm256_sub_ps(res, _mm256_castpd_ps(recv[2]));
            res = _mm256_permute_ps(res, 0b10110001);
            res = _mm256_addsub_ps(res, _mm256_castpd_ps(recv[3]));
            res = _mm256_permute_ps(res, 0b10110001);
            dex_w = j * 4 + 1;
            _STORE_RES_R2C_1ST_;

            // 3rd
            res = _mm256_add_ps(_mm256_sub_ps(_mm256_castpd_ps(recv[0]), _mm256_castpd_ps(recv[1])),
                                _mm256_sub_ps(_mm256_castpd_ps(recv[2]), _mm256_castpd_ps(recv[3])));
            dex_w = j * 4 + 2;
            _STORE_RES_R2C_1ST_;

            // 4th
            res = _mm256_sub_ps(_mm256_castpd_ps(recv[0]), _mm256_castpd_ps(recv[2]));
            res = _mm256_permute_ps(res, 0b10110001);
            res = _mm256_addsub_ps(res, _mm256_castpd_ps(recv[1]));
            res = _mm256_permute_ps(res, 0b10110001);
            res = _mm256_addsub_ps(res, _mm256_permute_ps(_mm256_castpd_ps(recv[3]), 0b10110001) );
            dex_w = j * 4 + 3;
            _STORE_RES_R2C_1ST_;
        }
        dex += (procW << 2);
    }
}




_THREAD_CALL_ void 
decx::signal::CPUK::_IFFT2D_R4_fp32_C2C_first_ST_vec4col_L4(const double* __restrict src,
                                                      double* __restrict dst, 
                                                      const uint signal_W,
                                                      const uint procW,
                                                      const uint procH,
                                                      const uint _Left)
{
    __m256d recv[4];
    __m256 res, tmp0, tmp1;

    size_t dex = 0;
    const uint b_op_num = signal_W / 4;
    uint dex_w = 0;

    for (int i = 0; i < procH; ++i) {
        for (int j = 0; j < b_op_num; ++j) {
            dex_w = j;
#ifdef _MSC_VER
            recv[0].m256d_f64[0] = src[dex + dex_w];
            recv[1].m256d_f64[0] = src[dex + dex_w + b_op_num];
            recv[2].m256d_f64[0] = src[dex + dex_w + (b_op_num << 1)];
            recv[3].m256d_f64[0] = src[dex + dex_w + b_op_num * 3];
            dex_w += procW;
            recv[0].m256d_f64[1] = src[dex + dex_w];
            recv[1].m256d_f64[1] = src[dex + dex_w + b_op_num];
            recv[2].m256d_f64[1] = src[dex + dex_w + (b_op_num << 1)];
            recv[3].m256d_f64[1] = src[dex + dex_w + b_op_num * 3];
            dex_w += procW;
            recv[0].m256d_f64[2] = src[dex + dex_w];
            recv[1].m256d_f64[2] = src[dex + dex_w + b_op_num];
            recv[2].m256d_f64[2] = src[dex + dex_w + (b_op_num << 1)];
            recv[3].m256d_f64[2] = src[dex + dex_w + b_op_num * 3];
            dex_w += procW;
            recv[0].m256d_f64[3] = src[dex + dex_w];
            recv[1].m256d_f64[3] = src[dex + dex_w + b_op_num];
            recv[2].m256d_f64[3] = src[dex + dex_w + (b_op_num << 1)];
            recv[3].m256d_f64[3] = src[dex + dex_w + b_op_num * 3];
#endif
#ifdef __GNUC__
            ((double*)&recv[0])[0] = src[dex + dex_w];
            ((double*)&recv[1])[0] = src[dex + dex_w + b_op_num];
            ((double*)&recv[2])[0] = src[dex + dex_w + (b_op_num << 1)];
            ((double*)&recv[3])[0] = src[dex + dex_w + b_op_num * 3];
            dex_w += procW;
            ((double*)&recv[0])[1] = src[dex + dex_w];
            ((double*)&recv[1])[1] = src[dex + dex_w + b_op_num];
            ((double*)&recv[2])[1] = src[dex + dex_w + (b_op_num << 1)];
            ((double*)&recv[3])[1] = src[dex + dex_w + b_op_num * 3];
            dex_w += procW;
            ((double*)&recv[0])[2] = src[dex + dex_w];
            ((double*)&recv[1])[2] = src[dex + dex_w + b_op_num];
            ((double*)&recv[2])[2] = src[dex + dex_w + (b_op_num << 1)];
            ((double*)&recv[3])[2] = src[dex + dex_w + b_op_num * 3];
            dex_w += procW;
            ((double*)&recv[0])[3] = src[dex + dex_w];
            ((double*)&recv[1])[3] = src[dex + dex_w + b_op_num];
            ((double*)&recv[2])[3] = src[dex + dex_w + (b_op_num << 1)];
            ((double*)&recv[3])[3] = src[dex + dex_w + b_op_num * 3];
#endif

            // conj
            recv[0] = _mm256_castps_pd(_mm256_mul_ps(_mm256_castpd_ps(recv[0]), _mm256_setr_ps(1, -1, 1, -1, 1, -1, 1, -1)));
            recv[1] = _mm256_castps_pd(_mm256_mul_ps(_mm256_castpd_ps(recv[1]), _mm256_setr_ps(1, -1, 1, -1, 1, -1, 1, -1)));
            recv[2] = _mm256_castps_pd(_mm256_mul_ps(_mm256_castpd_ps(recv[2]), _mm256_setr_ps(1, -1, 1, -1, 1, -1, 1, -1)));
            recv[3] = _mm256_castps_pd(_mm256_mul_ps(_mm256_castpd_ps(recv[3]), _mm256_setr_ps(1, -1, 1, -1, 1, -1, 1, -1)));
            // * /= signal_W
            recv[0] = _mm256_castps_pd(_mm256_div_ps(_mm256_castpd_ps(recv[0]), _mm256_set1_ps((float)signal_W)));
            recv[1] = _mm256_castps_pd(_mm256_div_ps(_mm256_castpd_ps(recv[1]), _mm256_set1_ps((float)signal_W)));
            recv[2] = _mm256_castps_pd(_mm256_div_ps(_mm256_castpd_ps(recv[2]), _mm256_set1_ps((float)signal_W)));
            recv[3] = _mm256_castps_pd(_mm256_div_ps(_mm256_castpd_ps(recv[3]), _mm256_set1_ps((float)signal_W)));

            // 1st 
            res = _mm256_add_ps(_mm256_add_ps(_mm256_castpd_ps(recv[0]), _mm256_castpd_ps(recv[1])),
                                _mm256_add_ps(_mm256_castpd_ps(recv[2]), _mm256_castpd_ps(recv[3])));
            dex_w = j * 4;
            _STORE_RES_R2C_1ST_;

            // 2nd
            res = _mm256_addsub_ps(_mm256_castpd_ps(recv[0]), 
                                   _mm256_permute_ps(_mm256_castpd_ps(recv[1]), 0b10110001) ) ;
            res = _mm256_sub_ps(res, _mm256_castpd_ps(recv[2]));
            res = _mm256_permute_ps(res, 0b10110001);
            res = _mm256_addsub_ps(res, _mm256_castpd_ps(recv[3]));
            res = _mm256_permute_ps(res, 0b10110001);
            dex_w = j * 4 + 1;
            _STORE_RES_R2C_1ST_;

            // 3rd
            res = _mm256_add_ps(_mm256_sub_ps(_mm256_castpd_ps(recv[0]), _mm256_castpd_ps(recv[1])),
                                _mm256_sub_ps(_mm256_castpd_ps(recv[2]), _mm256_castpd_ps(recv[3])));
            dex_w = j * 4 + 2;
            _STORE_RES_R2C_1ST_;

            // 4th
            res = _mm256_sub_ps(_mm256_castpd_ps(recv[0]), _mm256_castpd_ps(recv[2]));
            res = _mm256_permute_ps(res, 0b10110001);
            res = _mm256_addsub_ps(res, _mm256_castpd_ps(recv[1]));
            res = _mm256_permute_ps(res, 0b10110001);
            res = _mm256_addsub_ps(res, _mm256_permute_ps(_mm256_castpd_ps(recv[3]), 0b10110001) );
            dex_w = j * 4 + 3;
            _STORE_RES_R2C_1ST_;
        }
        dex += (procW << 2);
    }

    for (int j = 0; j < b_op_num; ++j) {
        dex_w = j;
        for (int k = 0; k < _Left; ++k) {
#ifdef _MSC_VER
            recv[0].m256d_f64[k] = src[dex + dex_w];
            recv[1].m256d_f64[k] = src[dex + dex_w + b_op_num];
            recv[2].m256d_f64[k] = src[dex + dex_w + (b_op_num << 1)];
            recv[3].m256d_f64[k] = src[dex + dex_w + b_op_num * 3];
#endif
#ifdef __GNUC__
            ((double*)&recv[0])[k] = src[dex + dex_w];
            ((double*)&recv[1])[k] = src[dex + dex_w + b_op_num];
            ((double*)&recv[2])[k] = src[dex + dex_w + (b_op_num << 1)];
            ((double*)&recv[3])[k] = src[dex + dex_w + b_op_num * 3];
#endif
            dex_w += procW;
        }

        // conj
        recv[0] = _mm256_castps_pd(_mm256_mul_ps(_mm256_castpd_ps(recv[0]), _mm256_setr_ps(1, -1, 1, -1, 1, -1, 1, -1)));
        recv[1] = _mm256_castps_pd(_mm256_mul_ps(_mm256_castpd_ps(recv[1]), _mm256_setr_ps(1, -1, 1, -1, 1, -1, 1, -1)));
        recv[2] = _mm256_castps_pd(_mm256_mul_ps(_mm256_castpd_ps(recv[2]), _mm256_setr_ps(1, -1, 1, -1, 1, -1, 1, -1)));
        recv[3] = _mm256_castps_pd(_mm256_mul_ps(_mm256_castpd_ps(recv[3]), _mm256_setr_ps(1, -1, 1, -1, 1, -1, 1, -1)));
        // * /= signal_W
        recv[0] = _mm256_castps_pd(_mm256_div_ps(_mm256_castpd_ps(recv[0]), _mm256_set1_ps((float)signal_W)));
        recv[1] = _mm256_castps_pd(_mm256_div_ps(_mm256_castpd_ps(recv[1]), _mm256_set1_ps((float)signal_W)));
        recv[2] = _mm256_castps_pd(_mm256_div_ps(_mm256_castpd_ps(recv[2]), _mm256_set1_ps((float)signal_W)));
        recv[3] = _mm256_castps_pd(_mm256_div_ps(_mm256_castpd_ps(recv[3]), _mm256_set1_ps((float)signal_W)));

        // 1st 
        res = _mm256_add_ps(_mm256_add_ps(_mm256_castpd_ps(recv[0]), _mm256_castpd_ps(recv[1])),
            _mm256_add_ps(_mm256_castpd_ps(recv[2]), _mm256_castpd_ps(recv[3])));
        dex_w = j * 4;
        for (int k = 0; k < _Left; ++k) {
#ifdef _MSC_VER
            dst[dex + dex_w] = _mm256_castps_pd(res).m256d_f64[k];  
#endif
#ifdef __GNUC__
            dst[dex + dex_w] = ((double*)&res)[k];  
#endif
            dex_w += procW;
        }

        // 2nd
        res = _mm256_addsub_ps(_mm256_castpd_ps(recv[0]),
            _mm256_permute_ps(_mm256_castpd_ps(recv[1]), 0b10110001));
        res = _mm256_sub_ps(res, _mm256_castpd_ps(recv[2]));
        res = _mm256_permute_ps(res, 0b10110001);
        res = _mm256_addsub_ps(res, _mm256_castpd_ps(recv[3]));
        res = _mm256_permute_ps(res, 0b10110001);
        dex_w = j * 4 + 1;
        for (int k = 0; k < _Left; ++k) {
#ifdef _MSC_VER
            dst[dex + dex_w] = _mm256_castps_pd(res).m256d_f64[k];  
#endif
#ifdef __GNUC__
            dst[dex + dex_w] = ((double*)&res)[k];  
#endif
            dex_w += procW;
        }

        // 3rd
        res = _mm256_add_ps(_mm256_sub_ps(_mm256_castpd_ps(recv[0]), _mm256_castpd_ps(recv[1])),
            _mm256_sub_ps(_mm256_castpd_ps(recv[2]), _mm256_castpd_ps(recv[3])));
        dex_w = j * 4 + 2;
        for (int k = 0; k < _Left; ++k) {
#ifdef _MSC_VER
            dst[dex + dex_w] = _mm256_castps_pd(res).m256d_f64[k];  
#endif
#ifdef __GNUC__
            dst[dex + dex_w] = ((double*)&res)[k];  
#endif
            dex_w += procW;
        }

        // 4th
        res = _mm256_sub_ps(_mm256_castpd_ps(recv[0]), _mm256_castpd_ps(recv[2]));
        res = _mm256_permute_ps(res, 0b10110001);
        res = _mm256_addsub_ps(res, _mm256_castpd_ps(recv[1]));
        res = _mm256_permute_ps(res, 0b10110001);
        res = _mm256_addsub_ps(res, _mm256_permute_ps(_mm256_castpd_ps(recv[3]), 0b10110001));
        dex_w = j * 4 + 3;
        for (int k = 0; k < _Left; ++k) {
#ifdef _MSC_VER
            dst[dex + dex_w] = _mm256_castps_pd(res).m256d_f64[k];  
#endif
#ifdef __GNUC__
            dst[dex + dex_w] = ((double*)&res)[k];  
#endif
            dex_w += procW;
        }
    }
}



_THREAD_CALL_ void 
decx::signal::CPUK::_FFT2D_R4_fp32_C2C_first_ST_vec4col_L4(const double* __restrict src,
                                                      double* __restrict dst, 
                                                      const uint signal_W,
                                                      const uint procW,
                                                      const uint procH,
                                                      const uint _Left)
{
    __m256d recv[4];
    __m256 res, tmp0, tmp1;

    size_t dex = 0;
    const uint b_op_num = signal_W / 4;
    uint dex_w = 0;

    for (int i = 0; i < procH; ++i) {
        for (int j = 0; j < b_op_num; ++j) {
            dex_w = j;
#ifdef _MSC_VER
            recv[0].m256d_f64[0] = src[dex + dex_w];
            recv[1].m256d_f64[0] = src[dex + dex_w + b_op_num];
            recv[2].m256d_f64[0] = src[dex + dex_w + (b_op_num << 1)];
            recv[3].m256d_f64[0] = src[dex + dex_w + b_op_num * 3];
            dex_w += procW;
            recv[0].m256d_f64[1] = src[dex + dex_w];
            recv[1].m256d_f64[1] = src[dex + dex_w + b_op_num];
            recv[2].m256d_f64[1] = src[dex + dex_w + (b_op_num << 1)];
            recv[3].m256d_f64[1] = src[dex + dex_w + b_op_num * 3];
            dex_w += procW;
            recv[0].m256d_f64[2] = src[dex + dex_w];
            recv[1].m256d_f64[2] = src[dex + dex_w + b_op_num];
            recv[2].m256d_f64[2] = src[dex + dex_w + (b_op_num << 1)];
            recv[3].m256d_f64[2] = src[dex + dex_w + b_op_num * 3];
            dex_w += procW;
            recv[0].m256d_f64[3] = src[dex + dex_w];
            recv[1].m256d_f64[3] = src[dex + dex_w + b_op_num];
            recv[2].m256d_f64[3] = src[dex + dex_w + (b_op_num << 1)];
            recv[3].m256d_f64[3] = src[dex + dex_w + b_op_num * 3];
#endif
#ifdef __GNUC__
            ((double*)&recv[0])[0] = src[dex + dex_w];
            ((double*)&recv[1])[0] = src[dex + dex_w + b_op_num];
            ((double*)&recv[2])[0] = src[dex + dex_w + (b_op_num << 1)];
            ((double*)&recv[3])[0] = src[dex + dex_w + b_op_num * 3];
            dex_w += procW;
            ((double*)&recv[0])[1] = src[dex + dex_w];
            ((double*)&recv[1])[1] = src[dex + dex_w + b_op_num];
            ((double*)&recv[2])[1] = src[dex + dex_w + (b_op_num << 1)];
            ((double*)&recv[3])[1] = src[dex + dex_w + b_op_num * 3];
            dex_w += procW;
            ((double*)&recv[0])[2] = src[dex + dex_w];
            ((double*)&recv[1])[2] = src[dex + dex_w + b_op_num];
            ((double*)&recv[2])[2] = src[dex + dex_w + (b_op_num << 1)];
            ((double*)&recv[3])[2] = src[dex + dex_w + b_op_num * 3];
            dex_w += procW;
            ((double*)&recv[0])[3] = src[dex + dex_w];
            ((double*)&recv[1])[3] = src[dex + dex_w + b_op_num];
            ((double*)&recv[2])[3] = src[dex + dex_w + (b_op_num << 1)];
            ((double*)&recv[3])[3] = src[dex + dex_w + b_op_num * 3];
#endif

            // 1st 
            res = _mm256_add_ps(_mm256_add_ps(_mm256_castpd_ps(recv[0]), _mm256_castpd_ps(recv[1])),
                                _mm256_add_ps(_mm256_castpd_ps(recv[2]), _mm256_castpd_ps(recv[3])));
            dex_w = j * 4;
            _STORE_RES_R2C_1ST_;

            // 2nd
            res = _mm256_addsub_ps(_mm256_castpd_ps(recv[0]), 
                                   _mm256_permute_ps(_mm256_castpd_ps(recv[1]), 0b10110001) ) ;
            res = _mm256_sub_ps(res, _mm256_castpd_ps(recv[2]));
            res = _mm256_permute_ps(res, 0b10110001);
            res = _mm256_addsub_ps(res, _mm256_castpd_ps(recv[3]));
            res = _mm256_permute_ps(res, 0b10110001);
            dex_w = j * 4 + 1;
            _STORE_RES_R2C_1ST_;

            // 3rd
            res = _mm256_add_ps(_mm256_sub_ps(_mm256_castpd_ps(recv[0]), _mm256_castpd_ps(recv[1])),
                                _mm256_sub_ps(_mm256_castpd_ps(recv[2]), _mm256_castpd_ps(recv[3])));
            dex_w = j * 4 + 2;
            _STORE_RES_R2C_1ST_;

            // 4th
            res = _mm256_sub_ps(_mm256_castpd_ps(recv[0]), _mm256_castpd_ps(recv[2]));
            res = _mm256_permute_ps(res, 0b10110001);
            res = _mm256_addsub_ps(res, _mm256_castpd_ps(recv[1]));
            res = _mm256_permute_ps(res, 0b10110001);
            res = _mm256_addsub_ps(res, _mm256_permute_ps(_mm256_castpd_ps(recv[3]), 0b10110001) );
            dex_w = j * 4 + 3;
            _STORE_RES_R2C_1ST_;
        }
        dex += (procW << 2);
    }

    for (int j = 0; j < b_op_num; ++j) {
        dex_w = j;
        for (int k = 0; k < _Left; ++k) {
#ifdef _MSC_VER
            recv[0].m256d_f64[k] = src[dex + dex_w];
            recv[1].m256d_f64[k] = src[dex + dex_w + b_op_num];
            recv[2].m256d_f64[k] = src[dex + dex_w + (b_op_num << 1)];
            recv[3].m256d_f64[k] = src[dex + dex_w + b_op_num * 3];
#endif
#ifdef __GNUC__
            ((double*)&recv[0])[k] = src[dex + dex_w];
            ((double*)&recv[1])[k] = src[dex + dex_w + b_op_num];
            ((double*)&recv[2])[k] = src[dex + dex_w + (b_op_num << 1)];
            ((double*)&recv[3])[k] = src[dex + dex_w + b_op_num * 3];
#endif
            dex_w += procW;
        }

        // 1st 
        res = _mm256_add_ps(_mm256_add_ps(_mm256_castpd_ps(recv[0]), _mm256_castpd_ps(recv[1])),
            _mm256_add_ps(_mm256_castpd_ps(recv[2]), _mm256_castpd_ps(recv[3])));
        dex_w = j * 4;
        for (int k = 0; k < _Left; ++k) {
#ifdef _MSC_VER
            dst[dex + dex_w] = _mm256_castps_pd(res).m256d_f64[k]; 
#endif
#ifdef __GNUC__
            dst[dex + dex_w] = ((double*)&res)[k]; 
#endif
            dex_w += procW;
        }

        // 2nd
        res = _mm256_addsub_ps(_mm256_castpd_ps(recv[0]),
            _mm256_permute_ps(_mm256_castpd_ps(recv[1]), 0b10110001));
        res = _mm256_sub_ps(res, _mm256_castpd_ps(recv[2]));
        res = _mm256_permute_ps(res, 0b10110001);
        res = _mm256_addsub_ps(res, _mm256_castpd_ps(recv[3]));
        res = _mm256_permute_ps(res, 0b10110001);
        dex_w = j * 4 + 1;
        for (int k = 0; k < _Left; ++k) {
#ifdef _MSC_VER
            dst[dex + dex_w] = _mm256_castps_pd(res).m256d_f64[k];
#endif 
#ifdef __GNUC__
            dst[dex + dex_w] = ((double*)&res)[k];
#endif
            dex_w += procW;
        }

        // 3rd
        res = _mm256_add_ps(_mm256_sub_ps(_mm256_castpd_ps(recv[0]), _mm256_castpd_ps(recv[1])),
            _mm256_sub_ps(_mm256_castpd_ps(recv[2]), _mm256_castpd_ps(recv[3])));
        dex_w = j * 4 + 2;
        for (int k = 0; k < _Left; ++k) {
#ifdef _MSC_VER
            dst[dex + dex_w] = _mm256_castps_pd(res).m256d_f64[k];
#endif 
#ifdef __GNUC__
            dst[dex + dex_w] = ((double*)&res)[k];
#endif
            dex_w += procW;
        }

        // 4th
        res = _mm256_sub_ps(_mm256_castpd_ps(recv[0]), _mm256_castpd_ps(recv[2]));
        res = _mm256_permute_ps(res, 0b10110001);
        res = _mm256_addsub_ps(res, _mm256_castpd_ps(recv[1]));
        res = _mm256_permute_ps(res, 0b10110001);
        res = _mm256_addsub_ps(res, _mm256_permute_ps(_mm256_castpd_ps(recv[3]), 0b10110001));
        dex_w = j * 4 + 3;
        for (int k = 0; k < _Left; ++k) {
#ifdef _MSC_VER
            dst[dex + dex_w] = _mm256_castps_pd(res).m256d_f64[k];
#endif 
#ifdef __GNUC__
            dst[dex + dex_w] = ((double*)&res)[k];
#endif
            dex_w += procW;
        }
    }
}



_THREAD_CALL_ void 
decx::signal::CPUK::_FFT2D_R4_fp32_R2C_first_ST_vec4col_L4(const float* __restrict src,
                                                  double* __restrict dst, 
                                                  const uint signal_W,
                                                  const uint Wsrc,
                                                  const uint procW,
                                                  const uint procH,
                                                  const uint _Left)
{
    __m256 recv[4];
    __m256 res, tmp0, tmp1;

    size_t dex = 0, dex_src = 0;
    const uint b_op_num = signal_W / 4;
    uint dex_w = 0;

    for (int i = 0; i < procH; ++i) {
        for (int j = 0; j < b_op_num; ++j) {
            dex_w = j;
#ifdef _MSC_VER
            recv[0].m256_f32[0] = src[dex_src + dex_w];
            recv[1].m256_f32[0] = src[dex_src + dex_w + b_op_num];
            recv[2].m256_f32[0] = src[dex_src + dex_w + (b_op_num << 1)];
            recv[3].m256_f32[0] = src[dex_src + dex_w + b_op_num * 3];
            dex_w += Wsrc;               
            recv[0].m256_f32[2] = src[dex_src + dex_w];
            recv[1].m256_f32[2] = src[dex_src + dex_w + b_op_num];
            recv[2].m256_f32[2] = src[dex_src + dex_w + (b_op_num << 1)];
            recv[3].m256_f32[2] = src[dex_src + dex_w + b_op_num * 3];
            dex_w += Wsrc;               
            recv[0].m256_f32[4] = src[dex_src + dex_w];
            recv[1].m256_f32[4] = src[dex_src + dex_w + b_op_num];
            recv[2].m256_f32[4] = src[dex_src + dex_w + (b_op_num << 1)];
            recv[3].m256_f32[4] = src[dex_src + dex_w + b_op_num * 3];
            dex_w += Wsrc;               
            recv[0].m256_f32[6] = src[dex_src + dex_w];
            recv[1].m256_f32[6] = src[dex_src + dex_w + b_op_num];
            recv[2].m256_f32[6] = src[dex_src + dex_w + (b_op_num << 1)];
            recv[3].m256_f32[6] = src[dex_src + dex_w + b_op_num * 3];
#endif
#ifdef __GNUC__
            ((float*)&recv[0])[0] = src[dex_src + dex_w];
            ((float*)&recv[1])[0] = src[dex_src + dex_w + b_op_num];
            ((float*)&recv[2])[0] = src[dex_src + dex_w + (b_op_num << 1)];
            ((float*)&recv[3])[0] = src[dex_src + dex_w + b_op_num * 3];
            dex_w += Wsrc;               
            ((float*)&recv[0])[2] = src[dex_src + dex_w];
            ((float*)&recv[1])[2] = src[dex_src + dex_w + b_op_num];
            ((float*)&recv[2])[2] = src[dex_src + dex_w + (b_op_num << 1)];
            ((float*)&recv[3])[2] = src[dex_src + dex_w + b_op_num * 3];
            dex_w += Wsrc;               
            ((float*)&recv[0])[4] = src[dex_src + dex_w];
            ((float*)&recv[1])[4] = src[dex_src + dex_w + b_op_num];
            ((float*)&recv[2])[4] = src[dex_src + dex_w + (b_op_num << 1)];
            ((float*)&recv[3])[4] = src[dex_src + dex_w + b_op_num * 3];
            dex_w += Wsrc;               
            ((float*)&recv[0])[6] = src[dex_src + dex_w];
            ((float*)&recv[1])[6] = src[dex_src + dex_w + b_op_num];
            ((float*)&recv[2])[6] = src[dex_src + dex_w + (b_op_num << 1)];
            ((float*)&recv[3])[6] = src[dex_src + dex_w + b_op_num * 3];
#endif

            // 1st 
            res = _mm256_add_ps(_mm256_add_ps(recv[0], recv[1]), 
                _mm256_add_ps(recv[2], recv[3]));
            dex_w = j * 4;
            _STORE_RES_R2C_1ST_;

            // 2nd
            res = _mm256_sub_ps(recv[0], recv[2]);
            res = _mm256_add_ps(_mm256_sub_ps(_mm256_permute_ps(recv[1], 0b10110001),
                                              _mm256_permute_ps(recv[3], 0b10110001)), res);
            dex_w = j * 4 + 1;
            _STORE_RES_R2C_1ST_;

            // 3rd
            res = _mm256_add_ps(_mm256_sub_ps(recv[0], recv[1]), _mm256_sub_ps(recv[2], recv[3]));
            dex_w = j * 4 + 2;
            _STORE_RES_R2C_1ST_;

            // 4th
            res = _mm256_sub_ps(recv[0], recv[2]);
            res = _mm256_add_ps(_mm256_sub_ps(_mm256_permute_ps(recv[3], 0b10110001),
                                              _mm256_permute_ps(recv[1], 0b10110001)), res);
            dex_w = j * 4 + 3;
            _STORE_RES_R2C_1ST_;
        }
        dex += (procW << 2);
        dex_src += (Wsrc << 2);
    }

    for (int j = 0; j < b_op_num; ++j) {
        dex_w = j;
        for (int k = 0; k < _Left; ++k) {
#ifdef _MSC_VER
            recv[0].m256_f32[2 * k] = src[dex_src + dex_w];
            recv[1].m256_f32[2 * k] = src[dex_src + dex_w + b_op_num];
            recv[2].m256_f32[2 * k] = src[dex_src + dex_w + (b_op_num << 1)];
            recv[3].m256_f32[2 * k] = src[dex_src + dex_w + b_op_num * 3];
#endif
#ifdef __GNUC__
            ((float*)&recv[0])[2 * k] = src[dex_src + dex_w];
            ((float*)&recv[1])[2 * k] = src[dex_src + dex_w + b_op_num];
            ((float*)&recv[2])[2 * k] = src[dex_src + dex_w + (b_op_num << 1)];
            ((float*)&recv[3])[2 * k] = src[dex_src + dex_w + b_op_num * 3];
#endif
            dex_w += Wsrc;
        }
        
        // 1st 
        res = _mm256_add_ps(_mm256_add_ps(recv[0], recv[1]),
            _mm256_add_ps(recv[2], recv[3]));
        dex_w = j * 4;
        for (int k = 0; k < _Left; ++k) {
#ifdef _MSC_VER
            dst[dex + dex_w] = _mm256_castps_pd(res).m256d_f64[k];
#endif
#ifdef __GNUC__
            dst[dex + dex_w] = ((double*)&res)[k];
#endif
            dex_w += procW;
        }

        // 2nd
        res = _mm256_sub_ps(recv[0], recv[2]);
        res = _mm256_add_ps(_mm256_sub_ps(_mm256_permute_ps(recv[1], 0b10110001),
            _mm256_permute_ps(recv[3], 0b10110001)), res);
        dex_w = j * 4 + 1;
        for (int k = 0; k < _Left; ++k) {
#ifdef _MSC_VER
            dst[dex + dex_w] = _mm256_castps_pd(res).m256d_f64[k];
#endif
#ifdef __GNUC__
            dst[dex + dex_w] = ((double*)&res)[k];
#endif
            dex_w += procW;
        }

        // 3rd
        res = _mm256_add_ps(_mm256_sub_ps(recv[0], recv[1]), _mm256_sub_ps(recv[2], recv[3]));
        dex_w = j * 4 + 2;
        for (int k = 0; k < _Left; ++k) {
#ifdef _MSC_VER
            dst[dex + dex_w] = _mm256_castps_pd(res).m256d_f64[k];
#endif
#ifdef __GNUC__
            dst[dex + dex_w] = ((double*)&res)[k];
#endif
            dex_w += procW;
        }

        // 4th
        res = _mm256_sub_ps(recv[0], recv[2]);
        res = _mm256_add_ps(_mm256_sub_ps(_mm256_permute_ps(recv[3], 0b10110001),
            _mm256_permute_ps(recv[1], 0b10110001)), res);
        dex_w = j * 4 + 3;
        for (int k = 0; k < _Left; ++k) {
#ifdef _MSC_VER
            dst[dex + dex_w] = _mm256_castps_pd(res).m256d_f64[k];
#endif
#ifdef __GNUC__
            dst[dex + dex_w] = ((double*)&res)[k];
#endif
            dex_w += procW;
        }
    }
}




_THREAD_CALL_ void 
decx::signal::CPUK::_FFT2D_R4_fp32_C2C_ST_vec4col(const double* __restrict src,
                                            double* __restrict dst,
                                            const uint warp_proc_len,       // in element
                                            const uint signal_W,            // in element
                                            const uint procW,
                                            const uint procH)
{
    __m256d recv[4], tmp, res;
    de::CPf W;
    size_t dex = 0;     // in vec4
    uint dex_w = 0;

    uint num_of_Bcalc_in_warp = warp_proc_len / 4,
        warp_loc_id;
    const size_t num_total_Bops = signal_W / 4;

    for (int i = 0; i < procH; ++i) {
        for (int j = 0; j < num_total_Bops; ++j) {
            dex_w = j;
#ifdef _MSC_VER
            recv[0].m256d_f64[0] = src[dex + dex_w];    
            recv[1].m256d_f64[0] = src[dex + dex_w + num_total_Bops];
            recv[2].m256d_f64[0] = src[dex + dex_w + num_total_Bops * 2];
            recv[3].m256d_f64[0] = src[dex + dex_w + num_total_Bops * 3];
            dex_w += procW;
            recv[0].m256d_f64[1] = src[dex + dex_w];    
            recv[1].m256d_f64[1] = src[dex + dex_w + num_total_Bops];
            recv[2].m256d_f64[1] = src[dex + dex_w + num_total_Bops * 2];
            recv[3].m256d_f64[1] = src[dex + dex_w + num_total_Bops * 3];
            dex_w += procW;
            recv[0].m256d_f64[2] = src[dex + dex_w];    
            recv[1].m256d_f64[2] = src[dex + dex_w + num_total_Bops];
            recv[2].m256d_f64[2] = src[dex + dex_w + num_total_Bops * 2];
            recv[3].m256d_f64[2] = src[dex + dex_w + num_total_Bops * 3];
            dex_w += procW;
            recv[0].m256d_f64[3] = src[dex + dex_w];    
            recv[1].m256d_f64[3] = src[dex + dex_w + num_total_Bops];
            recv[2].m256d_f64[3] = src[dex + dex_w + num_total_Bops * 2];
            recv[3].m256d_f64[3] = src[dex + dex_w + num_total_Bops * 3];
#endif
#ifdef __GNUC__
            ((double*)&recv[0])[0] = src[dex + dex_w];    
            ((double*)&recv[1])[0] = src[dex + dex_w + num_total_Bops];
            ((double*)&recv[2])[0] = src[dex + dex_w + num_total_Bops * 2];
            ((double*)&recv[3])[0] = src[dex + dex_w + num_total_Bops * 3];
            dex_w += procW;
            ((double*)&recv[0])[1] = src[dex + dex_w];    
            ((double*)&recv[1])[1] = src[dex + dex_w + num_total_Bops];
            ((double*)&recv[2])[1] = src[dex + dex_w + num_total_Bops * 2];
            ((double*)&recv[3])[1] = src[dex + dex_w + num_total_Bops * 3];
            dex_w += procW;
            ((double*)&recv[0])[2] = src[dex + dex_w];    
            ((double*)&recv[1])[2] = src[dex + dex_w + num_total_Bops];
            ((double*)&recv[2])[2] = src[dex + dex_w + num_total_Bops * 2];
            ((double*)&recv[3])[2] = src[dex + dex_w + num_total_Bops * 3];
            dex_w += procW;
            ((double*)&recv[0])[3] = src[dex + dex_w];    
            ((double*)&recv[1])[3] = src[dex + dex_w + num_total_Bops];
            ((double*)&recv[2])[3] = src[dex + dex_w + num_total_Bops * 2];
            ((double*)&recv[3])[3] = src[dex + dex_w + num_total_Bops * 3];
#endif

            warp_loc_id = j % num_of_Bcalc_in_warp;

            W.construct_with_phase(Two_Pi * (float)warp_loc_id / (float)warp_proc_len);
            recv[1] = _mm256_castps_pd(decx::signal::CPUK::_cp4_mul_cp1_fp32(_mm256_castpd_ps(recv[1]), W));

            W.construct_with_phase(Four_Pi * (float)warp_loc_id / (float)warp_proc_len);
            recv[2] = _mm256_castps_pd(decx::signal::CPUK::_cp4_mul_cp1_fp32(_mm256_castpd_ps(recv[2]), W));

            W.construct_with_phase(Six_Pi * (float)warp_loc_id / (float)warp_proc_len);
            recv[3] = _mm256_castps_pd(decx::signal::CPUK::_cp4_mul_cp1_fp32(_mm256_castpd_ps(recv[3]), W));

            // 1st port out
            res = _mm256_castps_pd(_mm256_add_ps(_mm256_add_ps(_mm256_castpd_ps(recv[0]), _mm256_castpd_ps(recv[1])),
                                                 _mm256_add_ps(_mm256_castpd_ps(recv[2]), _mm256_castpd_ps(recv[3]))));

            dex_w = (j / num_of_Bcalc_in_warp) * warp_proc_len + warp_loc_id;
            _STORE_RES_C2C_MID_;

            // 2nd port out
            res = _mm256_castps_pd(_mm256_addsub_ps(_mm256_castpd_ps(recv[0]), 
                                                    _mm256_permute_ps(_mm256_castpd_ps(recv[1]), 0b10110001) ) );
            res = _mm256_castps_pd(_mm256_sub_ps(_mm256_castpd_ps(res), _mm256_castpd_ps(recv[2])));
            res = _mm256_castps_pd(_mm256_permute_ps(_mm256_castpd_ps(res), 0b10110001));
            res = _mm256_castps_pd(_mm256_addsub_ps(_mm256_castpd_ps(res), _mm256_castpd_ps(recv[3])));
            res = _mm256_castps_pd(_mm256_permute_ps(_mm256_castpd_ps(res), 0b10110001));
            dex_w = (j / num_of_Bcalc_in_warp) * warp_proc_len + warp_loc_id + num_of_Bcalc_in_warp;
            _STORE_RES_C2C_MID_;

            res = _mm256_castps_pd(_mm256_add_ps(_mm256_sub_ps(_mm256_castpd_ps(recv[0]), _mm256_castpd_ps(recv[1])),
                                                 _mm256_sub_ps(_mm256_castpd_ps(recv[2]), _mm256_castpd_ps(recv[3]))));
            dex_w = (j / num_of_Bcalc_in_warp) * warp_proc_len + warp_loc_id + num_of_Bcalc_in_warp * 2;
            _STORE_RES_C2C_MID_;

            res = _mm256_castps_pd(_mm256_sub_ps(_mm256_castpd_ps(recv[0]), _mm256_castpd_ps(recv[2])));
            res = _mm256_castps_pd(_mm256_permute_ps(_mm256_castpd_ps(res), 0b10110001));
            res = _mm256_castps_pd(_mm256_addsub_ps(_mm256_castpd_ps(res), _mm256_castpd_ps(recv[1])));
            res = _mm256_castps_pd(_mm256_permute_ps(_mm256_castpd_ps(res), 0b10110001));
            res = _mm256_castps_pd(_mm256_addsub_ps(_mm256_castpd_ps(res), 
                                                    _mm256_permute_ps(_mm256_castpd_ps(recv[3]), 0b10110001) ) );
            dex_w = (j / num_of_Bcalc_in_warp) * warp_proc_len + warp_loc_id + num_of_Bcalc_in_warp * 3;
            _STORE_RES_C2C_MID_;
        }
        dex += (procW << 2);
    }
}




_THREAD_CALL_ void 
decx::signal::CPUK::_FFT2D_R4_fp32_C2C_ST_vec4col_L4(const double* __restrict src,
                                               double* __restrict dst,
                                               const uint warp_proc_len,       // in element
                                               const uint signal_W,            // in element
                                               const uint procW,
                                               const uint procH,
                                               const uint _Left)
{
    __m256d recv[4], tmp, res;
    de::CPf W;
    size_t dex = 0;     // in vec4
    uint dex_w = 0;

    uint num_of_Bcalc_in_warp = warp_proc_len / 4,
        warp_loc_id;
    const size_t num_total_Bops = signal_W / 4;

    for (int i = 0; i < procH; ++i) {
        for (int j = 0; j < num_total_Bops; ++j) {
            dex_w = j;
#ifdef _MSC_VER
            recv[0].m256d_f64[0] = src[dex + dex_w];    
            recv[1].m256d_f64[0] = src[dex + dex_w + num_total_Bops];
            recv[2].m256d_f64[0] = src[dex + dex_w + num_total_Bops * 2];
            recv[3].m256d_f64[0] = src[dex + dex_w + num_total_Bops * 3];
            dex_w += procW;
            recv[0].m256d_f64[1] = src[dex + dex_w];    
            recv[1].m256d_f64[1] = src[dex + dex_w + num_total_Bops];
            recv[2].m256d_f64[1] = src[dex + dex_w + num_total_Bops * 2];
            recv[3].m256d_f64[1] = src[dex + dex_w + num_total_Bops * 3];
            dex_w += procW;
            recv[0].m256d_f64[2] = src[dex + dex_w];    
            recv[1].m256d_f64[2] = src[dex + dex_w + num_total_Bops];
            recv[2].m256d_f64[2] = src[dex + dex_w + num_total_Bops * 2];
            recv[3].m256d_f64[2] = src[dex + dex_w + num_total_Bops * 3];
            dex_w += procW;
            recv[0].m256d_f64[3] = src[dex + dex_w];    
            recv[1].m256d_f64[3] = src[dex + dex_w + num_total_Bops];
            recv[2].m256d_f64[3] = src[dex + dex_w + num_total_Bops * 2];
            recv[3].m256d_f64[3] = src[dex + dex_w + num_total_Bops * 3];
#endif

            warp_loc_id = j % num_of_Bcalc_in_warp;

            W.construct_with_phase(Two_Pi * (float)warp_loc_id / (float)warp_proc_len);
            recv[1] = _mm256_castps_pd(decx::signal::CPUK::_cp4_mul_cp1_fp32(_mm256_castpd_ps(recv[1]), W));

            W.construct_with_phase(Four_Pi * (float)warp_loc_id / (float)warp_proc_len);
            recv[2] = _mm256_castps_pd(decx::signal::CPUK::_cp4_mul_cp1_fp32(_mm256_castpd_ps(recv[2]), W));

            W.construct_with_phase(Six_Pi * (float)warp_loc_id / (float)warp_proc_len);
            recv[3] = _mm256_castps_pd(decx::signal::CPUK::_cp4_mul_cp1_fp32(_mm256_castpd_ps(recv[3]), W));

            // 1st port out
            res = _mm256_castps_pd(_mm256_add_ps(_mm256_add_ps(_mm256_castpd_ps(recv[0]), _mm256_castpd_ps(recv[1])),
                                                 _mm256_add_ps(_mm256_castpd_ps(recv[2]), _mm256_castpd_ps(recv[3]))));

            dex_w = (j / num_of_Bcalc_in_warp) * warp_proc_len + warp_loc_id;
            _STORE_RES_C2C_MID_;

            // 2nd port out
            res = _mm256_castps_pd(_mm256_addsub_ps(_mm256_castpd_ps(recv[0]), 
                                                    _mm256_permute_ps(_mm256_castpd_ps(recv[1]), 0b10110001) ) );
            res = _mm256_castps_pd(_mm256_sub_ps(_mm256_castpd_ps(res), _mm256_castpd_ps(recv[2])));
            res = _mm256_castps_pd(_mm256_permute_ps(_mm256_castpd_ps(res), 0b10110001));
            res = _mm256_castps_pd(_mm256_addsub_ps(_mm256_castpd_ps(res), _mm256_castpd_ps(recv[3])));
            res = _mm256_castps_pd(_mm256_permute_ps(_mm256_castpd_ps(res), 0b10110001));
            dex_w = (j / num_of_Bcalc_in_warp) * warp_proc_len + warp_loc_id + num_of_Bcalc_in_warp;
            _STORE_RES_C2C_MID_;

            res = _mm256_castps_pd(_mm256_add_ps(_mm256_sub_ps(_mm256_castpd_ps(recv[0]), _mm256_castpd_ps(recv[1])),
                                                 _mm256_sub_ps(_mm256_castpd_ps(recv[2]), _mm256_castpd_ps(recv[3]))));
            dex_w = (j / num_of_Bcalc_in_warp) * warp_proc_len + warp_loc_id + num_of_Bcalc_in_warp * 2;
            _STORE_RES_C2C_MID_;

            res = _mm256_castps_pd(_mm256_sub_ps(_mm256_castpd_ps(recv[0]), _mm256_castpd_ps(recv[2])));
            res = _mm256_castps_pd(_mm256_permute_ps(_mm256_castpd_ps(res), 0b10110001));
            res = _mm256_castps_pd(_mm256_addsub_ps(_mm256_castpd_ps(res), _mm256_castpd_ps(recv[1])));
            res = _mm256_castps_pd(_mm256_permute_ps(_mm256_castpd_ps(res), 0b10110001));
            res = _mm256_castps_pd(_mm256_addsub_ps(_mm256_castpd_ps(res), 
                                                    _mm256_permute_ps(_mm256_castpd_ps(recv[3]), 0b10110001) ) );
            dex_w = (j / num_of_Bcalc_in_warp) * warp_proc_len + warp_loc_id + num_of_Bcalc_in_warp * 3;
            _STORE_RES_C2C_MID_;
        }
        dex += (procW << 2);
    }

    for (int j = 0; j < num_total_Bops; ++j) {
            dex_w = j;
            for (int k = 0; k < _Left; ++k) {
#ifdef _MSC_VER
                recv[0].m256d_f64[k] = src[dex + dex_w];
                recv[1].m256d_f64[k] = src[dex + dex_w + num_total_Bops];
                recv[2].m256d_f64[k] = src[dex + dex_w + num_total_Bops * 2];
                recv[3].m256d_f64[k] = src[dex + dex_w + num_total_Bops * 3];
#endif
#ifdef __GNUC__
                ((double*)&recv[0])[k] = src[dex + dex_w];
                ((double*)&recv[1])[k] = src[dex + dex_w + num_total_Bops];
                ((double*)&recv[2])[k] = src[dex + dex_w + num_total_Bops * 2];
                ((double*)&recv[3])[k] = src[dex + dex_w + num_total_Bops * 3];
#endif
                dex_w += procW;
            }
            
            warp_loc_id = j % num_of_Bcalc_in_warp;

            W.construct_with_phase(Two_Pi * (float)warp_loc_id / (float)warp_proc_len);
            recv[1] = _mm256_castps_pd(decx::signal::CPUK::_cp4_mul_cp1_fp32(_mm256_castpd_ps(recv[1]), W));

            W.construct_with_phase(Four_Pi * (float)warp_loc_id / (float)warp_proc_len);
            recv[2] = _mm256_castps_pd(decx::signal::CPUK::_cp4_mul_cp1_fp32(_mm256_castpd_ps(recv[2]), W));

            W.construct_with_phase(Six_Pi * (float)warp_loc_id / (float)warp_proc_len);
            recv[3] = _mm256_castps_pd(decx::signal::CPUK::_cp4_mul_cp1_fp32(_mm256_castpd_ps(recv[3]), W));

            // 1st port out
            res = _mm256_castps_pd(_mm256_add_ps(_mm256_add_ps(_mm256_castpd_ps(recv[0]), _mm256_castpd_ps(recv[1])),
                                                 _mm256_add_ps(_mm256_castpd_ps(recv[2]), _mm256_castpd_ps(recv[3]))));

            dex_w = (j / num_of_Bcalc_in_warp) * warp_proc_len + warp_loc_id;
            for (int k = 0; k < _Left; ++k) {
#ifdef _MSC_VER
                dst[dex + dex_w] = res.m256d_f64[k];
#endif
#ifdef __GNUC__
                dst[dex + dex_w] = ((double*)&res)[k];
#endif
                dex_w += procW;
            }

            // 2nd port out
            res = _mm256_castps_pd(_mm256_addsub_ps(_mm256_castpd_ps(recv[0]), 
                                                    _mm256_permute_ps(_mm256_castpd_ps(recv[1]), 0b10110001) ) );
            res = _mm256_castps_pd(_mm256_sub_ps(_mm256_castpd_ps(res), _mm256_castpd_ps(recv[2])));
            res = _mm256_castps_pd(_mm256_permute_ps(_mm256_castpd_ps(res), 0b10110001));
            res = _mm256_castps_pd(_mm256_addsub_ps(_mm256_castpd_ps(res), _mm256_castpd_ps(recv[3])));
            res = _mm256_castps_pd(_mm256_permute_ps(_mm256_castpd_ps(res), 0b10110001));
            dex_w = (j / num_of_Bcalc_in_warp) * warp_proc_len + warp_loc_id + num_of_Bcalc_in_warp;
            for (int k = 0; k < _Left; ++k) {
#ifdef _MSC_VER
                dst[dex + dex_w] = res.m256d_f64[k];
#endif
#ifdef __GNUC__
                dst[dex + dex_w] = ((double*)&res)[k];
#endif
                dex_w += procW;
            }

            res = _mm256_castps_pd(_mm256_add_ps(_mm256_sub_ps(_mm256_castpd_ps(recv[0]), _mm256_castpd_ps(recv[1])),
                                                 _mm256_sub_ps(_mm256_castpd_ps(recv[2]), _mm256_castpd_ps(recv[3]))));
            dex_w = (j / num_of_Bcalc_in_warp) * warp_proc_len + warp_loc_id + num_of_Bcalc_in_warp * 2;
            for (int k = 0; k < _Left; ++k) {
#ifdef _MSC_VER
                dst[dex + dex_w] = res.m256d_f64[k];
#endif
#ifdef __GNUC__
                dst[dex + dex_w] = ((double*)&res)[k];
#endif
                dex_w += procW;
            }

            res = _mm256_castps_pd(_mm256_sub_ps(_mm256_castpd_ps(recv[0]), _mm256_castpd_ps(recv[2])));
            res = _mm256_castps_pd(_mm256_permute_ps(_mm256_castpd_ps(res), 0b10110001));
            res = _mm256_castps_pd(_mm256_addsub_ps(_mm256_castpd_ps(res), _mm256_castpd_ps(recv[1])));
            res = _mm256_castps_pd(_mm256_permute_ps(_mm256_castpd_ps(res), 0b10110001));
            res = _mm256_castps_pd(_mm256_addsub_ps(_mm256_castpd_ps(res), 
                                                    _mm256_permute_ps(_mm256_castpd_ps(recv[3]), 0b10110001) ) );
            dex_w = (j / num_of_Bcalc_in_warp) * warp_proc_len + warp_loc_id + num_of_Bcalc_in_warp * 3;
            for (int k = 0; k < _Left; ++k) {
#ifdef _MSC_VER
                dst[dex + dex_w] = res.m256d_f64[k];
#endif
#ifdef __GNUC__
                dst[dex + dex_w] = ((double*)&res)[k];
#endif
                dex_w += procW;
            }
        }
}