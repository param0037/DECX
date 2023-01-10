/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/

#include "FFT1D_radix_5_kernel.h"


_THREAD_FUNCTION_
void decx::signal::CPUK::_FFT1D_R5_fp32_R2C_first_ST(const float* __restrict src, 
    double* __restrict dst, 
    const size_t signal_length,
    const uint2 b_op_dex_range)
{
    float recv[5];
    de::CPf res[5] = { de::CPf(0, 0) };

    size_t dex = 0;

    const size_t Bcalc_num = signal_length / 5;

    for (int i = b_op_dex_range.x; i < b_op_dex_range.y; ++i) {
        dex = i;
        recv[0] = src[dex];
        recv[1] = src[dex + Bcalc_num];
        recv[2] = src[dex + Bcalc_num * 2];
        recv[3] = src[dex + Bcalc_num * 3];
        recv[4] = src[dex + Bcalc_num * 4];

        res[0].real = recv[0] + recv[1] + recv[2] + recv[3] + recv[4];
        res[0].image = 0;

        res[1].real = recv[0] + recv[1] * 0.309017;
        res[1].image = recv[1] * 0.9510565;
        res[1].real += -0.809017 * recv[2];
        res[1].image += 0.5877853 * recv[2];
        res[1].real += -0.809017 * recv[3];
        res[1].image += -0.5877853 * recv[3];
        res[1].real += 0.309017 * recv[4];
        res[1].image += -0.9510565 * recv[4];

        res[2].real = recv[0] + recv[1] * -0.809017;
        res[2].image = recv[1] * 0.5877853;
        res[2].real += 0.309017 * recv[2];
        res[2].image += -0.9510565 * recv[2];
        res[2].real += 0.309017 * recv[3];
        res[2].image += 0.9510565 * recv[3];
        res[2].real += -0.809017 * recv[4];
        res[2].image += -0.5877853 * recv[4];

        res[3].real = recv[0] + recv[1] * -0.809017;
        res[3].image = recv[1] * -0.5877853;
        res[3].real += 0.309017 * recv[2];
        res[3].image += 0.9510565 * recv[2];
        res[3].real += 0.309017 * recv[3];
        res[3].image += -0.9510565 * recv[3];
        res[3].real += -0.809017 * recv[4];
        res[3].image += 0.5877853 * recv[4];

        res[4].real = recv[0] + recv[1] * 0.309017;
        res[4].image = recv[1] * -0.9510565;
        res[4].real += -0.809017 * recv[2];
        res[4].image += -0.5877853 * recv[2];
        res[4].real += -0.809017 * recv[3];
        res[4].image += 0.5877853 * recv[3];
        res[4].real += 0.309017 * recv[4];
        res[4].image += 0.9510565 * recv[4];

        dex = i * 5;
        dst[dex] = *((double*)&res[0]);
        dst[dex + 1] = *((double*)&res[1]);
        dst[dex + 2] = *((double*)&res[2]);
        dst[dex + 3] = *((double*)&res[3]);
        dst[dex + 4] = *((double*)&res[4]);
    }
}



_THREAD_FUNCTION_
void decx::signal::CPUK::_FFT1D_R5_fp32_C2C_first_ST(const double* __restrict src, 
    double* __restrict dst, 
    const size_t signal_length,
    const uint2 b_op_dex_range)
{
    using decx::signal::CPUK::_complex_fma;

    de::CPf recv[5], tmp;
    de::CPf res[5] = { de::CPf(0, 0) };

    size_t dex = 0;

    const size_t Bcalc_num = signal_length / 5;

    for (int i = b_op_dex_range.x; i < b_op_dex_range.y; ++i) {
        dex = i;
        *((double*)&recv[0]) = src[dex];
        *((double*)&recv[1]) = src[dex + Bcalc_num];
        *((double*)&recv[2]) = src[dex + Bcalc_num * 2];
        *((double*)&recv[3]) = src[dex + Bcalc_num * 3];
        *((double*)&recv[4]) = src[dex + Bcalc_num * 4];

        res[0].real = recv[0].real + recv[1].real + recv[2].real + recv[3].real + recv[4].real;
        res[0].image = recv[0].image + recv[1].image + recv[2].image + recv[3].image + recv[4].image;

        tmp = recv[0];
        tmp = _complex_fma(recv[1], de::CPf(0.309017, 0.9510565), tmp);
        tmp = _complex_fma(recv[2], de::CPf(-0.809017, 0.5877853), tmp);
        tmp = _complex_fma(recv[3], de::CPf(-0.809017, -0.5877853), tmp);
        res[1] = _complex_fma(recv[4], de::CPf(0.309017, -0.9510565), tmp);

        tmp = recv[0];
        tmp = _complex_fma(recv[1], de::CPf(-0.809017, 0.5877853), tmp);
        tmp = _complex_fma(recv[2], de::CPf(0.309017, -0.9510565), tmp);
        tmp = _complex_fma(recv[3], de::CPf(0.309017, 0.9510565), tmp);
        res[2] = _complex_fma(recv[4], de::CPf(-0.809017, -0.5877853), tmp);

        tmp = recv[0];
        tmp = _complex_fma(recv[1], de::CPf(-0.809017, -0.5877853), tmp);
        tmp = _complex_fma(recv[2], de::CPf(0.309017, 0.9510565), tmp);
        tmp = _complex_fma(recv[3], de::CPf(0.309017, -0.9510565), tmp);
        res[3] = _complex_fma(recv[4], de::CPf(-0.809017, 0.5877853), tmp);

        tmp = recv[0];
        tmp = _complex_fma(recv[1], de::CPf(0.309017, -0.9510565), tmp);
        tmp = _complex_fma(recv[2], de::CPf(-0.809017, -0.5877853), tmp);
        tmp = _complex_fma(recv[3], de::CPf(-0.809017, 0.5877853), tmp);
        res[4] = _complex_fma(recv[4], de::CPf(0.309017, 0.9510565), tmp);

        dex = i * 5;
        dst[dex] = *((double*)&res[0]);
        dst[dex + 1] = *((double*)&res[1]);
        dst[dex + 2] = *((double*)&res[2]);
        dst[dex + 3] = *((double*)&res[3]);
        dst[dex + 4] = *((double*)&res[4]);
    }
}



_THREAD_FUNCTION_
void decx::signal::CPUK::_IFFT1D_R5_fp32_C2C_first_ST(const double* __restrict src, 
    double* __restrict dst, 
    const size_t signal_length,
    const uint2 b_op_dex_range)
{
    using decx::signal::CPUK::_complex_fma;

    de::CPf recv[5], tmp;
    de::CPf res[5] = { de::CPf(0, 0) };

    size_t dex = 0;

    const size_t Bcalc_num = signal_length / 5;

    for (int i = b_op_dex_range.x; i < b_op_dex_range.y; ++i) {
        dex = i;
        *((double*)&recv[0]) = src[dex];
        *((double*)&recv[1]) = src[dex + Bcalc_num];
        *((double*)&recv[2]) = src[dex + Bcalc_num * 2];
        *((double*)&recv[3]) = src[dex + Bcalc_num * 3];
        *((double*)&recv[4]) = src[dex + Bcalc_num * 4];

#pragma unroll 5
        for (int k = 0; k < 5; ++k) {
            recv[k].real /= (float)signal_length;
            recv[k].image /= (-(float)signal_length);
        }

        res[0].real = recv[0].real + recv[1].real + recv[2].real + recv[3].real + recv[4].real;
        res[0].image = recv[0].image + recv[1].image + recv[2].image + recv[3].image + recv[4].image;

        tmp = recv[0];
        tmp = _complex_fma(recv[1], de::CPf(0.309017, 0.9510565), tmp);
        tmp = _complex_fma(recv[2], de::CPf(-0.809017, 0.5877853), tmp);
        tmp = _complex_fma(recv[3], de::CPf(-0.809017, -0.5877853), tmp);
        res[1] = _complex_fma(recv[4], de::CPf(0.309017, -0.9510565), tmp);

        tmp = recv[0];
        tmp = _complex_fma(recv[1], de::CPf(-0.809017, 0.5877853), tmp);
        tmp = _complex_fma(recv[2], de::CPf(0.309017, -0.9510565), tmp);
        tmp = _complex_fma(recv[3], de::CPf(0.309017, 0.9510565), tmp);
        res[2] = _complex_fma(recv[4], de::CPf(-0.809017, -0.5877853), tmp);

        tmp = recv[0];
        tmp = _complex_fma(recv[1], de::CPf(-0.809017, -0.5877853), tmp);
        tmp = _complex_fma(recv[2], de::CPf(0.309017, 0.9510565), tmp);
        tmp = _complex_fma(recv[3], de::CPf(0.309017, -0.9510565), tmp);
        res[3] = _complex_fma(recv[4], de::CPf(-0.809017, 0.5877853), tmp);

        tmp = recv[0];
        tmp = _complex_fma(recv[1], de::CPf(0.309017, -0.9510565), tmp);
        tmp = _complex_fma(recv[2], de::CPf(-0.809017, -0.5877853), tmp);
        tmp = _complex_fma(recv[3], de::CPf(-0.809017, 0.5877853), tmp);
        res[4] = _complex_fma(recv[4], de::CPf(0.309017, 0.9510565), tmp);

        dex = i * 5;
        dst[dex] = *((double*)&res[0]);
        dst[dex + 1] = *((double*)&res[1]);
        dst[dex + 2] = *((double*)&res[2]);
        dst[dex + 3] = *((double*)&res[3]);
        dst[dex + 4] = *((double*)&res[4]);
    }
}



_THREAD_FUNCTION_
void decx::signal::CPUK::_FFT1D_R5_fp32_C2C_ST(const double* __restrict src, 
                                         double* __restrict dst, 
                                         const size_t signal_length, 
                                         const size_t warp_proc_len,
                                         const uint2 b_op_dex_range)
{
    de::CPf recv[5], res[5], W, tmp;

    size_t dex = 0;
    uint num_of_Bcalc_in_warp = warp_proc_len / 5, warp_loc_id;
    
    const size_t total_Bcalc_num = signal_length / 5;

    for (int i = b_op_dex_range.x; i < b_op_dex_range.y; ++i) {
        dex = i;
        *((double*)&recv[0]) = src[dex];
        *((double*)&recv[1]) = src[dex + total_Bcalc_num];
        *((double*)&recv[2]) = src[dex + total_Bcalc_num * 2];
        *((double*)&recv[3]) = src[dex + total_Bcalc_num * 3];
        *((double*)&recv[4]) = src[dex + total_Bcalc_num * 4];

        warp_loc_id = i % num_of_Bcalc_in_warp;

        W.construct_with_phase(2 * Pi * (float)warp_loc_id / (float)warp_proc_len);
        recv[1] = decx::signal::CPUK::_complex_mul(recv[1], W);
        W.construct_with_phase(2 * Pi * (float)(warp_loc_id * 2) / (float)warp_proc_len);
        recv[2] = decx::signal::CPUK::_complex_mul(recv[2], W);
        W.construct_with_phase(2 * Pi * (float)(warp_loc_id * 3) / (float)warp_proc_len);
        recv[3] = decx::signal::CPUK::_complex_mul(recv[3], W);
        W.construct_with_phase(2 * Pi * (float)(warp_loc_id * 4) / (float)warp_proc_len);
        recv[4] = decx::signal::CPUK::_complex_mul(recv[4], W);

        res[0].real = recv[0].real + recv[1].real + recv[2].real + recv[3].real + recv[4].real;
        res[0].image = recv[0].image + recv[1].image + recv[2].image + recv[3].image + recv[4].image;

        tmp = recv[0];
        tmp = _complex_fma(recv[1], de::CPf(0.309017, 0.9510565), tmp);
        tmp = _complex_fma(recv[2], de::CPf(-0.809017, 0.5877853), tmp);
        tmp = _complex_fma(recv[3], de::CPf(-0.809017, -0.5877853), tmp);
        res[1] = _complex_fma(recv[4], de::CPf(0.309017, -0.9510565), tmp);

        tmp = recv[0];
        tmp = _complex_fma(recv[1], de::CPf(-0.809017, 0.5877853), tmp);
        tmp = _complex_fma(recv[2], de::CPf(0.309017, -0.9510565), tmp);
        tmp = _complex_fma(recv[3], de::CPf(0.309017, 0.9510565), tmp);
        res[2] = _complex_fma(recv[4], de::CPf(-0.809017, -0.5877853), tmp);

        tmp = recv[0];
        tmp = _complex_fma(recv[1], de::CPf(-0.809017, -0.5877853), tmp);
        tmp = _complex_fma(recv[2], de::CPf(0.309017, 0.9510565), tmp);
        tmp = _complex_fma(recv[3], de::CPf(0.309017, -0.9510565), tmp);
        res[3] = _complex_fma(recv[4], de::CPf(-0.809017, 0.5877853), tmp);

        tmp = recv[0];
        tmp = _complex_fma(recv[1], de::CPf(0.309017, -0.9510565), tmp);
        tmp = _complex_fma(recv[2], de::CPf(-0.809017, -0.5877853), tmp);
        tmp = _complex_fma(recv[3], de::CPf(-0.809017, 0.5877853), tmp);
        res[4] = _complex_fma(recv[4], de::CPf(0.309017, 0.9510565), tmp);

        dex = (i / num_of_Bcalc_in_warp) * warp_proc_len + warp_loc_id;
        dst[dex] = *((double*)&res[0]);
        dst[dex + num_of_Bcalc_in_warp] = *((double*)&res[1]);
        dst[dex + num_of_Bcalc_in_warp * 2] = *((double*)&res[2]);
        dst[dex + num_of_Bcalc_in_warp * 3] = *((double*)&res[3]);
        dst[dex + num_of_Bcalc_in_warp * 4] = *((double*)&res[4]);
    }
}



_THREAD_FUNCTION_
void decx::signal::CPUK::_FFT1D_R5_fp32_C2C_ST_vec4(const double* __restrict src, 
                                              double* __restrict dst, 
                                              const size_t signal_length, 
                                              const size_t warp_proc_len,
                                              const uint2 b_op_dex_range)
{
    __m256 recv[5];
    __m256 res;

    size_t dex = 0;     // in vec4
    uint num_of_Bcalc_in_warp = warp_proc_len / 5 / 4,  // in vec4
        warp_loc_id;        // in vec4

    const size_t total_Bcalc_num = signal_length / 5;

    for (int i = b_op_dex_range.x; i < b_op_dex_range.y; ++i) {
        recv[0] = _mm256_castpd_ps(_mm256_load_pd(src + (i << 2)));
        recv[1] = _mm256_castpd_ps(_mm256_load_pd(src + (i << 2) + total_Bcalc_num));
        recv[2] = _mm256_castpd_ps(_mm256_load_pd(src + (i << 2) + total_Bcalc_num * 2));
        recv[3] = _mm256_castpd_ps(_mm256_load_pd(src + (i << 2) + total_Bcalc_num * 3));
        recv[4] = _mm256_castpd_ps(_mm256_load_pd(src + (i << 2) + total_Bcalc_num * 4));

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

        recv[4] = decx::signal::CPUK::_cp4_mul_cp4_fp32(recv[4],
            _mm256_setr_ps(cosf(Two_Pi * (warp_loc_id * 4) * 4 / warp_proc_len), sinf(Two_Pi * ((warp_loc_id << 2)) * 4 / warp_proc_len),
                cosf(Two_Pi * ((warp_loc_id << 2) + 1) * 4 / warp_proc_len), sinf(Two_Pi * ((warp_loc_id << 2) + 1) * 4 / warp_proc_len),
                cosf(Two_Pi * ((warp_loc_id << 2) + 2) * 4 / warp_proc_len), sinf(Two_Pi * ((warp_loc_id << 2) + 2) * 4 / warp_proc_len),
                cosf(Two_Pi * ((warp_loc_id << 2) + 3) * 4 / warp_proc_len), sinf(Two_Pi * ((warp_loc_id << 2) + 3) * 4 / warp_proc_len)));

        res = _mm256_add_ps(_mm256_add_ps(recv[0], recv[1]), _mm256_add_ps(recv[2], recv[3]));
        res = _mm256_add_ps(res, recv[4]);
        _mm256_store_pd(dst + dex * 4, _mm256_castps_pd(res));
        //_expand_CPf32_MM256_
        dex += num_of_Bcalc_in_warp;
        res = _cp4_fma_cp4_fp32(recv[1], _expand_CPf32_MM256_(0.309017, 0.9510565), recv[0]);
        res = _cp4_fma_cp4_fp32(recv[2], _expand_CPf32_MM256_(-0.809017, 0.5877853), res);
        res = _cp4_fma_cp4_fp32(recv[3], _expand_CPf32_MM256_(-0.809017, -0.5877853), res);
        res = _cp4_fma_cp4_fp32(recv[4], _expand_CPf32_MM256_(0.309017, -0.9510565), res);
        _mm256_store_pd(dst + dex * 4, _mm256_castps_pd(res));

        dex += num_of_Bcalc_in_warp;
        res = _cp4_fma_cp4_fp32(recv[1], _expand_CPf32_MM256_(-0.809017, 0.5877853), recv[0]);
        res = _cp4_fma_cp4_fp32(recv[2], _expand_CPf32_MM256_(0.309017, -0.9510565), res);
        res = _cp4_fma_cp4_fp32(recv[3], _expand_CPf32_MM256_(0.309017, 0.9510565), res);
        res = _cp4_fma_cp4_fp32(recv[4], _expand_CPf32_MM256_(-0.809017, -0.5877853), res);
        _mm256_store_pd(dst + dex * 4, _mm256_castps_pd(res));

        dex += num_of_Bcalc_in_warp;
        res = _cp4_fma_cp4_fp32(recv[1], _expand_CPf32_MM256_(-0.809017, -0.5877853), recv[0]);
        res = _cp4_fma_cp4_fp32(recv[2], _expand_CPf32_MM256_(0.309017, 0.9510565), res);
        res = _cp4_fma_cp4_fp32(recv[3], _expand_CPf32_MM256_(0.309017, -0.9510565), res);
        res = _cp4_fma_cp4_fp32(recv[4], _expand_CPf32_MM256_(-0.809017, 0.5877853), res);
        _mm256_store_pd(dst + dex * 4, _mm256_castps_pd(res));

        dex += num_of_Bcalc_in_warp;
        res = _cp4_fma_cp4_fp32(recv[1], _expand_CPf32_MM256_(0.309017, -0.9510565), recv[0]);
        res = _cp4_fma_cp4_fp32(recv[2], _expand_CPf32_MM256_(-0.809017, -0.5877853), res);
        res = _cp4_fma_cp4_fp32(recv[3], _expand_CPf32_MM256_(-0.809017, 0.5877853), res);
        res = _cp4_fma_cp4_fp32(recv[4], _expand_CPf32_MM256_(0.309017, 0.9510565), res);
        _mm256_store_pd(dst + dex * 4, _mm256_castps_pd(res));
    }
}




// If the first kernel of FFT is radix-3, that means 4 or 2 are not going to show up in bases.
// Hence, it is no use to develop a vec4 function for R3_first
void decx::signal::cpu::FFT_R5_R2C_assign_task_1D_first(decx::utils::_thr_1D* t1D, decx::utils::frag_manager* f_mgr, 
    const float* __restrict src, double* __restrict dst, const size_t signal_length)
{
    const size_t _total_Bop_num = signal_length / 5;

    decx::utils::frag_manager_gen(f_mgr, _total_Bop_num, t1D->total_thread);

    uint2 start_end = make_uint2(0, 0);
    if (f_mgr->is_left) {
        for (int i = 0; i < t1D->total_thread - 1; ++i) {
            start_end.y += f_mgr->frag_len;
            t1D->_async_thread[i] = decx::cpu::register_task(&decx::thread_pool, 
                decx::signal::CPUK::_FFT1D_R5_fp32_R2C_first_ST, 
                src, dst, signal_length, start_end);
            start_end.x += f_mgr->frag_len;
        }
        start_end.y += f_mgr->frag_left_over;
        t1D->_async_thread[t1D->total_thread - 1] = decx::cpu::register_task(&decx::thread_pool, 
            decx::signal::CPUK::_FFT1D_R5_fp32_R2C_first_ST, 
            src, dst, signal_length, start_end);
    }
    else {
        for (int i = 0; i < t1D->total_thread; ++i) {
            start_end.y += f_mgr->frag_len;
            t1D->_async_thread[i] = decx::cpu::register_task(&decx::thread_pool, 
                decx::signal::CPUK::_FFT1D_R5_fp32_R2C_first_ST, 
                src, dst, signal_length, start_end);
            start_end.x += f_mgr->frag_len;
        }
    }
    t1D->__sync_all_threads();
}



// If the first kernel of FFT is radix-3, that means 4 or 2 are not going to show up in bases.
// Hence, it is no use to develop a vec4 function for R3_first
void decx::signal::cpu::FFT_R5_C2C_assign_task_1D_first(decx::utils::_thr_1D* t1D, decx::utils::frag_manager* f_mgr,
    const double* __restrict src, double* __restrict dst, const size_t signal_length)
{
    const size_t _total_Bop_num = signal_length / 5;

    decx::utils::frag_manager_gen(f_mgr, _total_Bop_num, t1D->total_thread);

    uint2 start_end = make_uint2(0, 0);
    if (f_mgr->is_left) {
        for (int i = 0; i < t1D->total_thread - 1; ++i) {
            start_end.y += f_mgr->frag_len;
            t1D->_async_thread[i] = decx::cpu::register_task(&decx::thread_pool,
                decx::signal::CPUK::_FFT1D_R5_fp32_C2C_first_ST,
                src, dst, signal_length, start_end);
            start_end.x += f_mgr->frag_len;
        }
        start_end.y += f_mgr->frag_left_over;
        t1D->_async_thread[t1D->total_thread - 1] = decx::cpu::register_task(&decx::thread_pool,
            decx::signal::CPUK::_FFT1D_R5_fp32_C2C_first_ST,
            src, dst, signal_length, start_end);
    }
    else {
        for (int i = 0; i < t1D->total_thread; ++i) {
            start_end.y += f_mgr->frag_len;
            t1D->_async_thread[i] = decx::cpu::register_task(&decx::thread_pool,
                decx::signal::CPUK::_FFT1D_R5_fp32_C2C_first_ST,
                src, dst, signal_length, start_end);
            start_end.x += f_mgr->frag_len;
        }
    }
    t1D->__sync_all_threads();
}


// If the first kernel of FFT is radix-3, that means 4 or 2 are not going to show up in bases.
// Hence, it is no use to develop a vec4 function for R3_first
void decx::signal::cpu::IFFT_R5_C2C_assign_task_1D_first(decx::utils::_thr_1D* t1D, decx::utils::frag_manager* f_mgr,
    const double* __restrict src, double* __restrict dst, const size_t signal_length)
{
    const size_t _total_Bop_num = signal_length / 5;

    decx::utils::frag_manager_gen(f_mgr, _total_Bop_num, t1D->total_thread);

    uint2 start_end = make_uint2(0, 0);
    if (f_mgr->is_left) {
        for (int i = 0; i < t1D->total_thread - 1; ++i) {
            start_end.y += f_mgr->frag_len;
            t1D->_async_thread[i] = decx::cpu::register_task(&decx::thread_pool,
                decx::signal::CPUK::_IFFT1D_R5_fp32_C2C_first_ST,
                src, dst, signal_length, start_end);
            start_end.x += f_mgr->frag_len;
        }
        start_end.y += f_mgr->frag_left_over;
        t1D->_async_thread[t1D->total_thread - 1] = decx::cpu::register_task(&decx::thread_pool,
            decx::signal::CPUK::_IFFT1D_R5_fp32_C2C_first_ST,
            src, dst, signal_length, start_end);
    }
    else {
        for (int i = 0; i < t1D->total_thread; ++i) {
            start_end.y += f_mgr->frag_len;
            t1D->_async_thread[i] = decx::cpu::register_task(&decx::thread_pool,
                decx::signal::CPUK::_IFFT1D_R5_fp32_C2C_first_ST,
                src, dst, signal_length, start_end);
            start_end.x += f_mgr->frag_len;
        }
    }
    t1D->__sync_all_threads();
}


void decx::signal::cpu::FFT_R5_C2C_assign_task_1D(decx::utils::_thr_1D* t1D, decx::utils::frag_manager* f_mgr,
    decx::alloc::MIF<double>* MIF_0, decx::alloc::MIF<double>* MIF_1, const size_t signal_length,
    const size_t warp_proc_len)
{
    decx::fft::__called_ST_func kernel_ptr = NULL;

    const size_t _total_Bop_num = signal_length / 5;

    if ((warp_proc_len / 5) % 4) {
        kernel_ptr = decx::signal::CPUK::_FFT1D_R5_fp32_C2C_ST;
        decx::utils::frag_manager_gen(f_mgr, _total_Bop_num, t1D->total_thread);
    }
    else {
        kernel_ptr = decx::signal::CPUK::_FFT1D_R5_fp32_C2C_ST_vec4;
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
