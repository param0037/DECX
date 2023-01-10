/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/


#include "FFT1D_radix_3_kernel.h"


_THREAD_FUNCTION_
void decx::signal::CPUK::_FFT1D_R3_fp32_R2C_first_ST(const float* __restrict src, 
    double* __restrict dst, 
    const size_t signal_length,
    const uint2 b_op_dex_range)
{
    float recv[3];
    de::CPf res[3];

    res[1].image = 0;
    res[2].image = 0;
    size_t dex = 0;

    const size_t Bcalc_num = signal_length / 3;

    for (int i = b_op_dex_range.x; i < b_op_dex_range.y; ++i) {
        dex = i;
        recv[0] = src[dex];
        recv[1] = src[dex + Bcalc_num];
        recv[2] = src[dex + Bcalc_num * 2];

        res[0].real = recv[0] + recv[1] + recv[2];
        res[0].image = 0;

        res[1].real = recv[0] + (recv[1] + recv[2]) * (-0.5);
        res[1].image = (recv[1] - recv[2]) * 0.8660254f;

        res[2].real = recv[0] + (recv[1] + recv[2]) * (-0.5);
        res[2].image = (recv[2] - recv[1]) * 0.8660254f;

        dex = i * 3;
        dst[dex] = *((double*)&res[0]);
        dst[dex + 1] = *((double*)&res[1]);
        dst[dex + 2] = *((double*)&res[2]);
    }
}



_THREAD_FUNCTION_
void decx::signal::CPUK::_IFFT1D_R3_fp32_C2C_first_ST(const double* __restrict src, 
                                                double* __restrict dst, 
                                                const size_t signal_length,
                                                const uint2 b_op_dex_range)
{
    de::CPf recv[3], res[3];
    size_t dex = 0;

    const size_t Bcalc_num = signal_length / 3;

    for (int i = b_op_dex_range.x; i < b_op_dex_range.y; ++i) {
        dex = i;
        *((double*)&recv[0]) = src[dex];
        *((double*)&recv[1]) = src[dex + Bcalc_num];
        *((double*)&recv[2]) = src[dex + Bcalc_num * 2];

        recv[0].real /= (float)signal_length;       recv[0].image /= -(float)signal_length;
        recv[1].real /= (float)signal_length;       recv[1].image /= -(float)signal_length;
        recv[2].real /= (float)signal_length;       recv[2].image /= -(float)signal_length;

        res[0].real = recv[0].real +recv[1].real +recv[2].real;
        res[0].image = recv[0].image + recv[1].image + recv[2].image;

        res[1] = decx::signal::CPUK::_complex_fma(recv[1], de::CPf(-0.5, 0.8660254f), recv[0]);
        res[1] = decx::signal::CPUK::_complex_fma(recv[2], de::CPf(-0.5, -0.8660254f), res[1]);

        res[2] = decx::signal::CPUK::_complex_fma(recv[1], de::CPf(-0.5, -0.8660254f), recv[0]);
        res[2] = decx::signal::CPUK::_complex_fma(recv[2], de::CPf(-0.5, 0.8660254f), res[2]);

        dex = i * 3;
        dst[dex] = *((double*)&res[0]);
        dst[dex + 1] = *((double*)&res[1]);
        dst[dex + 2] = *((double*)&res[2]);
    }
}



_THREAD_FUNCTION_
void decx::signal::CPUK::_FFT1D_R3_fp32_C2C_first_ST(const double* __restrict src,
    double* __restrict dst,
    const size_t signal_length,
    const uint2 b_op_dex_range)
{
    de::CPf recv[3], res[3];
    size_t dex = 0;

    const size_t Bcalc_num = signal_length / 3;

    for (int i = b_op_dex_range.x; i < b_op_dex_range.y; ++i) {
        dex = i;
        *((double*)&recv[0]) = src[dex];
        *((double*)&recv[1]) = src[dex + Bcalc_num];
        *((double*)&recv[2]) = src[dex + Bcalc_num * 2];

        res[0].real = recv[0].real + recv[1].real + recv[2].real;
        res[0].image = recv[0].image + recv[1].image + recv[2].image;

        res[1] = decx::signal::CPUK::_complex_fma(recv[1], de::CPf(-0.5, 0.8660254f), recv[0]);
        res[1] = decx::signal::CPUK::_complex_fma(recv[2], de::CPf(-0.5, -0.8660254f), res[1]);

        res[2] = decx::signal::CPUK::_complex_fma(recv[1], de::CPf(-0.5, -0.8660254f), recv[0]);
        res[2] = decx::signal::CPUK::_complex_fma(recv[2], de::CPf(-0.5, 0.8660254f), res[2]);

        dex = i * 3;
        dst[dex] = *((double*)&res[0]);
        dst[dex + 1] = *((double*)&res[1]);
        dst[dex + 2] = *((double*)&res[2]);
    }
}




_THREAD_FUNCTION_
void decx::signal::CPUK::_FFT1D_R3_fp32_C2C_ST(const double* __restrict src, 
                                         double* __restrict dst, 
                                         const size_t signal_length, 
                                         const size_t warp_proc_len,
                                         const uint2 b_op_dex_range)
{
    de::CPf recv[3], res[3], W, tmp;

    size_t dex = 0;
    uint num_of_Bcalc_in_warp = warp_proc_len / 3, warp_loc_id;
    
    const size_t total_Bcalc_num = signal_length / 3;

    for (int i = b_op_dex_range.x; i < b_op_dex_range.y; ++i) {
        dex = i;
        *((double*)&recv[0]) = src[dex];
        *((double*)&recv[1]) = src[dex + total_Bcalc_num];
        *((double*)&recv[2]) = src[dex + total_Bcalc_num * 2];

        warp_loc_id = i % num_of_Bcalc_in_warp;

        W.construct_with_phase(2 * Pi * (float)warp_loc_id / (float)warp_proc_len);
        recv[1] = decx::signal::CPUK::_complex_mul(recv[1], W);
        W.construct_with_phase(2 * Pi * (float)(warp_loc_id * 2) / (float)warp_proc_len);
        recv[2] = decx::signal::CPUK::_complex_mul(recv[2], W);

        res[0].real = recv[0].real + recv[1].real + recv[2].real;
        res[0].image = recv[0].image + recv[1].image + recv[2].image;

        tmp = recv[0];
        tmp = _complex_fma(recv[1], de::CPf(-0.5, 0.8660254f), tmp);
        res[1] = _complex_fma(recv[2], de::CPf(-0.5, -0.8660254f), tmp);

        tmp = recv[0];
        tmp = _complex_fma(recv[1], de::CPf(-0.5, -0.8660254f), tmp);
        res[2] = _complex_fma(recv[2], de::CPf(-0.5, 0.8660254f), tmp);

        dex = (i / num_of_Bcalc_in_warp) * warp_proc_len + warp_loc_id;
        dst[dex] = *((double*)&res[0]);
        dst[dex + num_of_Bcalc_in_warp] = *((double*)&res[1]);
        dst[dex + num_of_Bcalc_in_warp * 2] = *((double*)&res[2]);
    }
}



_THREAD_FUNCTION_
void decx::signal::CPUK::_FFT1D_R3_fp32_C2C_ST_vec4(const double* __restrict src, 
                                              double* __restrict dst, 
                                              const size_t signal_length, 
                                              const size_t warp_proc_len,
                                              const uint2 b_op_dex_range)
{
    __m256 recv[3];
    __m256 res;

    size_t dex = 0;     // in vec4
    uint num_of_Bcalc_in_warp = warp_proc_len / 3 / 4,  // in vec4
        warp_loc_id;        // in vec4

    const size_t total_Bcalc_num = signal_length / 3;

    for (int i = b_op_dex_range.x; i < b_op_dex_range.y; ++i) {
        recv[0] = _mm256_castpd_ps(_mm256_load_pd(src + (i << 2)));
        recv[1] = _mm256_castpd_ps(_mm256_load_pd(src + (i << 2) + total_Bcalc_num));
        recv[2] = _mm256_castpd_ps(_mm256_load_pd(src + (i << 2) + total_Bcalc_num * 2));

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

        res = _mm256_add_ps(recv[0], recv[1]);
        res = _mm256_add_ps(res, recv[2]);
        _mm256_store_pd(dst + dex * 4, _mm256_castps_pd(res));
        //_expand_CPf32_MM256_
        dex += num_of_Bcalc_in_warp;
        res = _cp4_fma_cp4_fp32(recv[1], _expand_CPf32_MM256_(-0.5, 0.8660254f), recv[0]);
        res = _cp4_fma_cp4_fp32(recv[2], _expand_CPf32_MM256_(-0.5, -0.8660254f), res);
        _mm256_store_pd(dst + dex * 4, _mm256_castps_pd(res));

        dex += num_of_Bcalc_in_warp;
        res = _cp4_fma_cp4_fp32(recv[1], _expand_CPf32_MM256_(-0.5, -0.8660254f), recv[0]);
        res = _cp4_fma_cp4_fp32(recv[2], _expand_CPf32_MM256_(-0.5, 0.8660254f), res);
        _mm256_store_pd(dst + dex * 4, _mm256_castps_pd(res));
    }
}



// If the first kernel of FFT is radix-3, that means 4 or 2 are not going to show up in bases.
// Hence, it is no use to develop a vec4 function for R3_first
void decx::signal::FFT_R3_R2C_assign_task_1D_first(decx::utils::_thr_1D* t1D, decx::utils::frag_manager* f_mgr, 
    const float* __restrict src, double* __restrict dst, const size_t signal_length)
{
    const size_t _total_Bop_num = signal_length / 3;
    
    decx::utils::frag_manager_gen(f_mgr, _total_Bop_num, t1D->total_thread);

    uint2 start_end = make_uint2(0, 0);
    if (f_mgr->is_left) {
        for (int i = 0; i < t1D->total_thread - 1; ++i) {
            start_end.y += f_mgr->frag_len;
            t1D->_async_thread[i] = decx::cpu::register_task(&decx::thread_pool, 
                decx::signal::CPUK::_FFT1D_R3_fp32_R2C_first_ST, 
                src, dst, signal_length, start_end);
            start_end.x += f_mgr->frag_len;
        }
        start_end.y += f_mgr->frag_left_over;
        t1D->_async_thread[t1D->total_thread - 1] = decx::cpu::register_task(&decx::thread_pool, 
            decx::signal::CPUK::_FFT1D_R3_fp32_R2C_first_ST, 
            src, dst, signal_length, start_end);
    }
    else {
        for (int i = 0; i < t1D->total_thread; ++i) {
            start_end.y += f_mgr->frag_len;
            t1D->_async_thread[i] = decx::cpu::register_task(&decx::thread_pool, 
                decx::signal::CPUK::_FFT1D_R3_fp32_R2C_first_ST, 
                src, dst, signal_length, start_end);
            start_end.x += f_mgr->frag_len;
        }
    }
    t1D->__sync_all_threads();
}



void decx::signal::IFFT_R3_C2C_assign_task_1D_first(decx::utils::_thr_1D* t1D, decx::utils::frag_manager* f_mgr, 
    const double* __restrict src, double* __restrict dst, const size_t signal_length)
{
    const size_t _total_Bop_num = signal_length / 3;
    decx::utils::frag_manager_gen(f_mgr, _total_Bop_num, t1D->total_thread);

    uint2 start_end = make_uint2(0, 0);
    if (f_mgr->is_left) {
        for (int i = 0; i < t1D->total_thread - 1; ++i) {
            start_end.y += f_mgr->frag_len;
            t1D->_async_thread[i] = decx::cpu::register_task(&decx::thread_pool, 
                decx::signal::CPUK::_IFFT1D_R3_fp32_C2C_first_ST, 
                src, dst, signal_length, start_end);
            start_end.x += f_mgr->frag_len;
        }
        start_end.y += f_mgr->frag_left_over;
        t1D->_async_thread[t1D->total_thread - 1] = decx::cpu::register_task(&decx::thread_pool, 
            decx::signal::CPUK::_IFFT1D_R3_fp32_C2C_first_ST, 
            src, dst, signal_length, start_end);
    }
    else {
        for (int i = 0; i < t1D->total_thread; ++i) {
            start_end.y += f_mgr->frag_len;
            t1D->_async_thread[i] = decx::cpu::register_task(&decx::thread_pool, 
                decx::signal::CPUK::_IFFT1D_R3_fp32_C2C_first_ST, 
                src, dst, signal_length, start_end);
            start_end.x += f_mgr->frag_len;
        }
    }
    t1D->__sync_all_threads();
}



void decx::signal::FFT_R3_C2C_assign_task_1D_first(decx::utils::_thr_1D* t1D, decx::utils::frag_manager* f_mgr,
    const double* __restrict src, double* __restrict dst, const size_t signal_length)
{
    const size_t _total_Bop_num = signal_length / 3;
    decx::utils::frag_manager_gen(f_mgr, _total_Bop_num, t1D->total_thread);

    uint2 start_end = make_uint2(0, 0);
    if (f_mgr->is_left) {
        for (int i = 0; i < t1D->total_thread - 1; ++i) {
            start_end.y += f_mgr->frag_len;
            t1D->_async_thread[i] = decx::cpu::register_task(&decx::thread_pool,
                decx::signal::CPUK::_FFT1D_R3_fp32_C2C_first_ST,
                src, dst, signal_length, start_end);
            start_end.x += f_mgr->frag_len;
        }
        start_end.y += f_mgr->frag_left_over;
        t1D->_async_thread[t1D->total_thread - 1] = decx::cpu::register_task(&decx::thread_pool,
            decx::signal::CPUK::_FFT1D_R3_fp32_C2C_first_ST,
            src, dst, signal_length, start_end);
    }
    else {
        for (int i = 0; i < t1D->total_thread; ++i) {
            start_end.y += f_mgr->frag_len;
            t1D->_async_thread[i] = decx::cpu::register_task(&decx::thread_pool,
                decx::signal::CPUK::_FFT1D_R3_fp32_C2C_first_ST,
                src, dst, signal_length, start_end);
            start_end.x += f_mgr->frag_len;
        }
    }
    t1D->__sync_all_threads();
}




void decx::signal::FFT_R3_C2C_assign_task_1D(decx::utils::_thr_1D* t1D, decx::utils::frag_manager* f_mgr,
    decx::alloc::MIF<double>* MIF_0, decx::alloc::MIF<double>* MIF_1, const size_t signal_length,
    const size_t warp_proc_len)
{
    decx::fft::__called_ST_func kernel_ptr = NULL;

    const size_t _total_Bop_num = signal_length / 3;

    if ((warp_proc_len / 3) % 4) {
        kernel_ptr = decx::signal::CPUK::_FFT1D_R3_fp32_C2C_ST;
        decx::utils::frag_manager_gen(f_mgr, _total_Bop_num, t1D->total_thread);
    }
    else {
        kernel_ptr = decx::signal::CPUK::_FFT1D_R3_fp32_C2C_ST_vec4;
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
