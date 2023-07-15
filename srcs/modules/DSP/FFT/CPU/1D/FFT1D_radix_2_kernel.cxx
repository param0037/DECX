/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#include "FFT1D_radix_2_kernel.h"


_THREAD_FUNCTION_
void decx::signal::CPUK::_FFT1D_R2_fp32_R2C_first_ST(const float* __restrict src, 
                                               double* __restrict dst, 
                                               const size_t signal_length, 
                                               const uint2 b_op_dex_range)
{
    float recv[2];
    de::CPf res[2] = { de::CPf(0, 0), de::CPf(0, 0) };
    size_t dex = 0;
    const size_t b_op_num = signal_length / 2;

    for (uint i = b_op_dex_range.x; i < b_op_dex_range.y; ++i) {
        dex = i;
        recv[0] = src[dex];
        recv[1] = src[dex + b_op_num];

        res[0].real = recv[0] + recv[1];
        res[1].real = recv[0] - recv[1];

        dex = i * 2;
        dst[dex] = *((double*)&res[0]);
        dst[dex + 1] = *((double*)&res[1]);
    }
}



_THREAD_FUNCTION_
void decx::signal::CPUK::_FFT1D_R2_fp32_C2C_first_ST(const double* __restrict src, 
                                               double* __restrict dst, 
                                               const size_t signal_length, 
                                               const uint2 b_op_dex_range)
{
    de::CPf recv[2];
    de::CPf res[2] = { de::CPf(0, 0), de::CPf(0, 0) };
    size_t dex = 0;
    const size_t b_op_num = signal_length / 2;

    for (uint i = b_op_dex_range.x; i < b_op_dex_range.y; ++i) {
        dex = i;
        *((double*)&recv[0]) = src[dex];
        *((double*)&recv[1]) = src[dex + b_op_num];

        res[0].real = recv[0].real + recv[1].real;
        res[0].image = recv[0].image + recv[1].image;
        res[1].real = recv[0].real - recv[1].real;
        res[1].image = recv[0].image - recv[1].image;

        dex = i * 2;
        dst[dex] = *((double*)&res[0]);
        dst[dex + 1] = *((double*)&res[1]);
    }
}



_THREAD_FUNCTION_
void decx::signal::CPUK::_IFFT1D_R2_fp32_C2C_first_ST(const double* __restrict src, 
                                                double* __restrict dst, 
                                                const size_t signal_length, 
                                                const uint2 b_op_dex_range)
{
    de::CPf recv[2];
    de::CPf res[2] = { de::CPf(0, 0), de::CPf(0, 0) };
    size_t dex = 0;
    const size_t b_op_num = signal_length / 2;

    for (uint i = b_op_dex_range.x; i < b_op_dex_range.y; ++i) {
        dex = i;
        *((double*)&recv[0]) = src[dex];
        *((double*)&recv[1]) = src[dex + b_op_num];

        *((__m128*)&recv[0]) = _mm_div_ps(*((__m128*)&recv[0]), _mm_set_ps1((float)signal_length));
        *((__m128*)&recv[0]) = _mm_mul_ps(*((__m128*)&recv[0]), _mm_setr_ps(1, -1, 1, -1));

        res[0].real = recv[0].real + recv[1].real;
        res[0].image = recv[0].image + recv[1].image;
        res[1].real = recv[0].real - recv[1].real;
        res[1].image = recv[0].image - recv[1].image;

        dex = i * 2;
        dst[dex] = *((double*)&res[0]);
        dst[dex + 1] = *((double*)&res[1]);
    }
}



_THREAD_FUNCTION_
void decx::signal::CPUK::_FFT1D_R2_fp32_R2C_first_ST_vec4(const float* __restrict src,
                                                          double* __restrict dst,
                                                          const size_t signal_length,
                                                          const uint2 b_op_dex_range)
{
    __m128 recv[2], tmp;
    __m256 res;
    size_t dex = 0;
    const size_t half_signal_len = signal_length / 2;

    for (uint i = b_op_dex_range.x; i < b_op_dex_range.y; i += 4) {
        dex = i * 2;
        recv[0] = _mm_load_ps(src + i);
        recv[1] = _mm_load_ps(src + i + half_signal_len);

        tmp = _mm_add_ps(recv[0], recv[1]);
        recv[0] = _mm_sub_ps(recv[0], recv[1]);
        
        recv[1] = _mm_shuffle_ps(tmp, recv[0], 0b01000100);
        // res = {+, 0, -, 0, +, 0, -, 0}
        res = _mm256_permutevar8x32_ps(_mm256_castps128_ps256(recv[1]), _mm256_set_epi32(7, 3, 5, 1, 6, 2, 4, 0));
        _mm256_store_pd(dst + dex, _mm256_castps_pd(res));

        recv[1] = _mm_shuffle_ps(tmp, recv[0], 0b11101110);
        res = _mm256_permutevar8x32_ps(_mm256_castps128_ps256(recv[1]), _mm256_set_epi32(7, 3, 5, 1, 6, 2, 4, 0));
        _mm256_store_pd(dst + dex + 4, _mm256_castps_pd(res));
    }
}




_THREAD_FUNCTION_
void decx::signal::CPUK::_FFT1D_R2_fp32_C2C_ST(const double* __restrict src, 
                                         double* __restrict dst, 
                                         const size_t signal_length, 
                                         const size_t warp_proc_len,
                                         const uint2 b_op_dex_range)
{
    de::CPf recv[2], res[2], W;

    size_t dex = 0;
    uint num_of_Bcalc_in_warp = warp_proc_len / 2, warp_loc_id;
    const size_t b_op_num = signal_length / 2;

    for (uint i = b_op_dex_range.x; i < b_op_dex_range.y; ++i) {
        dex = i;
        *((double*)&recv[0]) = src[dex];
        *((double*)&recv[1]) = src[dex + b_op_num];

        warp_loc_id = i % num_of_Bcalc_in_warp;

        W.construct_with_phase(Two_Pi * (float)warp_loc_id / (float)warp_proc_len);

        res[0].real = recv[1].real * W.real - recv[1].image * W.image + recv[0].real;
        res[0].image = recv[1].real * W.image + recv[1].image * W.real + recv[0].image;

        res[1].real = recv[0].real - recv[1].real * W.real - recv[1].image * W.image;
        res[1].image = recv[0].image - recv[1].real * W.image + recv[1].image * W.real;

        dex = (i / num_of_Bcalc_in_warp) * warp_proc_len + warp_loc_id;
        dst[dex] = *((double*)&res[0]);
        dst[dex + num_of_Bcalc_in_warp] = *((double*)&res[1]);
    }
}




_THREAD_FUNCTION_
void decx::signal::CPUK::_FFT1D_R2_fp32_C2C_ST_vec4(const double* __restrict src,
                                              double* __restrict dst,
                                              const size_t signal_length,     // in element
                                              const size_t warp_proc_len,     // in element
                                              const uint2 b_op_dex_range)     // in vec4
{
    __m256 recv[2], tmp, res;
    size_t dex = 0;     // in vec4

    uint num_of_Bcalc_in_warp = warp_proc_len / 2 / 4, 
        warp_loc_id;
    const size_t half_signal_len = signal_length / 2;

    for (uint i = b_op_dex_range.x; i < b_op_dex_range.y; ++i) {
        recv[0] = _mm256_castpd_ps(_mm256_load_pd(src + i * 4));
        recv[1] = _mm256_castpd_ps(_mm256_load_pd(src + i * 4 + half_signal_len));

        warp_loc_id = i % num_of_Bcalc_in_warp;
        dex = (i / num_of_Bcalc_in_warp) * warp_proc_len / 4 + warp_loc_id;

        recv[1] = decx::signal::CPUK::_cp4_mul_cp4_fp32(recv[1], _mm256_setr_ps(cosf(Two_Pi * (warp_loc_id << 2) / warp_proc_len), sinf(Two_Pi * ((warp_loc_id << 2)) / warp_proc_len),
            cosf(Two_Pi * ((warp_loc_id << 2) + 1) / warp_proc_len), sinf(Two_Pi * ((warp_loc_id << 2) + 1) / warp_proc_len),
            cosf(Two_Pi * ((warp_loc_id << 2) + 2) / warp_proc_len), sinf(Two_Pi * ((warp_loc_id << 2) + 2) / warp_proc_len),
            cosf(Two_Pi * ((warp_loc_id << 2) + 3) / warp_proc_len), sinf(Two_Pi * ((warp_loc_id << 2) + 3) / warp_proc_len)));

        res = _mm256_add_ps(recv[0], recv[1]);
        _mm256_store_pd(dst + dex * 4, _mm256_castps_pd(res));

        dex += num_of_Bcalc_in_warp;
        res = _mm256_sub_ps(recv[0], recv[1]);
        _mm256_store_pd(dst + dex * 4, _mm256_castps_pd(res));
    }
}



void decx::signal::cpu::FFT_R2_R2C_assign_task_1D_first(decx::utils::_thr_1D* t1D, decx::utils::frag_manager* f_mgr,
    const float* __restrict src, double* __restrict dst, const size_t signal_length)
{
    const size_t _total_Bop_num = signal_length / 2;
    decx::utils::frag_manager_gen(f_mgr, _total_Bop_num, t1D->total_thread);
    
    uint2 start_end = make_uint2(0, 0);
    if (f_mgr->is_left) {
        for (int i = 0; i < t1D->total_thread - 1; ++i) {
            start_end.y += f_mgr->frag_len;
            t1D->_async_thread[i] = decx::cpu::register_task_default( 
                decx::signal::CPUK::_FFT1D_R2_fp32_R2C_first_ST, src, dst, signal_length, start_end);
            start_end.x += f_mgr->frag_len;
        }
        start_end.y += f_mgr->frag_left_over;
        t1D->_async_thread[t1D->total_thread - 1] = decx::cpu::register_task_default( 
            decx::signal::CPUK::_FFT1D_R2_fp32_R2C_first_ST, src, dst, signal_length, start_end);
    }
    else {
        for (int i = 0; i < t1D->total_thread; ++i) {
            start_end.y += f_mgr->frag_len;
            t1D->_async_thread[i] = decx::cpu::register_task_default( 
                decx::signal::CPUK::_FFT1D_R2_fp32_R2C_first_ST, src, dst, signal_length, start_end);
            start_end.x += f_mgr->frag_len;
        }
    }
    t1D->__sync_all_threads();
}



void decx::signal::cpu::FFT_R2_C2C_assign_task_1D_first(decx::utils::_thr_1D* t1D, decx::utils::frag_manager* f_mgr,
    const double* __restrict src, double* __restrict dst, const size_t signal_length)
{
    const size_t _total_Bop_num = signal_length / 2;
    decx::utils::frag_manager_gen(f_mgr, _total_Bop_num, t1D->total_thread);

    uint2 start_end = make_uint2(0, 0);
    if (f_mgr->is_left) {
        for (int i = 0; i < t1D->total_thread - 1; ++i) {
            start_end.y += f_mgr->frag_len;
            t1D->_async_thread[i] = decx::cpu::register_task_default( 
                decx::signal::CPUK::_FFT1D_R2_fp32_C2C_first_ST, src, dst, signal_length, start_end);
            start_end.x += f_mgr->frag_len;
        }
        start_end.y += f_mgr->frag_left_over;
        t1D->_async_thread[t1D->total_thread - 1] = decx::cpu::register_task_default( 
            decx::signal::CPUK::_FFT1D_R2_fp32_C2C_first_ST, src, dst, signal_length, start_end);
    }
    else {
        for (int i = 0; i < t1D->total_thread; ++i) {
            start_end.y += f_mgr->frag_len;
            t1D->_async_thread[i] = decx::cpu::register_task_default( 
                decx::signal::CPUK::_FFT1D_R2_fp32_C2C_first_ST, src, dst, signal_length, start_end);
            start_end.x += f_mgr->frag_len;
        }
    }
    t1D->__sync_all_threads();
}




void decx::signal::cpu::IFFT_R2_C2C_assign_task_1D_first(decx::utils::_thr_1D* t1D, decx::utils::frag_manager* f_mgr,
    const double* __restrict src, double* __restrict dst, const size_t signal_length)
{
    const size_t _total_Bop_num = signal_length / 2;
    //printf("_total_Bop_num : %d\n", f_mgr->frag_len);
    decx::utils::frag_manager_gen(f_mgr, _total_Bop_num, t1D->total_thread);
    uint2 start_end = make_uint2(0, 0);
    if (f_mgr->is_left) {
        for (int i = 0; i < t1D->total_thread - 1; ++i) {
            start_end.y += f_mgr->frag_len;
            t1D->_async_thread[i] = decx::cpu::register_task_default( 
                decx::signal::CPUK::_IFFT1D_R2_fp32_C2C_first_ST, src, dst, signal_length, start_end);
            start_end.x += f_mgr->frag_len;
        }
        start_end.y += f_mgr->frag_left_over;
        t1D->_async_thread[t1D->total_thread - 1] = decx::cpu::register_task_default( 
            decx::signal::CPUK::_IFFT1D_R2_fp32_C2C_first_ST, src, dst, signal_length, start_end);
    }
    else {
        for (int i = 0; i < t1D->total_thread; ++i) {
            start_end.y += f_mgr->frag_len;
            t1D->_async_thread[i] = decx::cpu::register_task_default( 
                decx::signal::CPUK::_IFFT1D_R2_fp32_C2C_first_ST, src, dst, signal_length, start_end);
            start_end.x += f_mgr->frag_len;
        }
    }
    t1D->__sync_all_threads();
}




void decx::signal::cpu::FFT_R2_C2C_assign_task_1D(decx::utils::_thr_1D* t1D, decx::utils::frag_manager* f_mgr,
    decx::alloc::MIF<double> *MIF_0, decx::alloc::MIF<double>* MIF_1, const size_t signal_length,
    const size_t warp_proc_len)
{
    decx::fft::__called_ST_func kernel_ptr = NULL;

    const size_t _total_Bop_num = signal_length / 2;

    if ((warp_proc_len / 2) % 4) {
        kernel_ptr = decx::signal::CPUK::_FFT1D_R2_fp32_C2C_ST;
        decx::utils::frag_manager_gen(f_mgr, _total_Bop_num, t1D->total_thread);
    }
    else {
        kernel_ptr = decx::signal::CPUK::_FFT1D_R2_fp32_C2C_ST_vec4;
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
            t1D->_async_thread[i] = decx::cpu::register_task_default( kernel_ptr, read_ptr, write_ptr, signal_length, warp_proc_len, start_end);
            start_end.x += f_mgr->frag_len;
        }
        start_end.y += f_mgr->frag_left_over;
        t1D->_async_thread[t1D->total_thread - 1] = decx::cpu::register_task_default( kernel_ptr, read_ptr, write_ptr, signal_length, warp_proc_len, start_end);
    }
    else {
        for (int i = 0; i < t1D->total_thread; ++i) {
            start_end.y += f_mgr->frag_len;
            t1D->_async_thread[i] = decx::cpu::register_task_default( kernel_ptr, read_ptr, write_ptr, signal_length, warp_proc_len, start_end);
            start_end.x += f_mgr->frag_len;
        }
    }
    t1D->__sync_all_threads();
}
