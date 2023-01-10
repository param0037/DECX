/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/


#ifndef _CONSTANT_FILL_EXEC_INT32_H_
#define _CONSTANT_FILL_EXEC_INT32_H_

#include "../../../../core/thread_management/thread_pool.h"
#include "../../../../core/thread_management/thread_arrange.h"
#include "../../../../core/utils/leftovers.h"
#include "../../../../core/utils/fragment_arrangment.h"


namespace decx
{
    _THREAD_FUNCTION_ void fill_c_1D_int32(float* src, const int _val, const size_t len);


    _THREAD_FUNCTION_ void fill_c_1D_int32_last(float* src, const int _val, const size_t len,
        decx::utils::_left_8* __L);


    /*
    * @param frag_info : .x -> pitch(in __m256); .y -> height
    * @param pitch : in float
    */
    _THREAD_FUNCTION_ void fill_c_2D_int32(float* src, const int _val, const uint2 frag_info,
        decx::utils::_left_8* _row_bound_info, const size_t pitch);


    /*
    * @param actual_r_len : in float
    * @param dims_info : the dimension of the input 2D matirx, width in float(pitch)
    */
    void fill_2D_int32(int* src, const int _value, const size_t actual_r_len, const uint2 dims_info);


    void fill_1D_int32(int* src, const int _value, const size_t actual_len);
}


_THREAD_FUNCTION_
void decx::fill_c_1D_int32(float* src, const int _val, const size_t len)
{
    const __m256 c_fill = _mm256_set1_ps(*((float*)&_val));
    for (int i = 0; i < len; ++i) {
        _mm256_store_ps(src + (i << 3), c_fill);
    }
}


_THREAD_FUNCTION_
void decx::fill_c_2D_int32(float* src, const int _val, const uint2 frag_info,
    decx::utils::_left_8* _row_bound_info, const size_t pitch)
{
    const __m256 c_fill = _mm256_set1_ps(*((float*)&_val));
    const __m256i c_fill_last = _row_bound_info->get_filled_fragment_int(_val);
    size_t dex = 0;

    for (int i = 0; i < frag_info.y; ++i) {
        dex = i * pitch;
        for (int j = 0; j < frag_info.x - 1; ++j) {
            _mm256_store_ps(src + dex, c_fill);
            dex += 8;
        }
        _mm256_store_ps(src + dex, *((__m256*) & c_fill_last));
    }
}


_THREAD_FUNCTION_
void decx::fill_c_1D_int32_last(float* src, const int _val, const size_t len,
    decx::utils::_left_8* __L)
{
    const __m256 c_fill = _mm256_set1_ps(_val);
    const __m256i c_fill_L = __L->get_filled_fragment_int(_val);
    uint i = 0;
    for (i = 0; i < len - 1; ++i) {
        _mm256_store_ps(src + (i << 3), c_fill);
    }
    // store the leftover one
    _mm256_store_ps(src + (i << 3), *((__m256*) & c_fill_L));
}


void decx::fill_1D_int32(int* src, const int _value, const const size_t actual_len)
{
    decx::utils::frag_manager f_mgr;
    uint conc_thr = decx::cpI.cpu_concurrency;

    size_t _8_divided = decx::utils::ceil<size_t>(actual_len, 8);
    std::future<void>* fut = new std::future<void>[conc_thr];

    bool __crit = decx::utils::frag_manager_gen(&f_mgr, _8_divided, conc_thr);
    if (!__crit)
    {       // if leftover
        float* frag_head_ptr = reinterpret_cast<float*>(src);

        for (int i = 0; i < f_mgr.frag_num - 1; ++i) {
            fut[i] = decx::thread_pool.register_task(
                decx::fill_c_1D_int32, frag_head_ptr, _value, f_mgr.frag_len);
            frag_head_ptr += (f_mgr.frag_len << 3);
        }
        size_t bound = actual_len;
        decx::utils::_left_8 _L(bound, actual_len % 8);

        fut[conc_thr - 1] = decx::thread_pool.register_task(
            decx::fill_c_1D_int32_last, frag_head_ptr, _value, f_mgr.frag_left_over, &_L);
    }
    else {
        size_t len_in_frag = actual_len / 8;
    }

    for (int i = 0; i < conc_thr; ++i) {
        fut[i].get();
    }

    delete[] fut;
}


void decx::fill_2D_int32(int* src, const int _value, const size_t actual_r_len,
    const uint2 dims_info)
{
    uint conc_thr = decx::cpI.cpu_concurrency;

    std::future<void>* fut = new std::future<void>[conc_thr];

    decx::utils::frag_manager f_mgr;
    bool __crit = decx::utils::frag_manager_gen(&f_mgr, dims_info.y, conc_thr);

    if (!__crit)
    {
        uint2 frag_info = make_uint2(dims_info.x / 8, f_mgr.frag_len);
        float* frag_head_ptr = reinterpret_cast<float*>(src);

        decx::utils::_left_8 __L(NULL, actual_r_len % 8);

        for (int i = 0; i < f_mgr.frag_num - 1; ++i) {
            fut[i] = decx::thread_pool.register_task(
                decx::fill_c_2D_int32, frag_head_ptr, _value, frag_info, &__L, dims_info.x);
            frag_head_ptr += dims_info.x * f_mgr.frag_len;
        }

        frag_info.y = f_mgr.frag_left_over;
        fut[f_mgr.frag_num - 1] = decx::thread_pool.register_task(
            decx::fill_c_2D_int32, frag_head_ptr, _value, frag_info, &__L, dims_info.x);
    }
    else {
        decx::utils::_left_8 __L(NULL, actual_r_len % 8);
        uint2 frag_info = make_uint2(dims_info.x / 8, f_mgr.frag_len);
        float* frag_head_ptr = reinterpret_cast<float*>(src);

        for (int i = 0; i < f_mgr.frag_num; ++i) {
            fut[i] = decx::thread_pool.register_task(
                decx::fill_c_2D_int32, frag_head_ptr, _value, frag_info, &__L, dims_info.x);
            frag_head_ptr += dims_info.x * f_mgr.frag_len;
        }
    }

    for (int i = 0; i < conc_thr; ++i) {
        fut[i].get();
    }

    delete[] fut;
}



#endif