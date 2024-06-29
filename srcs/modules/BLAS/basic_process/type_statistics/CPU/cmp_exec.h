/**
*   ----------------------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ----------------------------------------------------------------------------------
* 
* This is a part of the open source project named "DECX", a high-performance scientific
* computational library. This project follows the MIT License. For more information 
* please visit https://github.com/param0037/DECX.
* 
* Copyright (c) 2021 Wayne Anderson
* 
* Permission is hereby granted, free of charge, to any person obtaining a copy of this 
* software and associated documentation files (the "Software"), to deal in the Software 
* without restriction, including without limitation the rights to use, copy, modify, 
* merge, publish, distribute, sublicense, and/or sell copies of the Software, and to 
* permit persons to whom the Software is furnished to do so, subject to the following 
* conditions:
* 
* The above copyright notice and this permission notice shall be included in all copies 
* or substantial portions of the Software.
* 
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
* INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR 
* PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE 
* FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR 
* OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
* DEALINGS IN THE SOFTWARE.
*/


#ifndef _CMP_EXEC_H_
#define _CMP_EXEC_H_


#include "../../../../core/thread_management/thread_pool.h"
#include "../../../../core/thread_management/thread_arrange.h"
#include "../../../../core/utils/fragment_arrangment.h"
#include "../../../../core/allocators.h"


namespace decx
{
    namespace bp
    {
        namespace CPUK 
        {
            static __m256i extend_shufflevar_v8(const uint8_t _occupyied_L);

            static __m256i extend_shufflevar_v4(const uint8_t _occupyied_L);

            static __m128i extend_shufflevar_v16(const uint8_t _occupyied_L);

            /*
            * @param src : the read-only memory
            * @param len : the proccess length of single thread, in __m256
            * @param res_vec : the result vector in __m256
            */
            _THREAD_FUNCTION_ void
                _maximum_vec8_fp32_1D(const float* src, const size_t len, float* res_vec, const uint8_t _occupied_length);


            _THREAD_FUNCTION_ void
                _minimum_vec8_fp32_1D(const float* src, const size_t len, float* res_vec, const uint8_t _occupied_length);


            _THREAD_FUNCTION_ void
                _min_max_vec8_fp32_1D(const float* src, const size_t len, float* res_min, float* res_max, const uint8_t _occupied_length);


            /*
            * @param src : the read-only memory
            * @param len : the proccess length of single thread, in __m256
            * @param res_vec : the result vector in __m256
            * @param _occupied_length : the logical width of source matrix (src->Width()) % 8
            */
            _THREAD_FUNCTION_ void
                _maximum_vec8_fp32_2D(const float* src, const uint2 _proc_dims, float* res_vec,
                    const uint32_t Wsrc, const uint8_t _occupied_length);

            _THREAD_FUNCTION_ void
                _minimum_vec8_fp32_2D(const float* src, const uint2 _proc_dims, float* res_vec,
                    const uint32_t Wsrc, const uint8_t _occupied_length);

            _THREAD_FUNCTION_ void
                _min_max_vec8_fp32_2D(const float* src, const uint2 _proc_dims, float* res_min,
                    float* res_max, const uint32_t Wsrc, const uint8_t _occupied_length);


            /*
            * @param src : the read-only memory
            * @param len : the proccess length of single thread, in __m256
            * @param res_vec : the result vector in __m256
            * @param _occupied_length : the logical width of source matrix (src->Width()) % 8
            */
            _THREAD_FUNCTION_ void
                _maximum_vec16_uint8_2D(const uint8_t* src, const uint2 _proc_dims, uint8_t* res_vec,
                    const uint32_t Wsrc, const uint8_t _occupied_length);


            _THREAD_FUNCTION_ void
                _minimum_vec16_uint8_2D(const uint8_t* src, const uint2 _proc_dims, uint8_t* res_vec,
                    const uint32_t Wsrc, const uint8_t _occupied_length);


            _THREAD_FUNCTION_ void
                _min_max_vec16_uint8_2D(const uint8_t* src, const uint2 _proc_dims, uint8_t* res_min, uint8_t* res_max,
                    const uint32_t Wsrc, const uint8_t _occupied_length);


            _THREAD_FUNCTION_ void
                _maximum_vec16_uint8_1D(const uint8_t* src, const uint64_t len, uint8_t* res_vec, const uint8_t _occupied_length);


            _THREAD_FUNCTION_ void
                _minimum_vec16_uint8_1D(const uint8_t* src, const uint64_t len, uint8_t* res_vec, const uint8_t _occupied_length);


            _THREAD_FUNCTION_ void
                _min_max_vec16_uint8_1D(const uint8_t* src, const uint64_t len, uint8_t* rea_min, uint8_t* res_max, const uint8_t _occupied_length);


            /*
            * @param src : the read-only memory
            * @param len : the proccess length of single thread, in __m256
            * @param res_vec : the result vector in __m256
            * @param _occupied_length : the logical width of source matrix (src->Width()) % 8
            */
            _THREAD_FUNCTION_ void
                _maximum_vec4_fp64_2D(const double* src, const uint2 _proc_dims, double* res_vec,
                    const uint32_t Wsrc, const uint8_t _occupied_length);


            _THREAD_FUNCTION_ void
                _minimum_vec4_fp64_2D(const double* src, const uint2 _proc_dims, double* res_vec,
                    const uint32_t Wsrc, const uint8_t _occupied_length);

            _THREAD_FUNCTION_ void
                _min_max_vec4_fp64_2D(const double* src, const uint2 _proc_dims, double* res_min,
                    double* res_max, const uint32_t Wsrc, const uint8_t _occupied_length);


            /*
            * @param src : the read-only memory
            * @param len : the proccess length of single thread, in __m256
            * @param res_vec : the result vector in __m256
            */
            _THREAD_FUNCTION_ void
                _maximum_vec4_fp64_1D(const double* src, const size_t len, double* res_vec, const uint8_t _occupied_length);


            _THREAD_FUNCTION_ void
                _minimum_vec4_fp64_1D(const double* src, const size_t len, double* res_vec, const uint8_t _occupied_length);


            _THREAD_FUNCTION_ void
                _min_max_vec4_fp64_1D(const double* src, const size_t len, double* res_min, double* res_max, const uint8_t _occupied_length);


            typedef void (*_cmp_kernel_fp32_1D) (const float*, const size_t, float*, const uint8_t);
            typedef void (*_cmp_kernel_fp64_1D) (const double*, const size_t, double*, const uint8_t);
            typedef void (*_cmp_kernel_uint8_1D) (const uint8_t*, const size_t, uint8_t*, const uint8_t);

            typedef void (*_bicmp_kernel_fp32_1D) (const float*, const size_t, float*, float*, const uint8_t);
            typedef void (*_bicmp_kernel_fp64_1D) (const double*, const size_t, double*, double*, const uint8_t);
            typedef void (*_bicmp_kernel_uint8_1D) (const uint8_t*, const size_t, uint8_t*, uint8_t*, const uint8_t);

            typedef void (*_cmp_kernel_fp32_2D) (const float*, const uint2, float*, const uint32_t, const uint8_t);
            typedef void (*_cmp_kernel_fp64_2D) (const double*, const uint2, double*, const uint32_t, const uint8_t);
            typedef void (*_cmp_kernel_uint8_2D) (const uint8_t*, const uint2, uint8_t*, const uint32_t, const uint8_t);

            typedef void (*_bicmp_kernel_fp32_2D) (const float*, const uint2, float*, float*, const uint32_t, const uint8_t);
            typedef void (*_bicmp_kernel_fp64_2D) (const double*, const uint2, double*, double*, const uint32_t, const uint8_t);
            typedef void (*_bicmp_kernel_uint8_2D) (const uint8_t*, const uint2, uint8_t*, uint8_t*, const uint32_t, const uint8_t);
        }

        /*
        * @param src : the read-only memory
        * @param len : the proccess length of single thread, in __m256
        * @param res_vec : the result vector in __m256
        */
        template <typename T_kernel, typename T_data, uint8_t _align>
        static void _maximum_1D_caller(T_kernel _cmp_kernel, const T_data* src, const size_t len, T_data* res_vec);


        template <typename T_kernel, typename T_data, uint8_t _align>
        static void _minimum_1D_caller(T_kernel _cmp_kernel, const T_data* src, const size_t len, T_data* res_vec);


        template <typename T_kernel, typename T_data, uint8_t _align>
        static void _min_max_1D_caller(T_kernel _cmp_kernel, const T_data* src, const size_t len, T_data* res_min, T_data* res_max);


        /*
        * @param src : the read-only memory
        * @param len : the proccess length of single thread, in __m256
        * @param res_vec : the result vector in __m256
        */
        template <typename T_kernel, typename T_data, uint8_t _align>
        void _maximum_2D_caller(T_kernel _cmp_kernel, const T_data* src, const uint2 proc_dims,
            const uint32_t Wsrc, T_data* res_vec);


        template <typename T_kernel, typename T_data, uint8_t _align>
        void _minimum_2D_caller(T_kernel _cmp_kernel, const T_data* src, const uint2 proc_dims,
            const uint32_t Wsrc, T_data* res_vec);


        template <typename T_kernel, typename T_data, uint8_t _align>
        void _min_max_2D_caller(T_kernel _cmp_kernel, const T_data* src, const uint2 proc_dims,
            const uint32_t Wsrc, T_data* res_min, T_data* res_max);
    }
}


#define _CMP_THREAD_LANE_(_operator, _vec_in, res, vec_length) {        \
    res = _vec_in[0];                                                   \
    for (uint32_t i = 1; i < vec_length; ++i) {                         \
        res = ((_vec_in)[i] _operator res) ? (_vec_in)[i] : res;        \
    }                                                                   \
}


static inline __m256i decx::bp::CPUK::extend_shufflevar_v8(const uint8_t _occupyied_L)
{
    const uint16_t index_bound = _occupyied_L - 1;
    __m256i _shuffle_var = _mm256_setr_epi32(decx::utils::clamp_max<uint8_t>(0, index_bound),
        decx::utils::clamp_max<uint8_t>(1, index_bound), 
        decx::utils::clamp_max<uint8_t>(2, index_bound), 
        decx::utils::clamp_max<uint8_t>(3, index_bound), 
        decx::utils::clamp_max<uint8_t>(4, index_bound), 
        decx::utils::clamp_max<uint8_t>(5, index_bound), 
        decx::utils::clamp_max<uint8_t>(6, index_bound), 
        decx::utils::clamp_max<uint8_t>(7, index_bound));

    return _shuffle_var;
}



static inline __m256i decx::bp::CPUK::extend_shufflevar_v4(const uint8_t _occupyied_L)
{
    const uint16_t index_bound = _occupyied_L - 1;
    __m256i _shuffle_var = _mm256_setr_epi32(decx::utils::clamp_max<uint8_t>(0, index_bound) * 2,
        decx::utils::clamp_max<uint8_t>(0, index_bound) * 2 + 1,
        decx::utils::clamp_max<uint8_t>(1, index_bound) * 2,
        decx::utils::clamp_max<uint8_t>(1, index_bound) * 2 + 1,
        decx::utils::clamp_max<uint8_t>(2, index_bound) * 2,
        decx::utils::clamp_max<uint8_t>(2, index_bound) * 2 + 1,
        decx::utils::clamp_max<uint8_t>(3, index_bound) * 2,
        decx::utils::clamp_max<uint8_t>(3, index_bound) * 2 + 1);

    return _shuffle_var;
}



static inline __m128i decx::bp::CPUK::extend_shufflevar_v16(const uint8_t _occupyied_L)
{
    const uint16_t index_bound = _occupyied_L - 1;
    __m128i _shuffle_var = _mm_setr_epi8(decx::utils::clamp_max<uint8_t>(0, index_bound), decx::utils::clamp_max<uint8_t>(1, index_bound),
        decx::utils::clamp_max<uint8_t>(2, index_bound), decx::utils::clamp_max<uint8_t>(3, index_bound),
        decx::utils::clamp_max<uint8_t>(4, index_bound), decx::utils::clamp_max<uint8_t>(5, index_bound),
        decx::utils::clamp_max<uint8_t>(6, index_bound), decx::utils::clamp_max<uint8_t>(7, index_bound),
        decx::utils::clamp_max<uint8_t>(8, index_bound), decx::utils::clamp_max<uint8_t>(9, index_bound),
        decx::utils::clamp_max<uint8_t>(10, index_bound), decx::utils::clamp_max<uint8_t>(11, index_bound),
        decx::utils::clamp_max<uint8_t>(12, index_bound), decx::utils::clamp_max<uint8_t>(13, index_bound),
        decx::utils::clamp_max<uint8_t>(14, index_bound), decx::utils::clamp_max<uint8_t>(15, index_bound));

    return _shuffle_var;
}



template <typename T_kernel, typename T_data, uint8_t _align>
void decx::bp::_maximum_1D_caller(T_kernel _cmp_kernel, const T_data* src, const size_t len, T_data* res_vec)
{
    // the number of available concurrent threads
    const uint conc_thr = decx::cpu::_get_permitted_concurrency();
    decx::utils::frag_manager fr_mgr;
    decx::utils::frag_manager_gen(&fr_mgr, decx::utils::ceil<uint64_t>(len, _align), conc_thr);

    decx::utils::_thread_arrange_1D t1D(conc_thr);
    T_data* res_arr = new T_data[conc_thr];
    const uint8_t _occupied_length = (len % _align);

    const T_data* tmp_src = src;
    const uint64_t proc_len = fr_mgr.frag_len * _align;
    for (int i = 0; i < conc_thr - 1; ++i) {
        t1D._async_thread[i] = decx::cpu::register_task_default(
            _cmp_kernel, tmp_src, proc_len / _align, res_arr + i, _occupied_length);
        tmp_src += proc_len;
    }
    const uint64_t _L = fr_mgr.is_left ? fr_mgr.frag_left_over : fr_mgr.frag_len;
    t1D._async_thread[conc_thr - 1] = decx::cpu::register_task_default(
        _cmp_kernel, tmp_src, _L, res_arr + conc_thr - 1, _occupied_length);

    t1D.__sync_all_threads();

    T_data res;
    _CMP_THREAD_LANE_(>, res_arr, res, conc_thr);
    *res_vec = res;

    delete[] res_arr;
}




template <typename T_kernel, typename T_data, uint8_t _align>
void decx::bp::_minimum_1D_caller(T_kernel _cmp_kernel, const T_data* src, const size_t len, T_data* res_vec)
{
    // the number of available concurrent threads
    const uint conc_thr = decx::cpu::_get_permitted_concurrency();
    decx::utils::frag_manager fr_mgr;
    decx::utils::frag_manager_gen(&fr_mgr, decx::utils::ceil<uint64_t>(len, _align), conc_thr);

    decx::utils::_thread_arrange_1D t1D(conc_thr);
    T_data* res_arr = new T_data[conc_thr];
    const uint8_t _occupied_length = (len % _align);

    const T_data* tmp_src = src;
    const uint64_t proc_len = fr_mgr.frag_len * _align;
    for (int i = 0; i < conc_thr - 1; ++i) {
        t1D._async_thread[i] = decx::cpu::register_task_default(
            _cmp_kernel, tmp_src, proc_len / _align, res_arr + i, _occupied_length);
        tmp_src += proc_len;
    }
    const uint64_t _L = fr_mgr.is_left ? fr_mgr.frag_left_over : fr_mgr.frag_len;
    t1D._async_thread[conc_thr - 1] = decx::cpu::register_task_default(
        _cmp_kernel, tmp_src, _L, res_arr + conc_thr - 1, _occupied_length);

    t1D.__sync_all_threads();

    T_data res;
    _CMP_THREAD_LANE_(< , res_arr, res, conc_thr);
    *res_vec = res;

    delete[] res_arr;
}



template <typename T_kernel, typename T_data, uint8_t _align>
static void decx::bp::_min_max_1D_caller(T_kernel _cmp_kernel, const T_data* src, const size_t len, T_data* res_min, T_data* res_max)
{
    // the number of available concurrent threads
    const uint conc_thr = decx::cpu::_get_permitted_concurrency();
    decx::utils::frag_manager fr_mgr;
    decx::utils::frag_manager_gen(&fr_mgr, decx::utils::ceil<uint64_t>(len, _align), conc_thr);

    decx::utils::_thread_arrange_1D t1D(conc_thr);
    decx::PtrInfo<T_data> vec_min, vec_max;
    if (decx::alloc::_host_virtual_page_malloc(&vec_min, conc_thr * sizeof(T_data))) {
        return;
    }
    if (decx::alloc::_host_virtual_page_malloc(&vec_max, conc_thr * sizeof(T_data))) {
        return;
    }
    const uint8_t _occupied_length = (len % _align);

    const T_data* tmp_src = src;
    const uint64_t proc_len = fr_mgr.frag_len * _align;
    for (int i = 0; i < conc_thr - 1; ++i) {
        t1D._async_thread[i] = decx::cpu::register_task_default(
            _cmp_kernel, tmp_src, proc_len / _align, vec_min.ptr + i, vec_max.ptr + i, _occupied_length);
        tmp_src += proc_len;
    }
    const uint64_t _L = fr_mgr.is_left ? fr_mgr.frag_left_over : fr_mgr.frag_len;
    t1D._async_thread[conc_thr - 1] = decx::cpu::register_task_default(
        _cmp_kernel, tmp_src, _L, vec_min.ptr + conc_thr - 1, vec_max.ptr + conc_thr - 1, _occupied_length);

    t1D.__sync_all_threads();

    T_data res;
    _CMP_THREAD_LANE_(< , vec_min.ptr, res, conc_thr);
    *res_min = res;

    _CMP_THREAD_LANE_(> , vec_max.ptr, res, conc_thr);
    *res_max = res;

    decx::alloc::_host_virtual_page_dealloc(&vec_min);
    decx::alloc::_host_virtual_page_dealloc(&vec_max);
}


template <typename T_kernel, typename T_data, uint8_t _align>
void decx::bp::_maximum_2D_caller(T_kernel          _cmp_kernel, 
                                  const T_data*     src, 
                                  const uint2       proc_dims,
                                  const uint32_t    Wsrc, 
                                  T_data*           res_vec)
{
    // the number of available concurrent threads
    const uint conc_thr = decx::cpu::_get_permitted_concurrency();
    decx::utils::frag_manager fr_mgr;
    decx::utils::frag_manager_gen(&fr_mgr, proc_dims.y, conc_thr);
    const uint8_t _occupied_length = (proc_dims.x % _align);

    decx::utils::_thread_arrange_1D t1D(conc_thr);
    T_data* res_arr = new T_data[conc_thr];

    const T_data* tmp_src = src;
    const size_t proc_size = fr_mgr.frag_len * Wsrc;
    for (int i = 0; i < conc_thr - 1; ++i)
    {
        t1D._async_thread[i] = decx::cpu::register_task_default(
            _cmp_kernel, tmp_src, make_uint2(decx::utils::ceil<uint32_t>(proc_dims.x, _align), fr_mgr.frag_len), res_arr + i, Wsrc, _occupied_length);
        tmp_src += proc_size;
    }
    const uint32_t _L = fr_mgr.is_left ? fr_mgr.frag_left_over : fr_mgr.frag_len;
    t1D._async_thread[conc_thr - 1] = decx::cpu::register_task_default(
        _cmp_kernel, tmp_src, make_uint2(decx::utils::ceil<uint32_t>(proc_dims.x, _align), _L), res_arr + conc_thr - 1, Wsrc, _occupied_length);

    t1D.__sync_all_threads();

    T_data res;
    _CMP_THREAD_LANE_(> , res_arr, res, conc_thr);
    *res_vec = res;

    delete[] res_arr;
}




template <typename T_kernel, typename T_data, uint8_t _align>
void decx::bp::_minimum_2D_caller(T_kernel _cmp_kernel, const T_data* src, const uint2 proc_dims,
    const uint32_t Wsrc, T_data* res_vec)
{
    // the number of available concurrent threads
    const uint conc_thr = decx::cpu::_get_permitted_concurrency();
    decx::utils::frag_manager fr_mgr;
    decx::utils::frag_manager_gen(&fr_mgr, proc_dims.y, conc_thr);
    const uint8_t _occupied_length = (proc_dims.x % _align);

    decx::utils::_thread_arrange_1D t1D(conc_thr);
    T_data* res_arr = new T_data[conc_thr];

    const T_data* tmp_src = src;
    const size_t proc_size = fr_mgr.frag_len * Wsrc;
    for (int i = 0; i < conc_thr - 1; ++i)
    {
        t1D._async_thread[i] = decx::cpu::register_task_default(
            _cmp_kernel, tmp_src, make_uint2(decx::utils::ceil<uint32_t>(proc_dims.x, _align), fr_mgr.frag_len), res_arr + i, Wsrc, _occupied_length);
        tmp_src += proc_size;
    }
    const uint32_t _L = fr_mgr.is_left ? fr_mgr.frag_left_over : fr_mgr.frag_len;
    t1D._async_thread[conc_thr - 1] = decx::cpu::register_task_default(
        _cmp_kernel, tmp_src, make_uint2(decx::utils::ceil<uint32_t>(proc_dims.x, _align), _L), res_arr + conc_thr - 1, Wsrc, _occupied_length);

    t1D.__sync_all_threads();

    T_data res;
    _CMP_THREAD_LANE_(< , res_arr, res, conc_thr);
    *res_vec = res;

    delete[] res_arr;
}




template <typename T_kernel, typename T_data, uint8_t _align>
void decx::bp::_min_max_2D_caller(T_kernel _cmp_kernel, const T_data* src, const uint2 proc_dims,
    const uint32_t Wsrc, T_data* res_min, T_data* res_max)
{
    // the number of available concurrent threads
    const uint conc_thr = decx::cpu::_get_permitted_concurrency();
    decx::utils::frag_manager fr_mgr;
    decx::utils::frag_manager_gen(&fr_mgr, proc_dims.y, conc_thr);
    const uint8_t _occupied_length = (proc_dims.x % _align);

    decx::utils::_thread_arrange_1D t1D(conc_thr);
    decx::PtrInfo<T_data> vec_min, vec_max;
    if (decx::alloc::_host_virtual_page_malloc(&vec_min, conc_thr * sizeof(T_data))) {
        return;
    }
    if (decx::alloc::_host_virtual_page_malloc(&vec_max, conc_thr * sizeof(T_data))) {
        return;
    }

    const T_data* tmp_src = src;
    const size_t proc_size = fr_mgr.frag_len * Wsrc;
    for (int i = 0; i < conc_thr - 1; ++i)
    {
        t1D._async_thread[i] = decx::cpu::register_task_default(
            _cmp_kernel, tmp_src, make_uint2(decx::utils::ceil<uint32_t>(proc_dims.x, _align), fr_mgr.frag_len), 
            vec_min.ptr + i, vec_max.ptr + i, Wsrc, _occupied_length);
        tmp_src += proc_size;
    }
    const uint32_t _L = fr_mgr.is_left ? fr_mgr.frag_left_over : fr_mgr.frag_len;
    t1D._async_thread[conc_thr - 1] = decx::cpu::register_task_default(
        _cmp_kernel, tmp_src, make_uint2(decx::utils::ceil<uint32_t>(proc_dims.x, _align), _L), 
        vec_min.ptr + conc_thr - 1, vec_max.ptr + conc_thr - 1, Wsrc, _occupied_length);

    t1D.__sync_all_threads();

    T_data res;
    _CMP_THREAD_LANE_(< , vec_min.ptr, res, conc_thr);
    *res_min = res;

    _CMP_THREAD_LANE_(> , vec_max.ptr, res, conc_thr);
    *res_max = res;

    decx::alloc::_host_virtual_page_dealloc(&vec_min);
    decx::alloc::_host_virtual_page_dealloc(&vec_max);
}


#endif