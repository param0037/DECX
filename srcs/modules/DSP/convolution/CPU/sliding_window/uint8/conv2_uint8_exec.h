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

#ifndef _CONV2_UINT8_EXEC_H_
#define _CONV2_UINT8_EXEC_H_


#include "../../../../../core/thread_management/thread_pool.h"
#include "../../../../../core/thread_management/thread_arrange.h"
#include "conv2_uint8_K_loop_core.h"
#include "../../../conv_utils.h"


namespace decx
{
    namespace conv {
        namespace CPUK {
            static _THREAD_CALL_ void _conv2_rN_rect_fixed_uint8_I32_ST(double* __restrict src, uint8_t* kernel, float* __restrict dst,
                const uint2 ker_dims, const uint Wsrc, const uint Wdst, const ushort reg_WL, const uint _loop);



            static _THREAD_CALL_ void _conv2_rN_rect_flex_uint8_I32_ST(double* __restrict src, uint8_t* kernel, float* __restrict dst,
                const uint2 proc_dim, const uint2 ker_dims, const uint Wsrc, const uint Wdst, const ushort reg_WL, const uint _loop);
        }
    }
}



_THREAD_CALL_
void decx::conv::CPUK::_conv2_rN_rect_fixed_uint8_I32_ST(double* __restrict   src, 
                                        uint8_t* __restrict   kernel, 
                                        float* __restrict   dst,
                                        const uint2         ker_dims,
                                        const uint          Wsrc,
                                        const uint          Wdst,
                                        const ushort        reg_WL,
                                        const uint          _loop)
{
    decx::conv::_v256_2i32 res_vec8;
    size_t dex_src = 0, dex_dst = 0;

    for (int i = 0; i < _BLOCKED_CONV2_UINT8_H_; ++i) {
#pragma unroll _BLOCKED_CONV2_UINT8_W_
        for (int j = 0; j < _BLOCKED_CONV2_UINT8_W_ / 2; ++j) {
            res_vec8 = decx::conv::CPUK::_conv2_uint8_i32_loop_in_kernel_16(src + dex_src, kernel, ker_dims, reg_WL, Wsrc, _loop);
            _mm256_store_ps(dst + dex_dst, _mm256_castsi256_ps(res_vec8._v1));
            _mm256_store_ps(dst + dex_dst + 8, _mm256_castsi256_ps(res_vec8._v2));
            dex_src += 2;
            dex_dst += 16;
        }
        dex_dst += (Wdst - _BLOCKED_CONV2_UINT8_W_) << 3;
        dex_src += Wsrc - _BLOCKED_CONV2_UINT8_W_;
    }
}



_THREAD_CALL_ void 
decx::conv::CPUK::_conv2_rN_rect_flex_uint8_I32_ST(double* __restrict   src, 
                                       uint8_t* __restrict   kernel, 
                                       float* __restrict   dst,
                                       const uint2         proc_dim,
                                       const uint2         ker_dims,
                                       const uint          Wsrc,
                                       const uint          Wdst,
                                       const ushort        reg_WL,
                                       const uint          _loop)
{
    decx::conv::_v256_2i32 res_vec8;
    size_t dex_src = 0, dex_dst = 0;

    const bool is_L8 = proc_dim.x % 2;

    for (int i = 0; i < proc_dim.y; ++i) {
        for (int j = 0; j < proc_dim.x / 2; ++j) {
            res_vec8 = decx::conv::CPUK::_conv2_uint8_i32_loop_in_kernel_16(src + dex_src, kernel, ker_dims, reg_WL, Wsrc, _loop);
            _mm256_store_ps(dst + dex_dst, _mm256_castsi256_ps(res_vec8._v1));
            _mm256_store_ps(dst + dex_dst + 8, _mm256_castsi256_ps(res_vec8._v2));
            dex_src += 2;
            dex_dst += 16;
        }
        if (is_L8) {
            __m256i res = decx::conv::CPUK::_conv2_uint8_i32_loop_in_kernel_8(src + dex_src, kernel, ker_dims, reg_WL, Wsrc, _loop);
            _mm256_store_ps(dst + dex_dst, _mm256_castsi256_ps(res));
            ++dex_src;
            dex_dst += 8;
        }
        dex_dst += (Wdst - proc_dim.x) << 3;
        dex_src += Wsrc - proc_dim.x;
    }
}



namespace decx
{
    namespace conv {
        namespace CPUK {
            static _THREAD_CALL_ void _conv2_rN_rect_fixed_uint8_F32_ST(double* __restrict src, float* kernel, float* __restrict dst,
                const uint2 ker_dims, const uint Wsrc, const uint Wdst, const ushort reg_WL, const uint _loop);



            static _THREAD_CALL_ void _conv2_rN_rect_flex_uint8_F32_ST(double* __restrict src, float* kernel, float* __restrict dst,
                const uint2 proc_dim, const uint2 ker_dims, const uint Wsrc, const uint Wdst, const ushort reg_WL, const uint _loop);



            static _THREAD_CALL_ void _conv2_rN_rect_fixed_uint8_uint8_ST(double* __restrict src, float* kernel, double* __restrict dst,
                const uint2 ker_dims, const uint Wsrc, const uint Wdst, const ushort reg_WL, const uint _loop);



            static _THREAD_CALL_ void _conv2_rN_rect_flex_uint8_uint8_ST(double* __restrict src, float* kernel, double* __restrict dst,
                const uint2 proc_dim, const uint2 ker_dims, const uint Wsrc, const uint Wdst, const ushort reg_WL, const uint _loop);
        }
    }
}



_THREAD_CALL_
void decx::conv::CPUK::_conv2_rN_rect_fixed_uint8_F32_ST(double* __restrict   src, 
                                        float* __restrict   kernel, 
                                        float* __restrict   dst,
                                        const uint2         ker_dims,
                                        const uint          Wsrc,
                                        const uint          Wdst,
                                        const ushort        reg_WL,
                                        const uint          _loop)
{
    decx::conv::_v256_2f32 res_vec8;
    size_t dex_src = 0, dex_dst = 0;

    for (int i = 0; i < _BLOCKED_CONV2_UINT8_H_; ++i) {
#pragma unroll _BLOCKED_CONV2_UINT8_W_
        for (int j = 0; j < _BLOCKED_CONV2_UINT8_W_ / 2; ++j) {
            res_vec8 = decx::conv::CPUK::_conv2_uint8_f32_loop_in_kernel_16(src + dex_src, kernel, ker_dims, reg_WL, Wsrc, _loop);
            _mm256_store_ps(dst + dex_dst, res_vec8._v1);
            _mm256_store_ps(dst + dex_dst + 8, res_vec8._v2);
            dex_src += 2;
            dex_dst += 16;
        }
        dex_dst += (Wdst << 3) - (_BLOCKED_CONV2_UINT8_W_ << 3);
        dex_src += Wsrc - _BLOCKED_CONV2_UINT8_W_;
    }
}



_THREAD_CALL_
void decx::conv::CPUK::_conv2_rN_rect_fixed_uint8_uint8_ST(double* __restrict   src, 
                                        float* __restrict   kernel, 
                                        double* __restrict   dst,
                                        const uint2         ker_dims,
                                        const uint          Wsrc,
                                        const uint          Wdst,
                                        const ushort        reg_WL,
                                        const uint          _loop)
{
    decx::conv::_v256_2f32 res_vec8;
    size_t dex_src = 0, dex_dst = 0;

    __m256i _iv1, _iv2;

    for (int i = 0; i < _BLOCKED_CONV2_UINT8_H_; ++i) {
#pragma unroll _BLOCKED_CONV2_UINT8_W_
        for (int j = 0; j < _BLOCKED_CONV2_UINT8_W_; ++j) {
            res_vec8 = decx::conv::CPUK::_conv2_uint8_f32_loop_in_kernel_16(src + dex_src, kernel, ker_dims, reg_WL, Wsrc << 1, _loop);
            _iv1 = _mm256_cvtps_epi32(res_vec8._v1);
            _iv2 = _mm256_cvtps_epi32(res_vec8._v2);
            _iv1 = _mm256_packs_epi32(_iv1, _iv2);
            _iv2 = _mm256_permutevar8x32_epi32(_mm256_packus_epi16(_iv1, _iv1), _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));

            _mm_store_pd(dst + dex_dst, _mm_castsi128_pd(_mm256_castsi256_si128(_iv2)));

            dex_src += 2;
            dex_dst += 2;
        }
        dex_dst += (Wdst - _BLOCKED_CONV2_UINT8_W_) * 2;
        dex_src += (Wsrc - _BLOCKED_CONV2_UINT8_W_) * 2;
    }
}




_THREAD_CALL_
void decx::conv::CPUK::_conv2_rN_rect_flex_uint8_F32_ST(double* __restrict   src, 
                                       float* __restrict   kernel, 
                                       float* __restrict   dst,
                                       const uint2         proc_dim,
                                       const uint2         ker_dims,
                                       const uint          Wsrc,
                                       const uint          Wdst,
                                       const ushort        reg_WL,
                                       const uint          _loop)
{
    decx::conv::_v256_2f32 res_vec8;
    size_t dex_src = 0, dex_dst = 0;

    const bool is_L8 = proc_dim.x % 2;

    for (int i = 0; i < proc_dim.y; ++i) {
        for (int j = 0; j < proc_dim.x / 2; ++j) {
            res_vec8 = decx::conv::CPUK::_conv2_uint8_f32_loop_in_kernel_16(src + dex_src, kernel, ker_dims, reg_WL, Wsrc, _loop);
            _mm256_store_ps(dst + dex_dst, res_vec8._v1);
            _mm256_store_ps(dst + dex_dst + 8, res_vec8._v2);
            dex_src += 2;
            dex_dst += 16;
        }
        if (is_L8) {
            __m256 res = decx::conv::CPUK::_conv2_uint8_f32_loop_in_kernel_8(src + dex_src, kernel, ker_dims, reg_WL, Wsrc, _loop);
            _mm256_store_ps(dst + dex_dst, res);
            ++dex_src;
            dex_dst += 8;
        }
        dex_dst += (Wdst << 3) - (proc_dim.x << 3);
        dex_src += Wsrc - proc_dim.x;
    }
}



_THREAD_CALL_
void decx::conv::CPUK::_conv2_rN_rect_flex_uint8_uint8_ST(double* __restrict   src, 
                                       float* __restrict   kernel, 
                                       double* __restrict   dst,
                                       const uint2         proc_dim,
                                       const uint2         ker_dims,
                                       const uint          Wsrc,
                                       const uint          Wdst,
                                       const ushort        reg_WL,
                                       const uint          _loop)
{
    decx::conv::_v256_2f32 res_vec8;
    size_t dex_src = 0, dex_dst = 0;

    __m256i _iv1, _iv2;

    for (int i = 0; i < proc_dim.y; ++i) {
        for (int j = 0; j < proc_dim.x; ++j) {
            res_vec8 = decx::conv::CPUK::_conv2_uint8_f32_loop_in_kernel_16(src + dex_src, kernel, ker_dims, reg_WL, Wsrc << 1, _loop);
            _iv1 = _mm256_cvtps_epi32(res_vec8._v1);
            _iv2 = _mm256_cvtps_epi32(res_vec8._v2);
            _iv1 = _mm256_packs_epi32(_iv1, _iv2);
            _iv2 = _mm256_permutevar8x32_epi32(_mm256_packus_epi16(_iv1, _iv1), _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));

            _mm_store_pd(dst + dex_dst, _mm_castsi128_pd(_mm256_castsi256_si128(_iv2)));

            dex_src += 2;
            dex_dst += 2;
        }
        dex_dst += (Wdst - proc_dim.x) * 2;
        dex_src += (Wsrc - proc_dim.x) * 2;
    }
}



namespace decx
{
    /*
    * This function is suitable only when 3 __m256 are needed to cover all the data on one row
    * which indicates that half_kerdim.x -> [5, 8]
    * 
    * @param proc_dim : .x -> in _m256, the width of proccess area of signle thread (on dst matrix)
    *                   .y -> the height of proccess area of signle thread (on dst matrix)
    * @param ker_dim : .x -> the width of kernel (in float); .y -> the height of kernel
    * @param reg_WL : the leftover on width. ( = (half_ker_dims.x * 2 + 8) - 2 * 8)
    * @param Wsrc : the pitch of src matrix, in __m256
    */
    _THREAD_FUNCTION_
    void _conv2_rN_uint8_I32_ST(double* src, uint8_t* kernel, float* dst, const uint2 proc_dim, const uint2 ker_dims, 
        const uint Wsrc, const uint Wdst, const ushort reg_WL, const uint _loop);


    _THREAD_FUNCTION_
    void _conv2_rN_uint8_F32_ST(double* src, float* kernel, float* dst, const uint2 proc_dim, const uint2 ker_dims, 
        const uint Wsrc, const uint Wdst, const ushort reg_WL, const uint _loop);


    _THREAD_FUNCTION_
    void _conv2_rN_uint8_uint8_ST(double* src, float* kernel, double* dst, const uint2 proc_dim, const uint2 ker_dims, 
        const uint Wsrc, const uint Wdst, const ushort reg_WL, const uint _loop);
}



namespace decx
{
    namespace conv{
        void _conv2_rN_uint8_I32_caller(double* src, uint8_t* kernel, float* dst, const uint2 proc_dim, const uint2 ker_dims,
            const uint Wsrc, const uint Wdst, const ushort reg_WL,
            decx::utils::_thr_1D* t1D, decx::utils::frag_manager* f_mgr,
            const uint _loop);


        void _conv2_rN_uint8_F32_caller(double* src, float* kernel, float* dst, const uint2 proc_dim, const uint2 ker_dims,
            const uint Wsrc, const uint Wdst, const ushort reg_WL,
            decx::utils::_thr_1D* t1D, decx::utils::frag_manager* f_mgr,
            const uint _loop);


        void _conv2_rN_uint8_uint8_caller(double* src, float* kernel, double* dst, const uint2 proc_dim, const uint2 ker_dims,
            const uint Wsrc, const uint Wdst, const ushort reg_WL,
            decx::utils::_thr_1D* t1D, decx::utils::frag_manager* f_mgr,
            const uint _loop);
    }
}



namespace decx
{
    namespace conv {
        /**
        * @param proc_dim : ~.x -> dst->pitch (in its element); ~.y -> dst->height
        * @param Wsrc : Pitch of source matrix, in its element (uint8_t)
        * @param Wsrc : Pitch of destinated matrix, in its element (uint8_t)
        */
        static void _conv2_r8_uint8_organiser(double* src, void* kernel, void* dst,
            const uint2 proc_dim, const uint2 ker_dims, const uint Wsrc,
            const uint Wdst, decx::utils::_thr_1D* t1D, decx::utils::frag_manager* f_mgr, 
            const int _output_type);
    }
}



void decx::conv::_conv2_r8_uint8_organiser(double*                            src,
                                           void*                              kernel, 
                                           void*                              dst, 
                                           const uint2                        proc_dim, 
                                           const uint2                        ker_dims,
                                           const uint                         Wsrc,
                                           const uint                         Wdst,
                                           decx::utils::_thr_1D*              t1D,
                                           decx::utils::frag_manager*         f_mgr,
                                           const int                          _output_type)
{
    const uint _loop = (ker_dims.x - 1) / 16;
    ushort reg_WL = (ushort)(ker_dims.x - _loop * 16);
    uint2 _proc_dims;
    switch (_output_type)
    {
    case de::_DATA_TYPES_FLAGS_::_INT32_:
        _proc_dims = make_uint2(proc_dim.x / 8, proc_dim.y);
        decx::conv::_conv2_rN_uint8_I32_caller(src, (uint8_t*)kernel, (float*)dst, _proc_dims, ker_dims, Wsrc / 8, Wdst / 8, reg_WL, t1D, f_mgr, _loop);
        break;

    case de::_DATA_TYPES_FLAGS_::_FP32_:
        _proc_dims = make_uint2(proc_dim.x / 8, proc_dim.y);
        decx::conv::_conv2_rN_uint8_F32_caller(src, (float*)kernel, (float*)dst, _proc_dims, ker_dims, Wsrc / 8, Wdst / 8, reg_WL, t1D, f_mgr, _loop);
        break;

    case de::_DATA_TYPES_FLAGS_::_UINT8_:
        _proc_dims = make_uint2(proc_dim.x / 16, proc_dim.y);
        decx::conv::_conv2_rN_uint8_uint8_caller(src, (float*)kernel, (double*)dst, _proc_dims, ker_dims, Wsrc / 16, Wdst / 16, reg_WL, t1D, f_mgr, _loop);
        break;

    default:
        break;
    }
}




#endif