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


// #include "_mm256_uint8_int32.h"
#include "typecast_exec_x86_64.h"


_THREAD_FUNCTION_ void
decx::type_cast::CPUK::_v256_cvtui8_i32_1D(const uint8_t* __restrict  src, 
                                           int32_t* __restrict        dst, 
                                           const size_t             proc_len)
{
    decx::utils::simd::xmm128_reg recv;
    decx::utils::simd::xmm256_reg store;

    for (int i = 0; i < proc_len; ++i) {
        recv._vf = _mm_loadu_ps((float*)(src + i * 8));
        store._vi = _mm256_cvtepu8_epi32(recv._vi);

        _mm256_store_ps((float*)dst + i * 8, store._vf);
    }
}


_THREAD_FUNCTION_ void decx::type_cast::CPUK::
_v256_cvti32_ui8_cyclic1D(const int32_t* __restrict    src, 
                          uint8_t* __restrict        dst, 
                          const uint64_t             proc_len)
{
    decx::utils::simd::xmm256_reg recv;
    __m256i reg0, reg1, _overflow_24;

    for (int32_t i = 0; i < proc_len; ++i) {
        recv._vf = _mm256_load_ps((float*)src + i * 8);

        _overflow_24 = _mm256_and_si256(recv._vi, _mm256_set1_epi32(0xffffff00));

        reg0 = _mm256_shuffle_epi8(_overflow_24,
            _mm256_setr_epi32(33751041, 101123077, 168495113, 235867149, 33751041, 101123077, 168495113, 235867149));

        reg1 = _mm256_slli_epi16(reg0, 4);

        reg1 = _mm256_or_si256(_overflow_24, reg0);
        reg0 = _mm256_slli_epi16(reg1, 4);
        reg1 = _mm256_or_si256(reg0, reg1);

        reg0 = _mm256_slli_epi16(reg1, 2);
        reg1 = _mm256_or_si256(reg0, reg1);

        reg0 = _mm256_slli_epi16(reg1, 1);
        reg1 = _mm256_or_si256(reg0, reg1);

        reg0 = _mm256_blendv_epi8(recv._vi, _mm256_set1_epi32(0), reg1);    // cycled_cast

        reg0 = _mm256_andnot_si256(_mm256_srai_epi32(recv._vi, 32), reg0);  // clamp negative integers to zero

        reg1 = _mm256_shuffle_epi8(reg0, _mm256_set1_epi32(201851904));     // in lane(128) shuffle

        *((uint32_t*)(dst + i * 8)) = _mm256_extract_epi32(reg1, 0);
        *((uint32_t*)(dst + i * 8 + 4)) = _mm256_extract_epi32(reg1, 4);
    }
}




_THREAD_FUNCTION_ void decx::type_cast::CPUK::
_v256_cvti32_ui8_saturated1D(const int32_t* __restrict      src, 
                             uint8_t* __restrict              dst, 
                             const uint64_t                 proc_len)
{
    decx::utils::simd::xmm256_reg recv;
    __m256i reg0, reg1, _overflow_24;

    for (int i = 0; i < proc_len; ++i) {
        recv._vf = _mm256_load_ps((float*)src + i * 8);

        _overflow_24 = _mm256_and_si256(recv._vi, _mm256_set1_epi32(0xffffff00));

        reg0 = _mm256_shuffle_epi8(_overflow_24,
            _mm256_setr_epi32(33751041, 101123077, 168495113, 235867149, 33751041, 101123077, 168495113, 235867149));

        reg1 = _mm256_slli_epi16(reg0, 4);

        reg1 = _mm256_or_si256(_overflow_24, reg0);
        reg0 = _mm256_slli_epi16(reg1, 4);
        reg1 = _mm256_or_si256(reg0, reg1);

        reg0 = _mm256_slli_epi16(reg1, 2);
        reg1 = _mm256_or_si256(reg0, reg1);

        reg0 = _mm256_slli_epi16(reg1, 1);
        reg1 = _mm256_or_si256(reg0, reg1);

        reg0 = _mm256_blendv_epi8(recv._vi, _mm256_set1_epi32(0xFFFFFFFF), reg1);    // cycled_cast

        reg0 = _mm256_andnot_si256(_mm256_srai_epi32(recv._vi, 32), reg0);  // clamp negative integers to zero

        reg1 = _mm256_shuffle_epi8(reg0, _mm256_set1_epi32(201851904));     // in lane(128) shuffle

        *((uint32_t*)(dst + i * 8)) = _mm256_extract_epi32(reg1, 0);
        *((uint32_t*)(dst + i * 8 + 4)) = _mm256_extract_epi32(reg1, 4);
    }
}


_THREAD_FUNCTION_ void
decx::type_cast::CPUK::_v256_cvti32_ui8_truncate1D(const int32_t* __restrict      src, 
                                                   uint8_t* __restrict              dst, 
                                                   const size_t                 proc_len)
{
    decx::utils::simd::xmm256_reg recv;
    __m256i reg;

    for (int i = 0; i < proc_len; ++i) {
        recv._vf = _mm256_load_ps((float*)src + i * 8);

        reg = _mm256_and_si256(recv._vi, _mm256_set1_epi32(0x000000ff));
        reg = _mm256_shuffle_epi8(reg, _mm256_set1_epi32(201851904));
        
        *((uint32_t*)(dst + i * 8)) = _mm256_extract_epi32(reg, 0);
        *((uint32_t*)(dst + i * 8 + 4)) = _mm256_extract_epi32(reg, 4);
    }
}



_THREAD_FUNCTION_ void decx::type_cast::CPUK::
_v256_cvti32_ui8_truncate_clamp_zero1D(const int32_t* __restrict      src, 
                                       uint8_t* __restrict            dst, 
                                       const uint64_t                 proc_len)
{
    decx::utils::simd::xmm256_reg recv;
    __m256i reg;

    for (int32_t i = 0; i < proc_len; ++i) {
        recv._vf = _mm256_load_ps((float*)src + i * 8);

        reg = _mm256_and_si256(recv._vi, _mm256_set1_epi32(0x000000ff));
        reg = _mm256_andnot_si256(_mm256_srai_epi32(recv._vi, 32), reg);
        reg = _mm256_shuffle_epi8(reg, _mm256_set1_epi32(201851904));
        
        *((uint32_t*)(dst + i * 8)) = _mm256_extract_epi32(reg, 0);
        *((uint32_t*)(dst + i * 8 + 4)) = _mm256_extract_epi32(reg, 4);
    }
}



// void decx::type_cast::_cvtui8_i32_caller1D(const float*         src, 
//                                            float*               dst, 
//                                            const size_t         proc_len)
// {
//     decx::utils::_thread_arrange_1D t1D(decx::cpu::_get_permitted_concurrency());

//     bool _is_MT = proc_len > 1024 * t1D.total_thread;

//     if (_is_MT) {
//         decx::utils::frag_manager f_mgr;
//         decx::utils::frag_manager_gen(&f_mgr, proc_len, t1D.total_thread);

//         const float* loc_src = src;
//         float* loc_dst = dst;

//         for (int i = 0; i < t1D.total_thread - 1; ++i) {
//             t1D._async_thread[i] = decx::cpu::register_task_default( decx::type_cast::CPUK::_v256_cvtui8_i32_1D,
//                 loc_src, loc_dst, f_mgr.frag_len);

//             loc_src += f_mgr.frag_len * 2;
//             loc_dst += f_mgr.frag_len * 8;
//         }
//         const size_t _L = f_mgr.is_left ? f_mgr.frag_left_over : f_mgr.frag_len;
//         t1D._async_thread[t1D.total_thread - 1] = decx::cpu::register_task_default( decx::type_cast::CPUK::_v256_cvtui8_i32_1D,
//             loc_src, loc_dst, _L);

//         t1D.__sync_all_threads();
//     }
//     else {
//         decx::type_cast::CPUK::_v256_cvtui8_i32_1D(src, dst, proc_len);
//     }
// }



decx::type_cast::CPUK::_cvt_kernel1D<int32_t, uint8_t>*
decx::type_cast::_cvti32_ui8_selector1D(const int32_t flag)
{
    using namespace decx::type_cast;
    decx::type_cast::CPUK::_cvt_kernel1D<int32_t, uint8_t>* exec_kernrel_ptr = NULL;

    switch (flag)
    {
    case (de::TypeCast_Method::CVT_INT32_UINT8 | 
            de::TypeCast_Method::CVT_UINT8_CLAMP_TO_ZERO | 
            de::TypeCast_Method::CVT_INT32_UINT8_TRUNCATE):        // 0b0011 (3)
        exec_kernrel_ptr = decx::type_cast::CPUK::_v256_cvti32_ui8_truncate_clamp_zero1D;
        break;

    case de::TypeCast_Method::CVT_INT32_UINT8 | de::TypeCast_Method::CVT_INT32_UINT8_TRUNCATE:      // 0b0010 (2)
        exec_kernrel_ptr = decx::type_cast::CPUK::_v256_cvti32_ui8_truncate1D;
        break;

    case (de::TypeCast_Method::CVT_INT32_UINT8 | de::TypeCast_Method::CVT_UINT8_CYCLIC):        // 0b0101 (5) or CVT_INT32_UINT8 | CVT_UINT8_CYCLIC | CVT_CLAMP_TO_ZERO
        exec_kernrel_ptr = decx::type_cast::CPUK::_v256_cvti32_ui8_cyclic1D;
        break;

    case (de::TypeCast_Method::CVT_INT32_UINT8 | de::TypeCast_Method::CVT_UINT8_SATURATED):     // 0b1001 (9) or CVT_INT32_UINT8 | CVT_UINT8_SATURATED | CVT_CLAMP_TO_ZERO
        exec_kernrel_ptr = decx::type_cast::CPUK::_v256_cvti32_ui8_saturated1D;
        break;
    }

    return exec_kernrel_ptr;
}


// ---------------------------------------------- 2D ---------------------------------------------------


_THREAD_FUNCTION_ void
decx::type_cast::CPUK::_v256_cvtui8_i32_2D(const uint8_t* __restrict  src, 
                                           int32_t* __restrict        dst, 
                                           const uint2              proc_dims, 
                                           const uint32_t           Wsrc, 
                                           const uint32_t           Wdst)
{
    decx::utils::simd::xmm128_reg recv;
    decx::utils::simd::xmm256_reg store;

    uint64_t dex_src = 0, dex_dst = 0;

    for (int i = 0; i < proc_dims.y; ++i) {
        dex_src = i * Wsrc;
        dex_dst = i * Wdst;
        for (int j = 0; j < proc_dims.x; ++j) {
            recv._vf = _mm_loadu_ps((float*)(src + dex_src));
            store._vi = _mm256_cvtepu8_epi32(recv._vi);

            _mm256_store_ps((float*)dst + dex_dst, store._vf);

            dex_src += 8;
            dex_dst += 8;
        }
    }
}



_THREAD_FUNCTION_ void
decx::type_cast::CPUK::_v256_cvti32_ui8_cyclic2D(const int32_t* __restrict    src, 
                                                 uint8_t* __restrict          dst, 
                                                 const uint2                  proc_dims,
                                                 const uint32_t               Wsrc,
                                                 const uint32_t               Wdst)
{
    decx::utils::simd::xmm256_reg recv;
    __m256i reg0, reg1, _overflow_24;

    uint64_t dex_src = 0, dex_dst = 0;

    for (int i = 0; i < proc_dims.y; ++i) {
        dex_src = i * Wsrc;
        dex_dst = i * Wdst;
        for (int j = 0; j < proc_dims.x; ++j) {
            recv._vf = _mm256_load_ps((float*)(src + dex_src));

            _overflow_24 = _mm256_and_si256(recv._vi, _mm256_set1_epi32(0xffffff00));

            reg0 = _mm256_shuffle_epi8(_overflow_24,
                _mm256_setr_epi32(33751041, 101123077, 168495113, 235867149, 33751041, 101123077, 168495113, 235867149));

            reg1 = _mm256_slli_epi16(reg0, 4);

            reg1 = _mm256_or_si256(_overflow_24, reg0);
            reg0 = _mm256_slli_epi16(reg1, 4);
            reg1 = _mm256_or_si256(reg0, reg1);

            reg0 = _mm256_slli_epi16(reg1, 2);
            reg1 = _mm256_or_si256(reg0, reg1);

            reg0 = _mm256_slli_epi16(reg1, 1);
            reg1 = _mm256_or_si256(reg0, reg1);

            reg0 = _mm256_blendv_epi8(recv._vi, _mm256_set1_epi32(0), reg1);    // cycled_cast

            reg0 = _mm256_andnot_si256(_mm256_srai_epi32(recv._vi, 32), reg0);  // clamp negative integers to zero

            reg1 = _mm256_shuffle_epi8(reg0, _mm256_set1_epi32(201851904));     // in lane(128) shuffle

            *((uint32_t*)(dst + dex_dst)) = _mm256_extract_epi32(reg1, 0);
            *((uint32_t*)(dst + dex_dst + 4)) = _mm256_extract_epi32(reg1, 4);

            dex_src += 8;
            dex_dst += 8;
        }
    }
}



_THREAD_FUNCTION_ void
decx::type_cast::CPUK::_v256_cvti32_ui8_saturated2D(const int32_t* __restrict      src, 
                                                    uint8_t* __restrict          dst, 
                                                    const uint2                  proc_dims,
                                                    const uint32_t              Wsrc,
                                                    const uint32_t              Wdst)
{
    decx::utils::simd::xmm256_reg recv;
    __m256i reg0, reg1, _overflow_24;

    uint64_t dex_src = 0, dex_dst = 0;

    for (int32_t i = 0; i < proc_dims.y; ++i) {
        dex_src = i * Wsrc;
        dex_dst = i * Wdst;
        for (int32_t j = 0; j < proc_dims.x; ++j) {
            recv._vf = _mm256_load_ps((float*)src + dex_src);

            _overflow_24 = _mm256_and_si256(recv._vi, _mm256_set1_epi32(0xffffff00));

            reg0 = _mm256_shuffle_epi8(_overflow_24,
                _mm256_setr_epi32(33751041, 101123077, 168495113, 235867149, 33751041, 101123077, 168495113, 235867149));

            reg1 = _mm256_slli_epi16(reg0, 4);

            reg1 = _mm256_or_si256(_overflow_24, reg0);
            reg0 = _mm256_slli_epi16(reg1, 4);
            reg1 = _mm256_or_si256(reg0, reg1);

            reg0 = _mm256_slli_epi16(reg1, 2);
            reg1 = _mm256_or_si256(reg0, reg1);

            reg0 = _mm256_slli_epi16(reg1, 1);
            reg1 = _mm256_or_si256(reg0, reg1);

            reg0 = _mm256_blendv_epi8(recv._vi, _mm256_set1_epi32(0xFFFFFFFF), reg1);    // cycled_cast

            reg0 = _mm256_andnot_si256(_mm256_srai_epi32(recv._vi, 32), reg0);  // clamp negative integers to zero

            reg1 = _mm256_shuffle_epi8(reg0, _mm256_set1_epi32(201851904));     // in lane(128) shuffle

            *((uint32_t*)(dst + dex_dst)) = _mm256_extract_epi32(reg1, 0);
            *((uint32_t*)(dst + dex_dst + 4)) = _mm256_extract_epi32(reg1, 4);

            dex_src += 8;
            dex_dst += 8;
        }
    }
}



_THREAD_FUNCTION_ void
decx::type_cast::CPUK::_v256_cvti32_ui8_truncate2D(const int32_t* __restrict      src, 
                                                   uint8_t* __restrict              dst, 
                                                   const uint2                  proc_dims,
                                                   const uint32_t              Wsrc,
                                                   const uint32_t              Wdst)
{
    decx::utils::simd::xmm256_reg recv;
    __m256i reg;

    uint64_t dex_src = 0, dex_dst = 0;

    for (int i = 0; i < proc_dims.y; ++i) {
        dex_src = i * Wsrc;
        dex_dst = i * Wdst;
        for (int j = 0; j < proc_dims.x; ++j) {
            recv._vf = _mm256_load_ps((float*)src + dex_src);

            reg = _mm256_and_si256(recv._vi, _mm256_set1_epi32(0x000000ff));
            reg = _mm256_shuffle_epi8(reg, _mm256_set1_epi32(201851904));

            *((uint32_t*)(dst + dex_dst)) = _mm256_extract_epi32(reg, 0);
            *((uint32_t*)(dst + dex_dst + 4)) = _mm256_extract_epi32(reg, 4);

            dex_src += 8;
            dex_dst += 8;
        }
    }
}



_THREAD_FUNCTION_ void decx::type_cast::CPUK::
_v256_cvti32_ui8_truncate_clamp_zero2D(const int32_t* __restrict      src, 
                                       uint8_t* __restrict          dst, 
                                       const uint2                  proc_dims,
                                       const uint                   Wsrc,
                                       const uint                   Wdst)
{
    decx::utils::simd::xmm256_reg recv;
    __m256i reg;

    size_t dex_src = 0, dex_dst = 0;

    for (int i = 0; i < proc_dims.y; ++i) {
        dex_src = i * Wsrc;
        dex_dst = i * Wdst;
        for (int j = 0; j < proc_dims.x; ++j) {
            recv._vf = _mm256_load_ps((float*)src + dex_src);

            reg = _mm256_and_si256(recv._vi, _mm256_set1_epi32(0x000000ff));
            reg = _mm256_andnot_si256(_mm256_srai_epi32(recv._vi, 32), reg);
            reg = _mm256_shuffle_epi8(reg, _mm256_set1_epi32(201851904));

            *((uint32_t*)(dst + dex_dst)) = _mm256_extract_epi32(reg, 0);
            *((uint32_t*)(dst + dex_dst + 4)) = _mm256_extract_epi32(reg, 4);

            dex_src += 8;
            dex_dst += 8;
        }
    }
}



decx::type_cast::CPUK::_cvt_kernel2D<int32_t, uint8_t>*
decx::type_cast::_cvti32_ui8_selector2D(const int32_t flag)
{
    using namespace decx::type_cast;
    decx::type_cast::CPUK::_cvt_kernel2D<int32_t, uint8_t>* exec_kernrel_ptr = NULL;

    switch (flag)
    {
    case (de::TypeCast_Method::CVT_INT32_UINT8 | 
            de::TypeCast_Method::CVT_UINT8_CLAMP_TO_ZERO | 
            de::TypeCast_Method::CVT_INT32_UINT8_TRUNCATE):        // 0b0011 (3)
        exec_kernrel_ptr = decx::type_cast::CPUK::_v256_cvti32_ui8_truncate_clamp_zero2D;
        break;

    case de::TypeCast_Method::CVT_INT32_UINT8 | de::TypeCast_Method::CVT_INT32_UINT8_TRUNCATE:      // 0b0010 (2)
        exec_kernrel_ptr = decx::type_cast::CPUK::_v256_cvti32_ui8_truncate2D;
        break;

    case (de::TypeCast_Method::CVT_INT32_UINT8 | de::TypeCast_Method::CVT_UINT8_CYCLIC):        // 0b0101 (5) or CVT_INT32_UINT8 | CVT_UINT8_CYCLIC | CVT_CLAMP_TO_ZERO
        exec_kernrel_ptr = decx::type_cast::CPUK::_v256_cvti32_ui8_cyclic2D;
        break;

    case (de::TypeCast_Method::CVT_INT32_UINT8 | de::TypeCast_Method::CVT_UINT8_SATURATED):     // 0b1001 (9) or CVT_INT32_UINT8 | CVT_UINT8_SATURATED | CVT_CLAMP_TO_ZERO
        exec_kernrel_ptr = decx::type_cast::CPUK::_v256_cvti32_ui8_saturated2D;
        break;
    }

    return exec_kernrel_ptr;
}


// void decx::type_cast::_cvtui8_i32_caller2D(const float*         src, 
//                                            float*               dst, 
//                                            const ulong2         proc_dims, 
//                                            const uint           Wsrc, 
//                                            const uint           Wdst)
// {
//     decx::utils::_thread_arrange_1D t1D(decx::cpu::_get_permitted_concurrency());
//     decx::utils::frag_manager f_mgr;
//     decx::utils::frag_manager_gen(&f_mgr, proc_dims.y, t1D.total_thread);

//     const float* loc_src = src;
//     float* loc_dst = dst;

//     const size_t frag_src = Wsrc * f_mgr.frag_len,
//         frag_dst = Wdst * f_mgr.frag_len;

//     for (int i = 0; i < t1D.total_thread - 1; ++i) {
//         t1D._async_thread[i] = decx::cpu::register_task_default( decx::type_cast::CPUK::_v256_cvtui8_i32_2D,
//             loc_src, loc_dst, make_uint2(proc_dims.x, f_mgr.frag_len), Wsrc, Wdst);

//         loc_src += frag_src;
//         loc_dst += frag_dst;
//     }
//     const uint _L = f_mgr.is_left ? f_mgr.frag_left_over : f_mgr.frag_len;
//     t1D._async_thread[t1D.total_thread - 1] = decx::cpu::register_task_default( decx::type_cast::CPUK::_v256_cvtui8_i32_2D,
//         loc_src, loc_dst, make_uint2(proc_dims.x, _L), Wsrc, Wdst);

//     t1D.__sync_all_threads();
// }




// template <bool _print>
// void decx::type_cast::_cvti32_ui8_caller2D(const float*         src,
//                                            int*                 dst, 
//                                            const ulong2         proc_dims, 
//                                            const uint           Wsrc, 
//                                            const uint           Wdst,
//                                            const int            flag,
//                                            de::DH*              handle)
// {
//     using namespace decx::type_cast;
//     decx::utils::_thread_arrange_1D t1D(decx::cpu::_get_permitted_concurrency());
//     decx::utils::frag_manager f_mgr;
//     decx::utils::frag_manager_gen(&f_mgr, proc_dims.y, t1D.total_thread);

//     const float* loc_src = src;
//     int* loc_dst = dst;

//     const size_t frag_src = Wsrc * f_mgr.frag_len,
//         frag_dst = Wdst * f_mgr.frag_len;

//     decx::type_cast::CPUK::_cvt_i32_u8_kernel2D exec_kernrel_ptr = NULL;

//     switch (flag)
//     {
//     case (CVT_INT32_UINT8 | CVT_UINT8_CLAMP_TO_ZERO | CVT_INT32_UINT8_TRUNCATE):        // 0b0011 (3)
//         exec_kernrel_ptr = decx::type_cast::CPUK::_v256_cvti32_ui8_truncate_clamp_zero2D;
//         break;

//     case CVT_INT32_UINT8 | CVT_INT32_UINT8_TRUNCATE:      // 0b0010 (2)
//         exec_kernrel_ptr = decx::type_cast::CPUK::_v256_cvti32_ui8_truncate2D;
//         break;

//     case (CVT_INT32_UINT8 | CVT_UINT8_CYCLIC):        // 0b0101 (5) or CVT_INT32_UINT8 | CVT_UINT8_CYCLIC | CVT_CLAMP_TO_ZERO
//         exec_kernrel_ptr = decx::type_cast::CPUK::_v256_cvti32_ui8_cyclic2D;
//         break;

//     case (CVT_INT32_UINT8 | CVT_UINT8_SATURATED):     // 0b1001 (9) or CVT_INT32_UINT8 | CVT_UINT8_SATURATED | CVT_CLAMP_TO_ZERO
//         exec_kernrel_ptr = decx::type_cast::CPUK::_v256_cvti32_ui8_saturated2D;
//         break;

//     default:
//         decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_INVALID_PARAM,
//             INVALID_PARAM);
//         return;
//         break;
//     }

//     for (int i = 0; i < t1D.total_thread - 1; ++i) {
//         t1D._async_thread[i] = decx::cpu::register_task_default( exec_kernrel_ptr,
//             loc_src, loc_dst, make_uint2(proc_dims.x, f_mgr.frag_len), Wsrc, Wdst);

//         loc_src += frag_src;
//         loc_dst += frag_dst;
//     }
//     const uint _L = f_mgr.is_left ? f_mgr.frag_left_over : f_mgr.frag_len;
//     t1D._async_thread[t1D.total_thread - 1] = decx::cpu::register_task_default( exec_kernrel_ptr,
//         loc_src, loc_dst, make_uint2(proc_dims.x, _L), Wsrc, Wdst);

//     t1D.__sync_all_threads();
// }


// template void decx::type_cast::_cvti32_ui8_caller2D<true>(const float* src, int* dst, const ulong2 proc_dims,
//     const uint Wsrc, const uint Wdst, const int flag, de::DH* handle);



// template void decx::type_cast::_cvti32_ui8_caller2D<false>(const float* src, int* dst, const ulong2 proc_dims,
//     const uint Wsrc, const uint Wdst, const int flag, de::DH* handle);
