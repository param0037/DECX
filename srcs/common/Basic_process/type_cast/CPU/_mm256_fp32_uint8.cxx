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


// #include "_mm256_fp32_uint8.h"
#include "typecast_exec_x86_64.h"


_THREAD_FUNCTION_ void
decx::type_cast::CPUK::_v256_cvtui8_f32_1D(const uint8_t* __restrict  src, 
                                           float* __restrict        dst, 
                                           const size_t             proc_len)
{
    decx::utils::simd::xmm128_reg recv;
    decx::utils::simd::xmm256_reg store;

    for (int i = 0; i < proc_len; ++i) {
        recv._vf = _mm_loadu_ps((float*)(src + i * 8));
        store._vi = _mm256_cvtepu8_epi32(recv._vi);
        store._vf = _mm256_cvtepi32_ps(store._vi);

        _mm256_store_ps(dst + i * 8, store._vf);
    }
}



_THREAD_FUNCTION_ void
decx::type_cast::CPUK::_v256_cvtf32_ui8_cyclic1D(const float* __restrict    src, 
                                                 uint8_t* __restrict              dst, 
                                                 const size_t                 proc_len)
{
    decx::utils::simd::xmm256_reg recv, _crit, store;

    for (int32_t i = 0; i < proc_len; ++i) {
        recv._vf = _mm256_load_ps(src + i * 8);

        _crit._vf = _mm256_cmp_ps(recv._vf, _mm256_set1_ps(0), _CMP_GT_OS);
        store._vi = _mm256_and_si256(recv._vi, _crit._vi);
        _crit._vf = _mm256_cmp_ps(recv._vf, _mm256_set1_ps(255), _CMP_LT_OS);     // cyclic_cast
        store._vi = _mm256_and_si256(store._vi, _crit._vi);

        store._vi = _mm256_cvtps_epi32(store._vf);
        store._vi = _mm256_shuffle_epi8(_mm256_and_si256(store._vi, _mm256_set1_epi32(0x000000ff)), _mm256_set1_epi32(201851904));
        
        *((uint32_t*)(dst + i * 8)) = store._arrui[0];
        *((uint32_t*)(dst + i * 8 + 4)) = store._arrui[4];
    }
}



_THREAD_FUNCTION_ void
decx::type_cast::CPUK::_v256_cvtf32_ui8_saturated1D(const float* __restrict      src, 
                                                    uint8_t* __restrict              dst, 
                                                    const uint64_t                 proc_len)
{
    decx::utils::simd::xmm256_reg recv, _crit, store;

    for (int i = 0; i < proc_len; ++i) {
        recv._vf = _mm256_load_ps(src + i * 8);

        _crit._vf = _mm256_cmp_ps(recv._vf, _mm256_set1_ps(0), _CMP_GT_OS);
        store._vi = _mm256_and_si256(recv._vi, _crit._vi);

        store._vi = _mm256_cvtps_epi32(store._vf);
        _crit._vf = _mm256_cmp_ps(recv._vf, _mm256_set1_ps(255), _CMP_GT_OS);     // saturated_cast
        store._vi = _mm256_or_si256(store._vi, _crit._vi);       // saturated_cast
        store._vi = _mm256_shuffle_epi8(_mm256_and_si256(store._vi, _mm256_set1_epi32(0x000000ff)), _mm256_set1_epi32(201851904));

        *((uint32_t*)(dst + i * 8)) = store._arrui[0];
        *((uint32_t*)(dst + i * 8 + 4)) = store._arrui[4];
    }
}



decx::type_cast::CPUK::_cvt_kernel1D<float, uint8_t>* 
decx::type_cast::_cvtf32_ui8_selector1D(const int32_t flag)
{
    using namespace decx::type_cast;
    decx::type_cast::CPUK::_cvt_kernel1D<float, uint8_t>* exec_kernrel_ptr = NULL;

    switch (flag)
    {
    case (de::TypeCast_Method::CVT_FP32_UINT8 | de::TypeCast_Method::CVT_UINT8_CYCLIC):        // 0b10101 (21) or CVT_FP32_UINT8 | CVT_UINT8_CYCLIC | CVT_UINT8_CLAMP_TO_ZERO
        exec_kernrel_ptr = decx::type_cast::CPUK::_v256_cvtf32_ui8_cyclic1D;
        break;

    case (de::TypeCast_Method::CVT_FP32_UINT8 | de::TypeCast_Method::CVT_UINT8_SATURATED):     // 0b11001 (25)  or CVT_FP32_UINT8 | CVT_UINT8_SATURATED | CVT_UINT8_CLAMP_TO_ZERO
        exec_kernrel_ptr = decx::type_cast::CPUK::_v256_cvtf32_ui8_saturated1D;
        break;
    }

    return exec_kernrel_ptr;
}



// ------------------------------------------------------ 2D ------------------------------------------------------------



_THREAD_FUNCTION_ void
decx::type_cast::CPUK::_v256_cvtui8_f32_2D(const uint8_t* __restrict  src, 
                                           float* __restrict        dst, 
                                           const uint2              proc_dims, 
                                           const uint               Wsrc, 
                                           const uint               Wdst)
{
    decx::utils::simd::xmm128_reg recv;
    decx::utils::simd::xmm256_reg store;

    uint64_t dex_src = 0, dex_dst = 0;

    for (int32_t i = 0; i < proc_dims.y; ++i) {
        dex_src = i * Wsrc;
        dex_dst = i * Wdst;
        for (int32_t j = 0; j < proc_dims.x; ++j) {
            recv._vf = _mm_loadu_ps((float*)(src + dex_src));
            store._vi = _mm256_cvtepu8_epi32(recv._vi);
            store._vf = _mm256_cvtepi32_ps(store._vi);

            _mm256_store_ps(dst + dex_dst, store._vf);

            dex_src += 8;
            dex_dst += 8;
        }
    }
}



_THREAD_FUNCTION_ void
decx::type_cast::CPUK::_v256_cvtf32_ui8_cyclic2D(const float* __restrict    src, 
                                                 uint8_t* __restrict          dst, 
                                                 const uint2                  proc_dims,
                                                 const uint                   Wsrc,
                                                 const uint                   Wdst)
{
    decx::utils::simd::xmm256_reg recv, _crit, store;

    uint64_t dex_src = 0, dex_dst = 0;

    for (int i = 0; i < proc_dims.y; ++i) {
        dex_src = i * Wsrc;
        dex_dst = i * Wdst;
        for (int j = 0; j < proc_dims.x; ++j) {
            recv._vf = _mm256_load_ps(src + dex_src);

            _crit._vf = _mm256_cmp_ps(recv._vf, _mm256_set1_ps(0), _CMP_GT_OS);
            store._vi = _mm256_and_si256(recv._vi, _crit._vi);
            _crit._vf = _mm256_cmp_ps(recv._vf, _mm256_set1_ps(255), _CMP_LT_OS);     // cyclic_cast
            store._vi = _mm256_and_si256(store._vi, _crit._vi);

            store._vi = _mm256_cvtps_epi32(store._vf);
            store._vi = _mm256_shuffle_epi8(_mm256_and_si256(store._vi, _mm256_set1_epi32(0x000000ff)), _mm256_set1_epi32(201851904));

            *((uint32_t*)(dst + dex_dst)) = store._arrui[0];
            *((uint32_t*)(dst + dex_dst + 4)) = store._arrui[4];

            dex_src += 8;
            dex_dst += 8;
        }
    }
}


_THREAD_FUNCTION_ void
decx::type_cast::CPUK::_v256_cvtf32_ui8_saturated2D(const float* __restrict      src, 
                                                    uint8_t* __restrict              dst, 
                                                    const uint2                  proc_dims,
                                                    const uint                   Wsrc,
                                                    const uint                   Wdst)
{
    decx::utils::simd::xmm256_reg recv, _crit, store;

    size_t dex_src = 0, dex_dst = 0;

    for (int i = 0; i < proc_dims.y; ++i) {
        dex_src = i * Wsrc;
        dex_dst = i * Wdst;
        for (int j = 0; j < proc_dims.x; ++j) {
            recv._vf = _mm256_load_ps(src + dex_src);

            _crit._vf = _mm256_cmp_ps(recv._vf, _mm256_set1_ps(0), _CMP_GT_OS);
            store._vi = _mm256_and_si256(recv._vi, _crit._vi);

            store._vi = _mm256_cvtps_epi32(store._vf);
            _crit._vf = _mm256_cmp_ps(recv._vf, _mm256_set1_ps(255), _CMP_GT_OS);     // saturated_cast
            store._vi = _mm256_or_si256(store._vi, _crit._vi);       // saturated_cast
            store._vi = _mm256_shuffle_epi8(_mm256_and_si256(store._vi, _mm256_set1_epi32(0x000000ff)), _mm256_set1_epi32(201851904));

            *((uint32_t*)(dst + dex_dst)) = store._arrui[0];
            *((uint32_t*)(dst + dex_dst + 4)) = store._arrui[4];

            dex_src += 8;
            dex_dst += 8;
        }
    }
}



// void decx::type_cast::_cvtui8_f32_caller2D(const float*         src, 
//                                            float*               dst, 
//                                            const ulong2         proc_dims, 
//                                            const uint           Wsrc, 
//                                            const uint           Wdst)
// {
//     if (proc_dims.y < decx::cpu::_get_permitted_concurrency()) {
//         decx::type_cast::CPUK::_v256_cvtui8_f32_2D(src, dst, make_uint2(proc_dims.x, proc_dims.y), Wsrc, Wdst);
//     }
//     else {
//         decx::utils::_thread_arrange_1D t1D(decx::cpu::_get_permitted_concurrency());
//         decx::utils::frag_manager f_mgr;
//         decx::utils::frag_manager_gen(&f_mgr, proc_dims.y, t1D.total_thread);

//         const float* loc_src = src;
//         float* loc_dst = dst;

//         const size_t frag_src = Wsrc * f_mgr.frag_len,
//             frag_dst = Wdst * f_mgr.frag_len;

//         for (int i = 0; i < t1D.total_thread - 1; ++i) {
//             t1D._async_thread[i] = decx::cpu::register_task_default( decx::type_cast::CPUK::_v256_cvtui8_f32_2D,
//                 loc_src, loc_dst, make_uint2(proc_dims.x, f_mgr.frag_len), Wsrc, Wdst);

//             loc_src += frag_src;
//             loc_dst += frag_dst;
//         }
//         const uint _L = f_mgr.is_left ? f_mgr.frag_left_over : f_mgr.frag_len;
//         t1D._async_thread[t1D.total_thread - 1] = decx::cpu::register_task_default( decx::type_cast::CPUK::_v256_cvtui8_f32_2D,
//             loc_src, loc_dst, make_uint2(proc_dims.x, _L), Wsrc, Wdst);

//         t1D.__sync_all_threads();
//     }
// }


// template <bool _print>
// void decx::type_cast::_cvtf32_ui8_caller2D(const float*         src,
//                                            int*                 dst, 
//                                            const ulong2         proc_dims, 
//                                            const uint           Wsrc, 
//                                            const uint           Wdst,
//                                            const int            flag,
//                                            de::DH*              handle)
// {
//     using namespace decx::type_cast;
//     decx::type_cast::CPUK::_cvt_f32_u8_kernel2D exec_kernrel_ptr = NULL;

//     switch (flag)
//     {
//     case (CVT_FP32_UINT8 | CVT_UINT8_CYCLIC):        // 0b10101 (21) or CVT_FP32_UINT8 | CVT_UINT8_CYCLIC | CVT_UINT8_CLAMP_TO_ZERO
//         exec_kernrel_ptr = decx::type_cast::CPUK::_v256_cvtf32_ui8_cyclic2D;
//         break;

//     case (CVT_FP32_UINT8 | CVT_UINT8_SATURATED):     // 0b11001 (25)  or CVT_FP32_UINT8 | CVT_UINT8_SATURATED | CVT_UINT8_CLAMP_TO_ZERO
//         exec_kernrel_ptr = decx::type_cast::CPUK::_v256_cvtf32_ui8_saturated2D;
//         break;

//     default:
//         decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_INVALID_PARAM,
//             INVALID_PARAM);
//         return;
//         break;
//     }

//     if (proc_dims.y < decx::cpu::_get_permitted_concurrency()) {
//         (*exec_kernrel_ptr)(src, dst, make_uint2(proc_dims.x, proc_dims.y), Wsrc, Wdst);
//     }
//     else {
//         decx::utils::_thread_arrange_1D t1D(decx::cpu::_get_permitted_concurrency());
//         decx::utils::frag_manager f_mgr;
//         decx::utils::frag_manager_gen(&f_mgr, proc_dims.y, t1D.total_thread);

//         const float* loc_src = src;
//         int* loc_dst = dst;

//         const size_t frag_src = Wsrc * f_mgr.frag_len,
//             frag_dst = Wdst * f_mgr.frag_len;


//         for (int i = 0; i < t1D.total_thread - 1; ++i) {
//             t1D._async_thread[i] = decx::cpu::register_task_default( exec_kernrel_ptr,
//                 loc_src, loc_dst, make_uint2(proc_dims.x, f_mgr.frag_len), Wsrc, Wdst);

//             loc_src += frag_src;
//             loc_dst += frag_dst;
//         }
//         const uint _L = f_mgr.is_left ? f_mgr.frag_left_over : f_mgr.frag_len;
//         t1D._async_thread[t1D.total_thread - 1] = decx::cpu::register_task_default( exec_kernrel_ptr,
//             loc_src, loc_dst, make_uint2(proc_dims.x, _L), Wsrc, Wdst);

//         t1D.__sync_all_threads();
//     }
// }


// template void decx::type_cast::_cvtf32_ui8_caller2D<true>(const float* src, int* dst, const ulong2 proc_dims,
//     const uint Wsrc, const uint Wdst, const int flag, de::DH* handle);

// template void decx::type_cast::_cvtf32_ui8_caller2D<false>(const float* src, int* dst, const ulong2 proc_dims,
//     const uint Wsrc, const uint Wdst, const int flag, de::DH* handle);