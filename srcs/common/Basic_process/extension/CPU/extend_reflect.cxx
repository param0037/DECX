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

#include "extend_reflect.h"
#include "extend_reflect_exec_params.h"
#include "../../../../modules/core/thread_management/thread_arrange.h"
#include "../../../FMGR/fragment_arrangment.h"


void decx::bp::_extend_reflect_b32_1D(const float* src, float* dst, const uint32_t _left, const uint32_t _right,
    const size_t _length_src, const size_t _actual_Lsrc, de::DH* handle)
{
#if defined(__x86_64__) || defined(__i386__)
    constexpr uint32_t _alignment = 8;
#endif
#if defined(__aarch64__) || defined(__arm__)
    constexpr uint32_t _alignment = 4;
#endif
    decx::bp::extend_reflect_exec_params b_rfct;
    decx::bp::e_rfct_exep_gen_b32(&b_rfct, _left, _right, _actual_Lsrc, _length_src / _alignment);

    decx::PtrInfo<float> buffer;
    if (decx::alloc::_host_virtual_page_malloc(&buffer, decx::bp::e_rfct_exep_get_buffer_len(&b_rfct) * sizeof(float))) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_ALLOCATION,
            ALLOC_FAIL);
        return;
    }

    decx::bp::CPUK::_extend_reflect1D_b32(src, buffer.ptr, dst, &b_rfct, _actual_Lsrc, _length_src / _alignment);

    decx::alloc::_host_virtual_page_dealloc(&buffer);
}



void decx::bp::_extend_reflect_b64_1D(const double* src, double* dst, const uint32_t _left, const uint32_t _right,
    const size_t _length_src, const size_t _actual_Lsrc, de::DH* handle)
{
#if defined(__x86_64__) || defined(__i386__)
    constexpr uint32_t _alignment = 4;
#endif
#if defined(__aarch64__) || defined(__arm__)
    constexpr uint32_t _alignment = 2;
#endif
    decx::bp::extend_reflect_exec_params b_rfct;
    decx::bp::e_rfct_exep_gen_b64(&b_rfct, _left, _right, _actual_Lsrc, _length_src / _alignment);

    decx::PtrInfo<double> buffer;
    if (decx::alloc::_host_virtual_page_malloc(&buffer, decx::bp::e_rfct_exep_get_buffer_len(&b_rfct) * sizeof(double))) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_ALLOCATION,
            ALLOC_FAIL);
        return;
    }

    decx::bp::CPUK::_extend_reflect1D_b64(src, buffer.ptr, dst, &b_rfct, _actual_Lsrc, _length_src / _alignment);

    decx::alloc::_host_virtual_page_dealloc(&buffer);
}



void decx::bp::_extend_reflect_b8_1D(const uint8_t* src, uint8_t* dst, const uint32_t _left, const uint32_t _right,
    const size_t _length_src, const size_t _actual_Lsrc, de::DH* handle)
{
    decx::bp::extend_reflect_exec_params b_rfct;
    decx::bp::e_rfct_exep_gen_b8(&b_rfct, _left, _right, _actual_Lsrc, _length_src / 16);

    decx::PtrInfo<uint8_t> buffer;
    if (decx::alloc::_host_virtual_page_malloc(&buffer, decx::bp::e_rfct_exep_get_buffer_len(&b_rfct) * sizeof(uint8_t))) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_ALLOCATION,
            ALLOC_FAIL);
        return;
    }

    decx::bp::CPUK::_extend_reflect1D_b8(src, buffer.ptr, dst, &b_rfct, _actual_Lsrc, _length_src / 16);

    decx::alloc::_host_virtual_page_dealloc(&buffer);
}


void decx::bp::_extend_reflect_b16_1D(const uint16_t* src,          uint16_t* dst, 
                                      const uint32_t _left,         const uint32_t _right,
                                      const uint64_t _length_src,   const uint64_t _actual_Lsrc, 
                                      de::DH* handle)
{
    decx::bp::extend_reflect_exec_params b_rfct;
    decx::bp::e_rfct_exep_gen_b8(&b_rfct, _left, _right, _actual_Lsrc, _length_src / 8);

    decx::PtrInfo<uint16_t> buffer;
    if (decx::alloc::_host_virtual_page_malloc(&buffer, decx::bp::e_rfct_exep_get_buffer_len(&b_rfct) * sizeof(uint16_t))) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_ALLOCATION,
            ALLOC_FAIL);
        return;
    }

    decx::bp::CPUK::_extend_reflect1D_b16(src, buffer.ptr, dst, &b_rfct, _actual_Lsrc, _length_src / 8);

    decx::alloc::_host_virtual_page_dealloc(&buffer);
}


void decx::bp::_extend_reflect_b32_2D(const float* src, float* dst, const uint4 _ext,
    const uint32_t Wsrc, const uint32_t Wdst, const uint32_t _actual_Wsrc, const uint32_t Hsrc, de::DH* handle)
{
#if defined(__x86_64__) || defined(__i386__)
    constexpr uint32_t _alignment = 8;
#endif
#if defined(__aarch64__) || defined(__arm__)
    constexpr uint32_t _alignment = 4;
#endif
    decx::bp::extend_reflect_exec_params b_rfct;
    decx::bp::e_rfct_exep_gen_b32(&b_rfct, _ext.x, _ext.y, _actual_Wsrc, Wsrc / _alignment);

    const uint32_t _conc = decx::cpu::_get_permitted_concurrency();
    const uint32_t _buffer_frag_len = decx::bp::e_rfct_exep_get_buffer_len(&b_rfct);

    decx::PtrInfo<float> buffer;
    if (decx::alloc::_host_virtual_page_malloc(&buffer, _conc * _buffer_frag_len * sizeof(float))) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_ALLOCATION, ALLOC_FAIL);
        return;
    }

    decx::utils::_thr_1D t1D(_conc);
    decx::utils::frag_manager fmgr_H;
    decx::utils::frag_manager_gen(&fmgr_H, Hsrc, t1D.total_thread);

    const float* loc_src = src;
    float* loc_dst = decx::utils::ptr_shift_xy<float, float>(dst, _ext.z, 0, Wdst);

    for (uint32_t i = 0; i < t1D.total_thread; ++i) {
        const uint2 proc_dims = make_uint2(Wsrc / _alignment, i == t1D.total_thread - 1 ?
                                                     fmgr_H.last_frag_len :
                                                     fmgr_H.frag_len);
        t1D._async_thread[i] = decx::cpu::register_task_default(decx::bp::CPUK::_extend_H_reflect2D_b32,
            loc_src, buffer.ptr + i * _buffer_frag_len, loc_dst, &b_rfct, Wsrc, Wdst, _actual_Wsrc, proc_dims);

        loc_src += fmgr_H.frag_len * Wsrc;
        loc_dst += fmgr_H.frag_len * Wdst;
    }

    t1D.__sync_all_threads();
#if defined(__x86_64__) || defined(__i386__)
    decx::bp::CPUK::_extend_V_reflect2D_m256(dst, _ext.z, _ext.w, Hsrc, Wdst);
#endif
#if defined(__aarch64__) || defined(__arm__)
    decx::bp::CPUK::_extend_V_reflect2D_m128(dst, _ext.z, _ext.w, Hsrc, Wdst);
#endif
    decx::alloc::_host_virtual_page_dealloc(&buffer);
}


#if 0
//
//void decx::bp::_extend_LR_reflect_b32_2D(const float* src, float* dst, const uint2 _ext,
//    const uint32_t Wsrc, const uint32_t Wdst, const uint32_t _actual_Wsrc, const uint32_t Hsrc, de::DH* handle)
//{
//    decx::bp::extend_reflect_exec_params b_rfct;
//    decx::bp::e_rfct_exep_gen_b32(&b_rfct, _ext.x, _ext.y, _actual_Wsrc, Wsrc / 8);
//
//    decx::PtrInfo<float> buffer;
//    if (decx::alloc::_host_virtual_page_malloc(&buffer, decx::bp::e_rfct_exep_get_buffer_len(&b_rfct) * sizeof(float))) {
//        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_ALLOCATION,
//            ALLOC_FAIL);
//    }
//
//    decx::bp::CPUK::_extend_H_reflect2D_b32(src, buffer.ptr, dst,&b_rfct,
//                                            Wsrc, Wdst, _actual_Wsrc, make_uint2(Wsrc / 8, Hsrc));
//
//    decx::alloc::_host_virtual_page_dealloc(&buffer);
//}

//
//void decx::bp::_extend_TB_reflect_b32_2D(float* src, const uint2 _ext,
//    const uint32_t Wsrc, const uint32_t Hsrc)
//{
//    decx::bp::CPUK::_extend_V_reflect2D_m256(src, _ext.x, _ext.y, Hsrc, Wsrc);
//}
#endif


void decx::bp::
_extend_reflect_b64_2D(const double* src,           double* dst, const uint4 _ext,
                       const uint32_t Wsrc,         const uint32_t Wdst, 
                       const uint32_t _actual_Wsrc, const uint32_t Hsrc, 
                       de::DH* handle)
{
#if defined(__x86_64__) || defined(__i386__)
    constexpr uint32_t _alignment = 4;
#endif
#if defined(__aarch64__) || defined(__arm__)
    constexpr uint32_t _alignment = 2;
#endif
    decx::bp::extend_reflect_exec_params b_rfct;
    decx::bp::e_rfct_exep_gen_b64(&b_rfct, _ext.x, _ext.y, _actual_Wsrc, Wsrc / _alignment);

    const uint32_t _conc = decx::cpu::_get_permitted_concurrency();
    const uint32_t _buffer_frag_len = decx::bp::e_rfct_exep_get_buffer_len(&b_rfct);

    decx::PtrInfo<double> buffer;
    if (decx::alloc::_host_virtual_page_malloc(&buffer, _buffer_frag_len * _conc * sizeof(double))) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_ALLOCATION,
            ALLOC_FAIL);
    }

    decx::utils::_thr_1D t1D(_conc);
    decx::utils::frag_manager fmgr_H;
    decx::utils::frag_manager_gen(&fmgr_H, Hsrc, t1D.total_thread);

    const double* loc_src = src;
    double* loc_dst = decx::utils::ptr_shift_xy<double, double>(dst, _ext.z, 0, Wdst);

    for (uint32_t i = 0; i < t1D.total_thread; ++i) {
        const uint2 proc_dims = make_uint2(Wsrc / _alignment, i == t1D.total_thread - 1 ?
                                                     fmgr_H.last_frag_len :
                                                     fmgr_H.frag_len);
        t1D._async_thread[i] = decx::cpu::register_task_default(decx::bp::CPUK::_extend_H_reflect2D_b64,
            loc_src, buffer.ptr + i * _buffer_frag_len, loc_dst, &b_rfct, Wsrc, Wdst, _actual_Wsrc, proc_dims);

        loc_src += fmgr_H.frag_len * Wsrc;
        loc_dst += fmgr_H.frag_len * Wdst;
    }

    t1D.__sync_all_threads();
#if defined(__x86_64__) || defined(__i386__)
    decx::bp::CPUK::_extend_V_reflect2D_m256((float*)dst, _ext.z, _ext.w, Hsrc, Wdst * 2);
#endif
#if defined(__aarch64__) || defined(__arm__)
    decx::bp::CPUK::_extend_V_reflect2D_m128((float*)dst, _ext.z, _ext.w, Hsrc, Wdst * 2);
#endif
    decx::alloc::_host_virtual_page_dealloc(&buffer);
}



void decx::bp::_extend_reflect_b8_2D(const uint8_t* src,    uint8_t* dst, 
                                     const uint4 _ext,      const uint32_t Wsrc, 
                                     const uint32_t Wdst,   const uint32_t _actual_Wsrc, 
                                     const uint32_t Hsrc,   de::DH* handle)
{
    decx::bp::extend_reflect_exec_params b_rfct;
    decx::bp::e_rfct_exep_gen_b8(&b_rfct, _ext.x, _ext.y, _actual_Wsrc, Wsrc / 16);

    const uint32_t _conc = decx::cpu::_get_permitted_concurrency();
    const uint32_t _buffer_frag_len = decx::bp::e_rfct_exep_get_buffer_len(&b_rfct);

    decx::PtrInfo<uint8_t> buffer;
    if (decx::alloc::_host_virtual_page_malloc(&buffer, _conc * _buffer_frag_len * sizeof(uint8_t))) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_ALLOCATION,
            ALLOC_FAIL);
        return;
    }

    decx::utils::_thr_1D t1D(_conc);
    decx::utils::frag_manager fmgr_H;
    decx::utils::frag_manager_gen(&fmgr_H, Hsrc, t1D.total_thread);

    const uint8_t* loc_src = src;
    uint8_t* loc_dst = decx::utils::ptr_shift_xy<uint8_t, uint8_t>(dst, _ext.z, 0, Wdst);

    for (uint32_t i = 0; i < t1D.total_thread; ++i) {
        const uint2 proc_dims = make_uint2(Wsrc / 16, i == t1D.total_thread - 1 ?
                                                     fmgr_H.last_frag_len :
                                                     fmgr_H.frag_len);
        t1D._async_thread[i] = decx::cpu::register_task_default(decx::bp::CPUK::_extend_H_reflect2D_b8,
            loc_src, buffer.ptr + i * _buffer_frag_len, loc_dst, &b_rfct, Wsrc, Wdst, _actual_Wsrc, proc_dims);

        loc_src += fmgr_H.frag_len * Wsrc;
        loc_dst += fmgr_H.frag_len * Wdst;
    }

    t1D.__sync_all_threads();
#if defined(__x86_64__) || defined(__i386__)
    decx::bp::CPUK::_extend_V_reflect2D_m256((float*)dst, _ext.z, _ext.w, Hsrc, Wdst / 4);
#endif
#if defined(__aarch64__) || defined(__arm__)
    decx::bp::CPUK::_extend_V_reflect2D_m128((float*)dst, _ext.z, _ext.w, Hsrc, Wdst / 4);
#endif
    decx::alloc::_host_virtual_page_dealloc(&buffer);
}

#if 0
//
//void decx::bp::_extend_LR_reflect_b8_2D(const uint8_t* src, uint8_t* dst, const uint2 _ext,
//    const uint32_t Wsrc, const uint32_t Wdst, const uint32_t _actual_Wsrc, const uint32_t Hsrc, de::DH* handle)
//{
//    decx::bp::extend_reflect_exec_params b_rfct;
//    decx::bp::e_rfct_exep_gen_b8(&b_rfct, _ext.x, _ext.y, _actual_Wsrc, Wsrc / 16);
//
//    const uint32_t _conc = decx::cpu::_get_permitted_concurrency();
//    const uint32_t _buffer_frag_len = decx::bp::e_rfct_exep_get_buffer_len(&b_rfct);
//
//    decx::PtrInfo<uint8_t> buffer;
//    if (decx::alloc::_host_virtual_page_malloc(&buffer, _conc * _buffer_frag_len * sizeof(uint8_t))) {
//        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_ALLOCATION,
//            ALLOC_FAIL);
//        return;
//    }
//
//    decx::utils::_thr_1D t1D(_conc);
//    decx::utils::frag_manager fmgr_H;
//    decx::utils::frag_manager_gen(&fmgr_H, Hsrc, t1D.total_thread);
//
//    const uint8_t* loc_src = src;
//    uint8_t* loc_dst = decx::utils::ptr_shift_xy<uint8_t, uint8_t>(dst, _ext.z, 0, Wdst);
//
//    for (uint32_t i = 0; i < t1D.total_thread; ++i) {
//        const uint2 proc_dims = make_uint2(Wsrc / 16, i == t1D.total_thread - 1 ?
//                                                     fmgr_H.last_frag_len :
//                                                     fmgr_H.frag_len);
//        t1D._async_thread[i] = decx::cpu::register_task_default(decx::bp::CPUK::_extend_H_reflect2D_b8,
//            loc_src, buffer.ptr + i * _buffer_frag_len, loc_dst, &b_rfct, Wsrc, Wdst, _actual_Wsrc, proc_dims);
//
//        loc_src += fmgr_H.frag_len * Wsrc;
//        loc_dst += fmgr_H.frag_len * Wdst;
//    }
//
//    t1D.__sync_all_threads();
//
//    /*decx::bp::CPUK::_extend_H_reflect2D_b8(src, buffer.ptr, dst,
//        &b_rfct, Wsrc, Wdst, _actual_Wsrc, 
//        make_uint2(decx::utils::ceil<uint32_t>(_actual_Wsrc, 16), Hsrc));*/
//
//    decx::alloc::_host_virtual_page_dealloc(&buffer);
//}
#endif


void decx::bp::_extend_reflect_b16_2D(const uint16_t* src,      uint16_t* dst, 
                                      const uint4 _ext,         const uint32_t Wsrc, 
                                      const uint32_t Wdst,      const uint32_t _actual_Wsrc, 
                                      const uint32_t Hsrc,      de::DH* handle)
{
    decx::bp::extend_reflect_exec_params b_rfct;
    decx::bp::e_rfct_exep_gen_b16(&b_rfct, _ext.x, _ext.y, _actual_Wsrc, Wsrc / 8);

    const uint32_t _conc = decx::cpu::_get_permitted_concurrency();
    const uint32_t _buffer_frag_len = decx::bp::e_rfct_exep_get_buffer_len(&b_rfct);

    decx::PtrInfo<uint16_t> buffer;
    if (decx::alloc::_host_virtual_page_malloc(&buffer, _conc * _buffer_frag_len * sizeof(uint16_t))) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_ALLOCATION,
            ALLOC_FAIL);
        return;
    }

    decx::utils::_thr_1D t1D(_conc);
    decx::utils::frag_manager fmgr_H;
    decx::utils::frag_manager_gen(&fmgr_H, Hsrc, t1D.total_thread);

    const uint16_t* loc_src = src;
    uint16_t* loc_dst = decx::utils::ptr_shift_xy<uint16_t, uint16_t>(dst, _ext.z, 0, Wdst);

    for (uint32_t i = 0; i < t1D.total_thread; ++i) {
        const uint2 proc_dims = make_uint2(Wsrc / 8, i == t1D.total_thread - 1 ?
                                                     fmgr_H.last_frag_len :
                                                     fmgr_H.frag_len);
        t1D._async_thread[i] = decx::cpu::register_task_default(decx::bp::CPUK::_extend_H_reflect2D_b16,
            loc_src, buffer.ptr + i * _buffer_frag_len, loc_dst, &b_rfct, Wsrc, Wdst, _actual_Wsrc, proc_dims);

        loc_src += fmgr_H.frag_len * Wsrc;
        loc_dst += fmgr_H.frag_len * Wdst;
    }

    t1D.__sync_all_threads();
#if defined(__x86_64__) || defined(__i386__)
    decx::bp::CPUK::_extend_V_reflect2D_m256((float*)dst, _ext.z, _ext.w, Hsrc, Wdst / 2);
#endif
#if defined(__aarch64__) || defined(__arm__)
    decx::bp::CPUK::_extend_V_reflect2D_m128((float*)dst, _ext.z, _ext.w, Hsrc, Wdst / 2);
#endif
    decx::alloc::_host_virtual_page_dealloc(&buffer);
}
