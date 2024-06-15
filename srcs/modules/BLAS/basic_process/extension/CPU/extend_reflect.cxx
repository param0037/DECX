/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#include "extend_reflect.h"
#include "extend_reflect_exec_params.h"


void decx::bp::_extend_reflect_b32_1D(const float* src, float* dst, const uint32_t _left, const uint32_t _right,
    const size_t _length_src, const size_t _actual_Lsrc, de::DH* handle)
{
    decx::bp::extend_reflect_exec_params b_rfct;
    decx::bp::e_rfct_exep_gen_b32(&b_rfct, _left, _right, _actual_Lsrc, _length_src / 8);

    decx::PtrInfo<float> buffer;
    if (decx::alloc::_host_virtual_page_malloc(&buffer, decx::bp::e_rfct_exep_get_buffer_len(&b_rfct) * sizeof(float))) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_ALLOCATION,
            ALLOC_FAIL);
        return;
    }

    decx::bp::CPUK::_extend_reflect1D_b32(src, buffer.ptr, dst, &b_rfct, _actual_Lsrc, _length_src / 8);

    decx::alloc::_host_virtual_page_dealloc(&buffer);
}



void decx::bp::_extend_reflect_b64_1D(const double* src, double* dst, const uint32_t _left, const uint32_t _right,
    const size_t _length_src, const size_t _actual_Lsrc, de::DH* handle)
{
    decx::bp::extend_reflect_exec_params b_rfct;
    decx::bp::e_rfct_exep_gen_b64(&b_rfct, _left, _right, _actual_Lsrc, _length_src / 4);

    decx::PtrInfo<double> buffer;
    if (decx::alloc::_host_virtual_page_malloc(&buffer, decx::bp::e_rfct_exep_get_buffer_len(&b_rfct) * sizeof(double))) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_ALLOCATION,
            ALLOC_FAIL);
        return;
    }

    decx::bp::CPUK::_extend_reflect1D_b64(src, buffer.ptr, dst, &b_rfct, _actual_Lsrc, _length_src / 4);

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


void decx::bp::_extend_reflect_b16_1D(const uint16_t* src, uint16_t* dst, const uint32_t _left, const uint32_t _right,
    const size_t _length_src, const size_t _actual_Lsrc, de::DH* handle)
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
    decx::bp::extend_reflect_exec_params b_rfct;
    decx::bp::e_rfct_exep_gen_b32(&b_rfct, _ext.x, _ext.y, _actual_Wsrc, Wsrc / 8);

    decx::PtrInfo<float> buffer;
    if (decx::alloc::_host_virtual_page_malloc(&buffer, decx::bp::e_rfct_exep_get_buffer_len(&b_rfct) * sizeof(float))) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_ALLOCATION, ALLOC_FAIL);
        return;
    }

    decx::bp::CPUK::_extend_H_reflect2D_b32(src, 
                                            buffer.ptr, 
                                            decx::utils::ptr_shift_xy<float, float>(dst, _ext.z, 0, Wdst),
                                            &b_rfct,
                                            Wsrc, Wdst, _actual_Wsrc, make_uint2(Wsrc / 8, Hsrc));

    decx::bp::CPUK::_extend_V_reflect2D_m256(dst, _ext.z, _ext.w, Hsrc, Wdst);

    decx::alloc::_host_virtual_page_dealloc(&buffer);
}


void decx::bp::_extend_LR_reflect_b32_2D(const float* src, float* dst, const uint2 _ext,
    const uint32_t Wsrc, const uint32_t Wdst, const uint32_t _actual_Wsrc, const uint32_t Hsrc, de::DH* handle)
{
    decx::bp::extend_reflect_exec_params b_rfct;
    decx::bp::e_rfct_exep_gen_b32(&b_rfct, _ext.x, _ext.y, _actual_Wsrc, Wsrc / 8);

    decx::PtrInfo<float> buffer;
    if (decx::alloc::_host_virtual_page_malloc(&buffer, decx::bp::e_rfct_exep_get_buffer_len(&b_rfct) * sizeof(float))) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_ALLOCATION,
            ALLOC_FAIL);
    }

    decx::bp::CPUK::_extend_H_reflect2D_b32(src, buffer.ptr, dst,&b_rfct,
                                            Wsrc, Wdst, _actual_Wsrc, make_uint2(Wsrc / 8, Hsrc));

    decx::alloc::_host_virtual_page_dealloc(&buffer);
}


void decx::bp::_extend_TB_reflect_b32_2D(float* src, const uint2 _ext,
    const uint32_t Wsrc, const uint32_t Hsrc)
{
    decx::bp::CPUK::_extend_V_reflect2D_m256(src, _ext.x, _ext.y, Hsrc, Wsrc);
}



void decx::bp::_extend_reflect_b64_2D(const double* src, double* dst, const uint4 _ext,
    const uint32_t Wsrc, const uint32_t Wdst, const uint32_t _actual_Wsrc, const uint32_t Hsrc, de::DH* handle)
{
    decx::bp::extend_reflect_exec_params b_rfct;
    decx::bp::e_rfct_exep_gen_b64(&b_rfct, _ext.x, _ext.y, _actual_Wsrc, Wsrc / 4);

    decx::PtrInfo<double> buffer;
    if (decx::alloc::_host_virtual_page_malloc(&buffer, decx::bp::e_rfct_exep_get_buffer_len(&b_rfct) * sizeof(double))) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_ALLOCATION,
            ALLOC_FAIL);
    }

    decx::bp::CPUK::_extend_H_reflect2D_b64(src, buffer.ptr,
        decx::utils::ptr_shift_xy<double, double>(dst, _ext.z, 0, Wdst),
        &b_rfct, Wsrc, Wdst, _actual_Wsrc, make_uint2(Wsrc / 4, Hsrc));

    decx::bp::CPUK::_extend_V_reflect2D_m256((float*)dst, _ext.z, _ext.w, Hsrc, Wdst * 2);

    decx::alloc::_host_virtual_page_dealloc(&buffer);
}



void decx::bp::_extend_reflect_b8_2D(const uint8_t* src, uint8_t* dst, const uint4 _ext,
    const uint32_t Wsrc, const uint32_t Wdst, const uint32_t _actual_Wsrc, const uint32_t Hsrc, de::DH* handle)
{
    decx::bp::extend_reflect_exec_params b_rfct;
    decx::bp::e_rfct_exep_gen_b8(&b_rfct, _ext.x, _ext.y, _actual_Wsrc, Wsrc / 16);

    decx::PtrInfo<uint8_t> buffer;
    if (decx::alloc::_host_virtual_page_malloc(&buffer, decx::bp::e_rfct_exep_get_buffer_len(&b_rfct) * sizeof(uint8_t))) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_ALLOCATION,
            ALLOC_FAIL);
        return;
    }

    decx::bp::CPUK::_extend_H_reflect2D_b8(src, buffer.ptr, 
        decx::utils::ptr_shift_xy<uint8_t, uint8_t>(dst, _ext.z, 0, Wdst),
        &b_rfct, Wsrc, Wdst, _actual_Wsrc, 
        make_uint2(decx::utils::ceil<uint32_t>(_actual_Wsrc, 16), Hsrc));

    decx::bp::CPUK::_extend_V_reflect2D_m256((float*)dst, _ext.z, _ext.w, Hsrc, Wdst / 4);

    decx::alloc::_host_virtual_page_dealloc(&buffer);
}


void decx::bp::_extend_LR_reflect_b8_2D(const uint8_t* src, uint8_t* dst, const uint2 _ext,
    const uint32_t Wsrc, const uint32_t Wdst, const uint32_t _actual_Wsrc, const uint32_t Hsrc, de::DH* handle)
{
    decx::bp::extend_reflect_exec_params b_rfct;
    decx::bp::e_rfct_exep_gen_b8(&b_rfct, _ext.x, _ext.y, _actual_Wsrc, Wsrc / 16);

    decx::PtrInfo<uint8_t> buffer;
    if (decx::alloc::_host_virtual_page_malloc(&buffer, decx::bp::e_rfct_exep_get_buffer_len(&b_rfct) * sizeof(uint8_t))) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_ALLOCATION,
            ALLOC_FAIL);
        return;
    }

    decx::bp::CPUK::_extend_H_reflect2D_b8(src, buffer.ptr, dst,
        &b_rfct, Wsrc, Wdst, _actual_Wsrc, 
        make_uint2(decx::utils::ceil<uint32_t>(_actual_Wsrc, 16), Hsrc));

    decx::alloc::_host_virtual_page_dealloc(&buffer);
}


void decx::bp::_extend_reflect_b16_2D(const uint16_t* src, uint16_t* dst, const uint4 _ext,
    const uint32_t Wsrc, const uint32_t Wdst, const uint32_t _actual_Wsrc, const uint32_t Hsrc, de::DH* handle)
{
    decx::bp::extend_reflect_exec_params b_rfct;
    decx::bp::e_rfct_exep_gen_b16(&b_rfct, _ext.x, _ext.y, _actual_Wsrc, Wsrc / 8);

    decx::PtrInfo<uint16_t> buffer;
    if (decx::alloc::_host_virtual_page_malloc(&buffer, decx::bp::e_rfct_exep_get_buffer_len(&b_rfct) * sizeof(uint16_t))) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_ALLOCATION,
            ALLOC_FAIL);
        return;
    }

    decx::bp::CPUK::_extend_H_reflect2D_b16(src, buffer.ptr,
        decx::utils::ptr_shift_xy<uint16_t, uint16_t>(dst, _ext.z, 0, Wdst),
        &b_rfct, Wsrc, Wdst, _actual_Wsrc,
        make_uint2(decx::utils::ceil<uint32_t>(_actual_Wsrc, 8), Hsrc));

    decx::bp::CPUK::_extend_V_reflect2D_m256((float*)dst, _ext.z, _ext.w, Hsrc, Wdst / 2);

    decx::alloc::_host_virtual_page_dealloc(&buffer);
}
