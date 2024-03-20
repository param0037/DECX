/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _GAUSSIAN_FILTER_H_
#define _GAUSSIAN_FILTER_H_


#include "Gaussian_filter_exec.h"
#include "../../../classes/Matrix.h"
#include "Gaussian_kernel.h"
#include "../../../BLAS/basic_process/extension/CPU/extend.h"
#include "../../../BLAS/basic_process/extension/CPU/extend_reflect.h"
#include "../../../BLAS/basic_process/rect_and_cube/CPU/rect_copy2D_exec.h"
#include "../../../DSP/convolution/conv_utils.h"


namespace decx
{
    namespace vis
    {
        template <bool _print>
        static void _gaussian_uint8_NB(decx::_Matrix* src, decx::vis::gaussian_kernel1D *kernel_H, decx::vis::gaussian_kernel1D *kernel_V, decx::_Matrix* dst, 
            de::DH* handle);



        template <bool _print>
        static void _gaussian_uint8_BC_zero(decx::_Matrix* src, decx::vis::gaussian_kernel1D *kernel_H, decx::vis::gaussian_kernel1D *kernel_V, decx::_Matrix* dst, 
            de::DH* handle);


        template <bool _print>
        static void _gaussian_uint8_BC_reflect(decx::_Matrix* src, decx::vis::gaussian_kernel1D *kernel_H, decx::vis::gaussian_kernel1D *kernel_V, decx::_Matrix* dst, 
            de::DH* handle);
    }
}



template <bool _print>
static void decx::vis::_gaussian_uint8_NB(decx::_Matrix* src, decx::vis::gaussian_kernel1D* kernel_H, decx::vis::gaussian_kernel1D* kernel_V, decx::_Matrix* dst,
    de::DH* handle)
{
    const uint2 Hconv_res_dims = make_uint2(decx::utils::ceil<uint32_t>(dst->Width(), 16) * 16, 
        src->Height() + kernel_V->_ker_length - 1);

    decx::PtrInfo<float> _Hconv_res;
    if (decx::alloc::_host_virtual_page_malloc(&_Hconv_res, Hconv_res_dims.x * Hconv_res_dims.y * sizeof(float))) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_ALLOCATION,
            ALLOC_FAIL);
        return;
    }

    const uint conc_thr = (uint)decx::cpu::_get_permitted_concurrency();
    decx::utils::_thr_1D t1D(conc_thr);

    const uint _loop = (kernel_H->_ker_length - 1) / 16;
    ushort reg_WL = (ushort)(kernel_H->_ker_length - _loop * 16);

    decx::vis::_gaussian_H_uint8_caller((double*)src->Mat.ptr, 
                                        kernel_H->_kernel_data.ptr, 
                                        _Hconv_res.ptr, 
                                        make_uint2(Hconv_res_dims.x / 16, src->Height()), 
                                        kernel_H->_ker_length,
                                        src->Pitch() / 8,
                                        Hconv_res_dims.x, 
                                        reg_WL, &t1D, _loop);

    decx::vis::_gaussian_V_uint8_caller(_Hconv_res.ptr, 
                                        kernel_V->_kernel_data.ptr, 
                                        (double*)dst->Mat.ptr,
                                        make_uint2(Hconv_res_dims.x / 16, dst->Height()), 
                                        kernel_V->_ker_length, 
                                        Hconv_res_dims.x, 
                                        dst->Pitch() / 8, &t1D);

    decx::alloc::_host_virtual_page_dealloc(&_Hconv_res);
}




template <bool _print>
static void decx::vis::_gaussian_uint8_BC_zero(decx::_Matrix* src, decx::vis::gaussian_kernel1D* kernel_H, decx::vis::gaussian_kernel1D* kernel_V, decx::_Matrix* dst,
    de::DH* handle)
{
    const uint2 tmp_src_dims = make_uint2(decx::utils::ceil<uint>(src->Width() + kernel_H->_ker_length - 1, 16) * 16,
        src->Height());
    const uint2 Hconv_res_dims = make_uint2(decx::utils::ceil<uint32_t>(dst->Width(), 16) * 16, 
        src->Height() + kernel_V->_ker_length - 1);

    decx::PtrInfo<uint8_t> tmp_src;
    decx::PtrInfo<float> _Hconv_res;
    if (decx::alloc::_host_virtual_page_malloc(&tmp_src, tmp_src_dims.x * tmp_src_dims.y * sizeof(uint8_t))) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_ALLOCATION,
            ALLOC_FAIL);
        return;
    }
    if (decx::alloc::_host_virtual_page_malloc(&_Hconv_res, Hconv_res_dims.x * Hconv_res_dims.y * sizeof(float))) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_ALLOCATION,
            ALLOC_FAIL);
        return;
    }

    uint8_t* start_place_src = DECX_PTR_SHF_XY<uint8_t, uint8_t>(tmp_src.ptr, 0, kernel_H->_ker_length / 2, tmp_src_dims.x);
    decx::_cpy2D_anybit_caller<uint8_t>((uint8_t*)src->Mat.ptr, 
                                        start_place_src, 
                                        src->Pitch(), 
                                        tmp_src_dims.x, 
                                        make_uint2(src->Width(), src->Height()));

    const uint conc_thr = (uint)decx::cpu::_get_permitted_concurrency();
    decx::utils::_thr_1D t1D(conc_thr);

    const uint _loop = (kernel_H->_ker_length - 1) / 16;
    ushort reg_WL = (ushort)(kernel_H->_ker_length - _loop * 16);

    float* _mid_ptr = DECX_PTR_SHF_XY<float, float>(_Hconv_res.ptr, kernel_V->_ker_length / 2, 0, Hconv_res_dims.x);
    decx::vis::_gaussian_H_uint8_caller((double*)tmp_src.ptr, 
                                        kernel_H->_kernel_data.ptr, 
                                        _mid_ptr, 
                                        make_uint2(Hconv_res_dims.x / 16, dst->Height()), 
                                        kernel_H->_ker_length,
                                        tmp_src_dims.x / 8,
                                        Hconv_res_dims.x, 
                                        reg_WL, &t1D, _loop);

    decx::vis::_gaussian_V_uint8_caller(_Hconv_res.ptr, 
                                        kernel_V->_kernel_data.ptr, 
                                        (double*)dst->Mat.ptr,
                                        make_uint2(Hconv_res_dims.x / 16, dst->Height()), 
                                        kernel_V->_ker_length, 
                                        Hconv_res_dims.x, 
                                        dst->Pitch() / 8, &t1D);

    decx::alloc::_host_virtual_page_dealloc(&tmp_src);
    decx::alloc::_host_virtual_page_dealloc(&_Hconv_res);
}




template <bool _print>
static void decx::vis::_gaussian_uint8_BC_reflect(decx::_Matrix* src, decx::vis::gaussian_kernel1D* kernel_H, decx::vis::gaussian_kernel1D* kernel_V, decx::_Matrix* dst,
    de::DH* handle)
{
    const uint2 tmp_src_dims = make_uint2(decx::utils::ceil<uint>(src->Width() + kernel_H->_ker_length - 1, 16) * 16,
        src->Height());
    const uint2 Hconv_res_dims = make_uint2(decx::utils::ceil<uint32_t>(dst->Width(), 16) * 16, 
        src->Height() + kernel_V->_ker_length - 1);

    decx::PtrInfo<uint8_t> tmp_src;
    decx::PtrInfo<float> _Hconv_res;
    if (decx::alloc::_host_virtual_page_malloc(&tmp_src, tmp_src_dims.x * tmp_src_dims.y * sizeof(uint8_t))) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_ALLOCATION,
            ALLOC_FAIL);
        return;
    }
    if (decx::alloc::_host_virtual_page_malloc(&_Hconv_res, Hconv_res_dims.x * Hconv_res_dims.y * sizeof(float))) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_ALLOCATION,
            ALLOC_FAIL);
        return;
    }

    uint8_t* start_place_src = DECX_PTR_SHF_XY<uint8_t, uint8_t>(tmp_src.ptr, 0, kernel_H->_ker_length / 2, tmp_src_dims.x);
    decx::bp::_extend_LR_reflect_b8_2D<_print>((uint8_t*)src->Mat.ptr, tmp_src.ptr, make_uint2(kernel_H->_ker_length / 2, kernel_H->_ker_length / 2),
        src->Pitch(), tmp_src_dims.x, src->Width(), src->Height(), handle);

    const uint conc_thr = (uint)decx::cpu::_get_permitted_concurrency();
    decx::utils::_thr_1D t1D(conc_thr);

    const uint _loop = (kernel_H->_ker_length - 1) / 16;
    ushort reg_WL = (ushort)(kernel_H->_ker_length - _loop * 16);

    float* _mid_ptr = DECX_PTR_SHF_XY<float, float>(_Hconv_res.ptr, kernel_V->_ker_length / 2, 0, Hconv_res_dims.x);
    decx::vis::_gaussian_H_uint8_caller((double*)tmp_src.ptr, 
                                        kernel_H->_kernel_data.ptr, 
                                        _mid_ptr, 
                                        make_uint2(Hconv_res_dims.x / 16, dst->Height()), 
                                        kernel_H->_ker_length,
                                        tmp_src_dims.x / 8,
                                        Hconv_res_dims.x, 
                                        reg_WL, &t1D, _loop);

    decx::bp::_extend_TB_reflect_b32_2D(_Hconv_res.ptr, make_uint2(kernel_V->_ker_length / 2, kernel_V->_ker_length / 2), Hconv_res_dims.x, dst->Height());

    decx::vis::_gaussian_V_uint8_caller(_Hconv_res.ptr, 
                                        kernel_V->_kernel_data.ptr, 
                                        (double*)dst->Mat.ptr,
                                        make_uint2(Hconv_res_dims.x / 16, dst->Height()), 
                                        kernel_V->_ker_length, 
                                        Hconv_res_dims.x, 
                                        dst->Pitch() / 8, &t1D);

    decx::alloc::_host_virtual_page_dealloc(&tmp_src);
    decx::alloc::_host_virtual_page_dealloc(&_Hconv_res);
}



namespace decx
{
    namespace vis {
        template <bool _print>
        static void _Gaussian_filter_uint8_organisor(decx::_Matrix* src, decx::_Matrix* dst, const uint2 neighbor_dims, const float2 sigmaXY,
            const uint2 centerXY, de::DH* handle, const bool _is_central, const int border_type);



        template <bool _print>
        static void _Gaussian_filter_uchar4_organisor(decx::_Matrix* src, decx::_Matrix* dst, const uint2 neighbor_dims, const float2 sigmaXY,
            const uint2 centerXY, de::DH* handle, const bool _is_central, const int border_type);
    }
}


template <bool _print>
static void 
decx::vis::_Gaussian_filter_uint8_organisor(decx::_Matrix* src, decx::_Matrix* dst, const uint2 neighbor_dims, const float2 sigmaXY,
    const uint2 centerXY, de::DH* handle, const bool _is_central, const int border_type)
{
    decx::vis::gaussian_kernel1D kernel_H(neighbor_dims.x, sigmaXY.x, _is_central, centerXY.x);
    decx::vis::gaussian_kernel1D kernel_V(neighbor_dims.y, sigmaXY.y, _is_central, centerXY.y);
    kernel_H.generate();
    kernel_V.generate();

    switch (border_type)
    {
    case de::extend_label::_EXTEND_NONE_:
        dst->re_construct(src->Type(), src->Width() - neighbor_dims.x + 1, src->Height() - neighbor_dims.y + 1);
        decx::vis::_gaussian_uint8_NB<_print>(src, &kernel_H, &kernel_V, dst, handle);
        break;

    case de::extend_label::_EXTEND_CONSTANT_:
        dst->re_construct(src->Type(), src->Width(), src->Height());
        decx::vis::_gaussian_uint8_BC_zero<_print>(src, &kernel_H, &kernel_V, dst, handle);
        break;

    case de::extend_label::_EXTEND_REFLECT_:
        dst->re_construct(src->Type(), src->Width(), src->Height());
        decx::vis::_gaussian_uint8_BC_reflect<_print>(src, &kernel_H, &kernel_V, dst, handle);
        break;
    default:
        break;
    }
}




namespace decx
{
    namespace vis {
        template <bool _print>
        static void _gaussian_uchar4_NB(decx::_Matrix* src, decx::vis::gaussian_kernel1D* kernel_H, decx::vis::gaussian_kernel1D* kernel_V, decx::_Matrix* dst,
            de::DH* handle);



        template <bool _print>
        static void _gaussian_uchar4_BC_zero(decx::_Matrix* src, decx::vis::gaussian_kernel1D* kernel_H, decx::vis::gaussian_kernel1D* kernel_V, decx::_Matrix* dst,
            de::DH* handle);


        template <bool _print>
        static void _gaussian_uchar4_BC_reflect(decx::_Matrix* src, decx::vis::gaussian_kernel1D* kernel_H, decx::vis::gaussian_kernel1D* kernel_V, decx::_Matrix* dst,
            de::DH* handle);
    }
}




template <bool _print>
static void decx::vis::_gaussian_uchar4_NB(decx::_Matrix* src, decx::vis::gaussian_kernel1D* kernel_H, decx::vis::gaussian_kernel1D* kernel_V, decx::_Matrix* dst,
    de::DH* handle)
{
    const uint2 Hconv_res_dims = make_uint2(decx::utils::ceil<uint32_t>(dst->Width(), 4) * 4 * 4, 
        src->Height() + kernel_V->_ker_length - 1);

    decx::PtrInfo<float> _Hconv_res;
    if (decx::alloc::_host_virtual_page_malloc(&_Hconv_res, Hconv_res_dims.x * Hconv_res_dims.y * sizeof(float))) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_ALLOCATION,
            ALLOC_FAIL);
        return;
    }

    const uint conc_thr = (uint)decx::cpu::_get_permitted_concurrency();
    decx::utils::_thr_1D t1D(conc_thr);

    const uint _loop = (kernel_H->_ker_length - 1) / 4;
    ushort reg_WL = (ushort)(kernel_H->_ker_length - _loop * 4);

    decx::vis::_gaussian_H_uchar4_caller((float*)src->Mat.ptr, 
                                        kernel_H->_kernel_data.ptr, 
                                        _Hconv_res.ptr, 
                                        make_uint2(dst->Pitch() / 4, src->Height()), 
                                        kernel_H->_ker_length,
                                        src->Pitch(),
                                        Hconv_res_dims.x, 
                                        reg_WL, &t1D, _loop);

    decx::vis::_gaussian_V_uint8_caller(_Hconv_res.ptr, 
                                        kernel_V->_kernel_data.ptr, 
                                        (double*)dst->Mat.ptr,
                                        make_uint2(dst->Pitch() / 4, dst->Height()), 
                                        kernel_V->_ker_length, 
                                        Hconv_res_dims.x, 
                                        dst->Pitch() / 2, &t1D);

    decx::alloc::_host_virtual_page_dealloc(&_Hconv_res);
}




template <bool _print>
static void decx::vis::_gaussian_uchar4_BC_zero(decx::_Matrix* src, decx::vis::gaussian_kernel1D* kernel_H, decx::vis::gaussian_kernel1D* kernel_V, decx::_Matrix* dst,
    de::DH* handle)
{
    const uint2 tmp_src_dims = make_uint2(decx::utils::ceil<uint>(src->Width() + kernel_H->_ker_length - 1, 4) * 4,
        src->Height());
    const uint2 Hconv_res_dims = make_uint2(decx::utils::ceil<uint32_t>(dst->Width(), 4) * 4 * 4, 
        src->Height() + kernel_V->_ker_length - 1);

    decx::PtrInfo<float> tmp_src;
    decx::PtrInfo<float> _Hconv_res;
    if (decx::alloc::_host_virtual_page_malloc(&tmp_src, (size_t)tmp_src_dims.x * (size_t)tmp_src_dims.y * sizeof(uchar4))) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_ALLOCATION,
            ALLOC_FAIL);
        return;
    }
    if (decx::alloc::_host_virtual_page_malloc(&_Hconv_res, (size_t)Hconv_res_dims.x * (size_t)Hconv_res_dims.y * sizeof(float))) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_ALLOCATION,
            ALLOC_FAIL);
        return;
    }

    float* start_place_src = DECX_PTR_SHF_XY<float, float>(tmp_src.ptr, 0, kernel_H->_ker_length / 2, tmp_src_dims.x);
    decx::_cpy2D_anybit_caller<float>((float*)src->Mat.ptr, 
                                      start_place_src, 
                                      src->Pitch(), 
                                      tmp_src_dims.x, 
                                      make_uint2(src->Width(), src->Height()));

    const uint conc_thr = (uint)decx::cpu::_get_permitted_concurrency();
    decx::utils::_thr_1D t1D(conc_thr);

    const uint _loop = (kernel_H->_ker_length - 1) / 4;
    ushort reg_WL = (ushort)(kernel_H->_ker_length - _loop * 4);

    float* _mid_ptr = DECX_PTR_SHF_XY<float, float>(_Hconv_res.ptr, kernel_V->_ker_length / 2, 0, Hconv_res_dims.x);
    decx::vis::_gaussian_H_uchar4_caller((float*)tmp_src.ptr, 
                                        kernel_H->_kernel_data.ptr, 
                                        _mid_ptr, 
                                        make_uint2(decx::utils::ceil<uint32_t>(dst->Pitch(), 4), dst->Height()),
                                        kernel_H->_ker_length,
                                        tmp_src_dims.x,
                                        Hconv_res_dims.x, 
                                        reg_WL, &t1D, _loop);

    decx::vis::_gaussian_V_uint8_caller(_Hconv_res.ptr, 
                                        kernel_V->_kernel_data.ptr, 
                                        (double*)dst->Mat.ptr,
                                        make_uint2(decx::utils::ceil<uint32_t>(dst->Pitch(), 4), dst->Height()),
                                        kernel_V->_ker_length, 
                                        Hconv_res_dims.x, 
                                        dst->Pitch() / 2, &t1D);

    decx::alloc::_host_virtual_page_dealloc(&tmp_src);
    decx::alloc::_host_virtual_page_dealloc(&_Hconv_res);
}





template <bool _print>
static void decx::vis::_gaussian_uchar4_BC_reflect(decx::_Matrix* src, decx::vis::gaussian_kernel1D* kernel_H, decx::vis::gaussian_kernel1D* kernel_V, decx::_Matrix* dst,
    de::DH* handle)
{
    const uint2 tmp_src_dims = make_uint2(decx::utils::ceil<uint>(src->Width() + kernel_H->_ker_length - 1, 4) * 4,
        src->Height());
    const uint2 Hconv_res_dims = make_uint2(decx::utils::ceil<uint32_t>(dst->Width(), 4) * 4 * 4, 
        src->Height() + kernel_V->_ker_length - 1);

    decx::PtrInfo<float> tmp_src;
    decx::PtrInfo<float> _Hconv_res;

    if (decx::alloc::_host_virtual_page_malloc(&tmp_src, tmp_src_dims.x * tmp_src_dims.y * sizeof(uchar4))) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_ALLOCATION,
            ALLOC_FAIL);
        return;
    }
    if (decx::alloc::_host_virtual_page_malloc(&_Hconv_res, Hconv_res_dims.x * Hconv_res_dims.y * sizeof(float))) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_ALLOCATION,
            ALLOC_FAIL);
        return;
    }

    float* start_place_src = DECX_PTR_SHF_XY<float, float>(tmp_src.ptr, 0, kernel_H->_ker_length / 2, tmp_src_dims.x);
    decx::bp::_extend_LR_reflect_b32_2D<_print>((float*)src->Mat.ptr, tmp_src.ptr, make_uint2(kernel_H->_ker_length / 2, kernel_H->_ker_length / 2),
        src->Pitch(), tmp_src_dims.x, src->Width(), src->Height(), handle);

    const uint conc_thr = (uint)decx::cpu::_get_permitted_concurrency();
    decx::utils::_thr_1D t1D(conc_thr);

    const uint _loop = (kernel_H->_ker_length - 1) / 4;
    ushort reg_WL = (ushort)(kernel_H->_ker_length - _loop * 4);

    float* _mid_ptr = DECX_PTR_SHF_XY<float, float>(_Hconv_res.ptr, kernel_V->_ker_length / 2, 0, Hconv_res_dims.x);
    decx::vis::_gaussian_H_uchar4_caller((float*)tmp_src.ptr, 
                                        kernel_H->_kernel_data.ptr, 
                                        _mid_ptr, 
                                        make_uint2(decx::utils::ceil<uint32_t>(dst->Pitch(), 4), dst->Height()), 
                                        kernel_H->_ker_length,
                                        tmp_src_dims.x,
                                        Hconv_res_dims.x, 
                                        reg_WL, &t1D, _loop);
    
    decx::bp::_extend_TB_reflect_b32_2D(_Hconv_res.ptr, make_uint2(kernel_V->_ker_length / 2, kernel_V->_ker_length / 2), Hconv_res_dims.x, dst->Height());
    
    decx::vis::_gaussian_V_uint8_caller(_Hconv_res.ptr, 
                                        kernel_V->_kernel_data.ptr, 
                                        (double*)dst->Mat.ptr,
                                        make_uint2(decx::utils::ceil<uint32_t>(dst->Pitch(), 4), dst->Height()),
                                        kernel_V->_ker_length, 
                                        Hconv_res_dims.x, 
                                        dst->Pitch() / 2, &t1D);
    
    decx::alloc::_host_virtual_page_dealloc(&tmp_src);
    decx::alloc::_host_virtual_page_dealloc(&_Hconv_res);
}





template <bool _print>
static void 
decx::vis::_Gaussian_filter_uchar4_organisor(decx::_Matrix* src, decx::_Matrix* dst, const uint2 neighbor_dims, const float2 sigmaXY,
    const uint2 centerXY, de::DH* handle, const bool _is_central, const int border_type)
{
    
    decx::vis::gaussian_kernel1D kernel_H(neighbor_dims.x, sigmaXY.x, _is_central, centerXY.x);
    decx::vis::gaussian_kernel1D kernel_V(neighbor_dims.y, sigmaXY.y, _is_central, centerXY.y);
    kernel_H.generate();
    kernel_V.generate();
    
    clock_t s, e;
    switch (border_type)
    {
    case de::extend_label::_EXTEND_NONE_:
        dst->re_construct(src->Type(), src->Width() - neighbor_dims.x + 1, src->Height() - neighbor_dims.y + 1);
        decx::vis::_gaussian_uchar4_NB<_print>(src, &kernel_H, &kernel_V, dst, handle);
        break;

    case de::extend_label::_EXTEND_CONSTANT_:
        
        dst->re_construct(src->Type(), src->Width(), src->Height());
        
        decx::vis::_gaussian_uchar4_BC_zero<_print>(src, &kernel_H, &kernel_V, dst, handle);
        break;

    case de::extend_label::_EXTEND_REFLECT_:
        dst->re_construct(src->Type(), src->Width(), src->Height());
        decx::vis::_gaussian_uchar4_BC_reflect<_print>(src, &kernel_H, &kernel_V, dst, handle);
        break;
    default:
        break;
    }
}



namespace de
{
    namespace vis {
        namespace cpu 
        {
            _DECX_API_ de::DH Gaussian_Filter(de::Matrix& src, de::Matrix& dst, const de::Point2D neighbor_dims,
                const de::Point2D_f sigmaXY, 
                const int border_type, 
                const bool _is_central = true, 
                const de::Point2D centerXY = de::Point2D(0, 0));
        }
    }
}


#endif