/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _BILATERAL_FILTER_H_
#define _BILATERAL_FILTER_H_


#include "bilateral_filter_exec.h"
#include "../../../core/memory_management/PtrInfo.h"
#include "../../../core/allocators.h"
#include "../../../classes/Matrix.h"
#include "../../../DSP/convolution/conv_utils.h"
#include "../../../BLAS/basic_process/extension/CPU/extend_reflect.h"
#include "../../../BLAS/basic_process/extension/extend_flags.h"
#include "exp_values_LUT.h"
#include "../../../BLAS/basic_process/rect_and_cube/CPU/rect_copy2D_exec.h"


namespace decx
{
    namespace vis {
        /**
        * @param proc_dim : ~.x -> dst->Pitch() (in its element); ~.y -> dst->Height()
        * @param Wsrc : Pitch of source matrix, in its element (uint8_t)
        * @param Wsrc : Pitch of destinated matrix, in its element (uint8_t)
        */
        template <bool _print>
        static void _bilateral_uint8_organiser(const double* src, double* dst, const float2 sigmas,
            const uint2 proc_dim, const uint2 ker_dims, const uint Wsrc,
            const uint Wdst, decx::utils::_thr_1D* t1D, decx::utils::frag_manager* f_mgr,
            de::DH *handle);


        template <bool _print>
        static void _bilateral_uchar4_organiser(const float* src, float* dst, const float2 sigmas,
            const uint2 proc_dim, const uint2 ker_dims, const uint Wsrc,
            const uint Wdst, decx::utils::_thr_1D* t1D, decx::utils::frag_manager* f_mgr,
            de::DH* handle);
    }
}



template <bool _print>
void decx::vis::_bilateral_uint8_organiser(const double*                      src, 
                                           double*                            dst,
                                           const float2                       sigmas,       // .x -> sigma space; .y -> sigma color
                                           const uint2                        proc_dim, 
                                           const uint2                        ker_dims,
                                           const uint                         Wsrc,
                                           const uint                         Wdst,
                                           decx::utils::_thr_1D*              t1D,
                                           decx::utils::frag_manager*         f_mgr,
                                           de::DH*                            handle)
{
    const uint _exp_chart_dist_len = max(ker_dims.x, ker_dims.y) / 2 + 1;
    constexpr uint _exp_chart_diff_len = 256;
    
    decx::PtrInfo<float> _exp_chart_dist, _exp_chart_diff;
    if (decx::alloc::_host_virtual_page_malloc(&_exp_chart_dist, _exp_chart_dist_len * sizeof(float))) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_ALLOCATION,
            ALLOC_FAIL);
        return;
    }
    if (decx::alloc::_host_virtual_page_malloc(&_exp_chart_diff, _exp_chart_diff_len * sizeof(float))) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_ALLOCATION,
            ALLOC_FAIL);
        return;
    }

    for (int i = 0; i < _exp_chart_dist_len; ++i) {
        _exp_chart_dist.ptr[i] = expf(-(float)(i * i) / sigmas.x);
    }

    for (int i = 0; i < _exp_chart_diff_len; ++i) {
        _exp_chart_diff.ptr[i] = expf(-(float)(i * i) / sigmas.y);
    }

    const uint _loop = (ker_dims.x - 1) / 16;
    const ushort reg_WL = (ushort)(ker_dims.x - _loop * 16);
    const uint2 _proc_dims = make_uint2(proc_dim.x / 16, proc_dim.y);

    decx::vis::_bilateral_uint8_caller(src, _exp_chart_dist.ptr, _exp_chart_diff.ptr, dst,
                                       _proc_dims, ker_dims, Wsrc / 16, Wdst / 16, reg_WL, 
                                       t1D, f_mgr, _loop);

    decx::alloc::_host_virtual_page_dealloc(&_exp_chart_dist);
    decx::alloc::_host_virtual_page_dealloc(&_exp_chart_diff);
}





template <bool _print>
void decx::vis::_bilateral_uchar4_organiser(const float*                      src, 
                                           float*                             dst,
                                           const float2                       sigmas,       // .x -> sigma space; .y -> sigma color
                                           const uint2                        proc_dim, 
                                           const uint2                        ker_dims,
                                           const uint                         Wsrc,
                                           const uint                         Wdst,
                                           decx::utils::_thr_1D*              t1D,
                                           decx::utils::frag_manager*         f_mgr,
                                           de::DH*                            handle)
{
    decx::PtrInfo<float> _exp_chart_dist, _exp_chart_diff;
    const uint _exp_chart_dist_len = max(ker_dims.x, ker_dims.y) / 2 + 1;
    constexpr uint _exp_chart_diff_len = 256;
    if (decx::alloc::_host_virtual_page_malloc(&_exp_chart_dist, _exp_chart_dist_len * sizeof(float))) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_ALLOCATION,
            ALLOC_FAIL);
        return;
    }
    if (decx::alloc::_host_virtual_page_malloc(&_exp_chart_diff, _exp_chart_diff_len * sizeof(float))) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_ALLOCATION,
            ALLOC_FAIL);
        return;
    }

    for (int i = 0; i < _exp_chart_dist_len; ++i) {
        _exp_chart_dist.ptr[i] = expf(-(float)(i * i) / sigmas.x);
    }

    for (int i = 0; i < _exp_chart_diff_len; ++i) {
        _exp_chart_diff.ptr[i] = expf(-(float)(i * i) / sigmas.y);
    }


    const uint _loop = (ker_dims.x - 1) / 4;
    const ushort reg_WL = (ushort)(ker_dims.x - _loop * 4);
    const uint2 _proc_dims = make_uint2(proc_dim.x / 4, proc_dim.y);

    decx::vis::_bilateral_uchar4_caller(src, _exp_chart_dist.ptr, _exp_chart_diff.ptr, dst, 
                                       _proc_dims, ker_dims, Wsrc, Wdst, reg_WL, 
                                       t1D, f_mgr, _loop);

    decx::alloc::_host_virtual_page_dealloc(&_exp_chart_dist);
    decx::alloc::_host_virtual_page_dealloc(&_exp_chart_diff);
}



namespace decx
{
    namespace vis
    {
        template <bool _print>
        static void _bilateral_uint8_NB(decx::_Matrix* src, decx::_Matrix* dst, const uint2 neighbor_dims, const float2 sigmas_raw, de::DH* handle);



        template <bool _print>
        static void _bilateral_uint8_BC(decx::_Matrix* src, decx::_Matrix* dst, const uint2 neighbor_dims, const float2 sigmas_raw, de::DH* handle, const int border_type);



        template <bool _print>
        static void _bilateral_uchar4_NB(decx::_Matrix* src, decx::_Matrix* dst, const uint2 neighbor_dims, const float2 sigmas_raw, de::DH* handle);



        template <bool _print>
        static void _bilateral_uchar4_BC(decx::_Matrix* src, decx::_Matrix* dst, const uint2 neighbor_dims, const float2 sigmas_raw, de::DH* handle, const int border_type);
    }
}



template <bool _print>
static void decx::vis::_bilateral_uint8_NB(decx::_Matrix* src, decx::_Matrix* dst, const uint2 neighbor_dims, const float2 sigmas_raw, de::DH* handle)
{
    const uint conc_thr = (uint)decx::cpu::_get_permitted_concurrency();
    decx::utils::frag_manager* f_mgr = NULL;
    decx::_thread_dispatch_for_conv2(&f_mgr, dst->Height(), conc_thr, _BLOCKED_CONV2_UINT8_H_, dst->Pitch() / 8);
    if (f_mgr == NULL) {
        return;
    }
    decx::utils::_thr_1D t1D(conc_thr);

    decx::vis::_bilateral_uint8_organiser<_print>((double*)src->Mat.ptr, 
                                          (double*)dst->Mat.ptr,
                                          make_float2(powf(sigmas_raw.x, 2) * 2, powf(sigmas_raw.y, 2) * 2),
                                          make_uint2(dst->Pitch(), dst->Height()), 
                                          neighbor_dims,
                                          src->Pitch(), dst->Pitch(), 
                                          &t1D, f_mgr, handle);
}




template <bool _print> static void 
decx::vis::_bilateral_uchar4_NB(decx::_Matrix* src, decx::_Matrix* dst, const uint2 neighbor_dims, const float2 sigmas_raw, de::DH* handle)
{
    const uint conc_thr = (uint)decx::cpu::_get_permitted_concurrency();
    decx::utils::frag_manager* f_mgr = NULL;
    decx::_thread_dispatch_for_conv2(&f_mgr, dst->Height(), conc_thr, _BLOCKED_CONV2_UINT8_H_, dst->Pitch() / 4);
    if (f_mgr == NULL) {
        return;
    }
    decx::utils::_thr_1D t1D(conc_thr);

    decx::vis::_bilateral_uchar4_organiser<_print>((float*)src->Mat.ptr, 
                                          (float*)dst->Mat.ptr,
                                          make_float2(powf(sigmas_raw.x, 2) * 2, powf(sigmas_raw.y, 2) * 2),
                                          make_uint2(dst->Pitch(), dst->Height()), 
                                          neighbor_dims,
                                          src->Pitch(), dst->Pitch(), 
                                          &t1D, f_mgr, handle);
}




template <bool _print>
static void decx::vis::_bilateral_uint8_BC(decx::_Matrix* src, decx::_Matrix* dst, const uint2 neighbor_dims, const float2 sigmas_raw, 
    de::DH* handle, const int border_type)
{
    uint2 tmp_src_dims = make_uint2(decx::utils::ceil<uint>(src->Width() + neighbor_dims.x - 1, 16) * 16,
        src->Height() + neighbor_dims.y - 1);

    decx::PtrInfo<uint8_t> tmp_src;
    if (decx::alloc::_host_virtual_page_malloc(&tmp_src, tmp_src_dims.x * tmp_src_dims.y * sizeof(uint8_t))) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_ALLOCATION,
            ALLOC_FAIL);
        return;
    }

    uint8_t* start_place_src = DECX_PTR_SHF_XY<uint8_t, uint8_t>(tmp_src.ptr, neighbor_dims.y / 2, neighbor_dims.x / 2, tmp_src_dims.x);
    switch (border_type)
    {
    case (de::extend_label::_EXTEND_CONSTANT_):
        decx::_cpy2D_anybit_caller<uint8_t>(
            (uint8_t*)src->Mat.ptr, start_place_src, src->Pitch(), tmp_src_dims.x, make_uint2(src->Width(), src->Height()));
        break;
    case (de::extend_label::_EXTEND_REFLECT_):
        decx::bp::_extend_reflect_b8_2D<_print>((uint8_t*)src->Mat.ptr, tmp_src.ptr,
            make_uint4(neighbor_dims.x / 2, neighbor_dims.x / 2, neighbor_dims.y / 2, neighbor_dims.y / 2),
            src->Pitch(), tmp_src_dims.x, src->Width(), src->Height(), handle);
        break;
    default:
        break;
    }

    const uint conc_thr = (uint)decx::cpu::_get_permitted_concurrency();
    decx::utils::frag_manager* f_mgr = NULL;
    decx::_thread_dispatch_for_conv2(&f_mgr, dst->Height(), conc_thr, _BLOCKED_CONV2_UINT8_H_, dst->Pitch() / 8);
    if (f_mgr == NULL) {
        return;
    }
    decx::utils::_thr_1D t1D(conc_thr);

    decx::vis::_bilateral_uint8_organiser<_print>((double*)tmp_src.ptr, (double*)dst->Mat.ptr,
                                                   make_float2(powf(sigmas_raw.x, 2) * 2, powf(sigmas_raw.y, 2) * 2),
                                                   make_uint2(dst->Pitch(), dst->Height()), 
                                                   neighbor_dims,
                                                   tmp_src_dims.x, dst->Pitch(), 
                                                   &t1D, f_mgr, handle);

    decx::alloc::_host_virtual_page_dealloc(&tmp_src);
    delete f_mgr;
}





template <bool _print>
static void decx::vis::_bilateral_uchar4_BC(decx::_Matrix* src, decx::_Matrix* dst, const uint2 neighbor_dims, const float2 sigmas_raw, 
    de::DH* handle, const int border_type)
{
    uint2 tmp_src_dims = make_uint2(decx::utils::ceil<uint>(src->Width() + neighbor_dims.x - 1, 8) * 8,
        src->Height() + neighbor_dims.y - 1);

    decx::PtrInfo<float> tmp_src;
    if (decx::alloc::_host_virtual_page_malloc(&tmp_src, tmp_src_dims.x * tmp_src_dims.y * sizeof(uchar4))) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_ALLOCATION,
            ALLOC_FAIL);
        return;
    }

    float* start_place_src = DECX_PTR_SHF_XY<float, float>(tmp_src.ptr, neighbor_dims.y / 2, neighbor_dims.x / 2, tmp_src_dims.x);
    switch (border_type)
    {
    case (de::extend_label::_EXTEND_CONSTANT_):
        decx::_cpy2D_anybit_caller<float>(
            (float*)src->Mat.ptr, start_place_src, src->Pitch(), tmp_src_dims.x, make_uint2(src->Width(), src->Height()));
        break;
    case (de::extend_label::_EXTEND_REFLECT_):
        decx::bp::_extend_reflect_b32_2D<_print>((float*)src->Mat.ptr, tmp_src.ptr,
            make_uint4(neighbor_dims.x / 2, neighbor_dims.x / 2, neighbor_dims.y / 2, neighbor_dims.y / 2),
            src->Pitch(), tmp_src_dims.x, src->Width(), src->Height(), handle);
        break;
    default:
        break;
    }

    const uint conc_thr = (uint)decx::cpu::_get_permitted_concurrency();
    decx::utils::frag_manager* f_mgr = NULL;
    decx::_thread_dispatch_for_conv2(&f_mgr, dst->Height(), conc_thr, _BLOCKED_CONV2_UINT8_H_, dst->Pitch() / 4);
    
    if (f_mgr == NULL) {
        return;
    }
    decx::utils::_thr_1D t1D(conc_thr);

    decx::vis::_bilateral_uchar4_organiser<_print>((float*)tmp_src.ptr, (float*)dst->Mat.ptr,
                                                   make_float2(powf(sigmas_raw.x, 2) * 2, powf(sigmas_raw.y, 2) * 2),
                                                   make_uint2(dst->Pitch(), dst->Height()), 
                                                   neighbor_dims,
                                                   tmp_src_dims.x, dst->Pitch(), 
                                                   &t1D, f_mgr, handle);

    decx::alloc::_host_virtual_page_dealloc(&tmp_src);
    delete f_mgr;
}




namespace de
{
    namespace vis {
        namespace cpu {
            _DECX_API_ de::DH Bilateral_Filter(de::Matrix& src, de::Matrix& dst, const de::Point2D neighbor_dims,
                const float sigma_space, const float sigma_color, const int border_type);
        }
    }
}


#endif