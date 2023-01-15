/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/


#include "../../../core/basic.h"
#include "../../../classes/Matrix.h"
#include "edge_det_ops.h"
#include "../../../core/utils/fragment_arrangment.h"


namespace de
{
    namespace vis {
        enum Canny_Methods {
            DE_SOBEL = 0,
            DE_SCHARR = 1
        };

        namespace cpu {
            _DECX_API_ de::DH
                Find_Edge(de::Matrix& src, de::Matrix& dst, const float _L_threshold, const float _H_threshold, const int method);
        }
    }
}


_DECX_API_ de::DH
de::vis::cpu::Find_Edge(de::Matrix& src, de::Matrix& dst, const float _L_threshold, const float _H_threshold, const int method)
{
    de::DH handle;
    if (!decx::cpI.is_init) {
        Print_Error_Message(4, CPU_NOT_INIT);
        decx::err::CPU_Not_init(&handle);
        return handle;
    }

    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);
    decx::_Matrix* _dst = dynamic_cast<decx::_Matrix*>(&dst);
    const uint2 _proc_dims = make_uint2(decx::utils::ceil<uint>(_src->width, 8), _src->height);
    //const uint2 Gmap_dims = make_uint2(decx::utils::ceil<uint>(_src->width, 8) * 16, _proc_dims.y);
    const uint2 Gmap_dims = make_uint2(decx::utils::ceil<uint>(_src->width, 8) * 8, _proc_dims.y);

    decx::PtrInfo<float> gradient_info_map, dir_info_map;
    if (decx::alloc::_host_virtual_page_malloc(&gradient_info_map, Gmap_dims.x * Gmap_dims.y * sizeof(float))) {
        Print_Error_Message(4, ALLOC_FAIL);
        decx::err::AllocateFailure(&handle);
        return handle;
    }
    if (decx::alloc::_host_virtual_page_malloc(&dir_info_map, Gmap_dims.x * Gmap_dims.y * sizeof(float))) {
        Print_Error_Message(4, ALLOC_FAIL);
        decx::err::AllocateFailure(&handle);
        return handle;
    }

    uint8_t* start_ptr = decx::utils::ptr_shift_xy<uint8_t, uint8_t>((uint8_t*)_dst->Mat.ptr, 1, 1, _dst->pitch);
    float* start_ptr_G = decx::utils::ptr_shift_xy<float, float>((float*)gradient_info_map.ptr, 1, 1, Gmap_dims.x);
    float* start_ptr_D = decx::utils::ptr_shift_xy<float, float>((float*)dir_info_map.ptr, 1, 1, Gmap_dims.x);

    decx::vis::CPUK::Canny_operator_kernel _ops_kernel_T = NULL,
        _ops_kernel_B = NULL, _ops_kernel = NULL, _ops_kernel_TB = NULL;
    switch (method)
    {
    case de::vis::Canny_Methods::DE_SOBEL:
        _ops_kernel_T = decx::vis::CPUK::Sobel_XY_uint8_T;
        _ops_kernel_B = decx::vis::CPUK::Sobel_XY_uint8_B;
        _ops_kernel = decx::vis::CPUK::Sobel_XY_uint8;
        _ops_kernel_TB = decx::vis::CPUK::Sobel_XY_uint8_TB;
        break;
    case de::vis::Canny_Methods::DE_SCHARR:
        _ops_kernel_T = decx::vis::CPUK::Scharr_XY_uint8_T;
        _ops_kernel_B = decx::vis::CPUK::Scharr_XY_uint8_B;
        _ops_kernel = decx::vis::CPUK::Scharr_XY_uint8;
        _ops_kernel_TB = decx::vis::CPUK::Scharr_XY_uint8_TB;
        break;
    default:
        break;
    }

    if (_proc_dims.y > decx::cpI.cpu_concurrency * 3) {
        decx::utils::_thread_arrange_1D t1D(decx::cpI.cpu_concurrency);
        decx::utils::frag_manager f_mgr_prep, f_mgr_sufx;
        decx::utils::frag_manager_gen(&f_mgr_prep, _proc_dims.y, t1D.total_thread);
        decx::utils::frag_manager_gen(&f_mgr_sufx, _proc_dims.y - 2, t1D.total_thread);

        const size_t src_frag_size = (size_t)_src->pitch * (size_t)f_mgr_prep.frag_len;
        const size_t dst_frag_size = (size_t)Gmap_dims.x * (size_t)f_mgr_prep.frag_len;

        const uint8_t* loc_src = reinterpret_cast<uint8_t*>(_src->Mat.ptr);
        float* loc_dst = gradient_info_map.ptr;
        float* loc_dst_D = dir_info_map.ptr;

        // Calculate for gradients and directions
        t1D._async_thread[0] = decx::cpu::register_task(&decx::thread_pool, _ops_kernel_T,
            loc_src, loc_dst, dir_info_map.ptr, _src->pitch, Gmap_dims.x, make_uint2(_proc_dims.x, f_mgr_prep.frag_len));

        loc_src += src_frag_size;
        loc_dst += dst_frag_size;
        loc_dst_D += dst_frag_size;

        for (int i = 1; i < t1D.total_thread - 1; ++i) {
            t1D._async_thread[i] = decx::cpu::register_task(&decx::thread_pool, _ops_kernel,
                loc_src, loc_dst, dir_info_map.ptr, _src->pitch, Gmap_dims.x, make_uint2(_proc_dims.x, f_mgr_prep.frag_len));

            loc_src += src_frag_size;
            loc_dst += dst_frag_size;
            loc_dst_D += dst_frag_size;
        }
        t1D._async_thread[t1D.total_thread - 1] = decx::cpu::register_task(&decx::thread_pool, _ops_kernel_B,
            loc_src, loc_dst, dir_info_map.ptr, _src->pitch, Gmap_dims.x,
            make_uint2(_proc_dims.x, f_mgr_prep.is_left ? f_mgr_prep.frag_left_over : f_mgr_prep.frag_len));

        t1D.__sync_all_threads();

        // Canny post-processing
        for (int i = 0; i < t1D.total_thread - 1; ++i) {
            t1D._async_thread[i] = decx::cpu::register_task(&decx::thread_pool, decx::vis::CPUK::Edge_Detector_Post_processing,
                start_ptr_G, start_ptr_D, start_ptr, Gmap_dims.x, _dst->pitch,
                make_uint2(_dst->width - 2, f_mgr_sufx.frag_len), make_float2(powf(_L_threshold, 2), powf(_H_threshold, 2)));

            start_ptr_G += Gmap_dims.x * f_mgr_sufx.frag_len;
            start_ptr_D += Gmap_dims.x * f_mgr_sufx.frag_len;
            start_ptr += _dst->pitch * f_mgr_sufx.frag_len;
        }
        t1D._async_thread[t1D.total_thread - 1] = decx::cpu::register_task(&decx::thread_pool, decx::vis::CPUK::Edge_Detector_Post_processing,
            start_ptr_G, start_ptr_D, start_ptr, Gmap_dims.x, _dst->pitch,
            make_uint2(_dst->width - 2, f_mgr_sufx.is_left ? f_mgr_sufx.frag_left_over : f_mgr_sufx.frag_len), make_float2(powf(_L_threshold, 2), powf(_H_threshold, 2)));

        t1D.__sync_all_threads();
    }
    else {
        (*_ops_kernel_TB)((uint8_t*)_src->Mat.ptr, gradient_info_map.ptr, dir_info_map.ptr, _src->pitch, Gmap_dims.x,
            make_uint2(_proc_dims.x, _proc_dims.y));

        decx::vis::CPUK::Edge_Detector_Post_processing(start_ptr_G, start_ptr_D, start_ptr, Gmap_dims.x, _dst->pitch,
            make_uint2(_dst->width - 2, _dst->height - 2), make_float2(powf(_L_threshold, 2), powf(_H_threshold, 2)));
    }
    
    decx::alloc::_host_virtual_page_dealloc(&gradient_info_map);

    decx::err::Success(&handle);
    return handle;
}