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


#include "edge_detectors.h"
#include "../../../core/thread_management/thread_arrange.h"


_DECX_API_ de::DH
de::vis::cpu::Find_Edge(de::Matrix& src, de::Matrix& dst, const float _L_threshold, const float _H_threshold, const int method)
{
    de::DH handle;
    if (!decx::cpu::_is_CPU_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CPU_not_init,
            CPU_NOT_INIT);
        return handle;
    }

    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);
    decx::_Matrix* _dst = dynamic_cast<decx::_Matrix*>(&dst);
    
    const uint2 Dmap_dims = make_uint2(decx::utils::ceil<uint>(_src->Width() - 2, 8) * 8, _src->Height() - 2);
    const uint2 Gmap_dims = make_uint2(decx::utils::ceil<uint>(Dmap_dims.x + 2, 8) * 8, _src->Height());
    const uint2 _proc_dims = Dmap_dims;

    decx::PtrInfo<float> gradient_info_map, dir_info_map;
    if (decx::alloc::_host_virtual_page_malloc(&gradient_info_map, Gmap_dims.x * Gmap_dims.y * sizeof(float), true)) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_ALLOCATION,
            ALLOC_FAIL);
        return handle;
    }
    if (decx::alloc::_host_virtual_page_malloc(&dir_info_map, Dmap_dims.x * Dmap_dims.y * sizeof(float), true)) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_ALLOCATION,
            ALLOC_FAIL);
        return handle;
    }

    uint8_t* start_ptr = decx::utils::ptr_shift_xy<uint8_t, uint8_t>((uint8_t*)_dst->Mat.ptr, 1, 1, _dst->Pitch());

    decx::PtrInfo<float> cache;

    if (_proc_dims.y > decx::cpu::_get_permitted_concurrency() * 64) 
    {
        decx::vis::CPUK::_canny_operator_ptr _op_ptr = NULL;
        switch (method)
        {
        case de::vis::DE_SOBEL:
            _op_ptr = &decx::vis::CPUK::Sobel_XY_uint8;
            break;
        case de::vis::DE_SCHARR:
            _op_ptr = &decx::vis::CPUK::Scharr_XY_uint8;
            break;
        default:
            break;
        }

        decx::utils::_thread_arrange_1D t1D(decx::cpu::_get_permitted_concurrency());
        decx::utils::frag_manager f_mgr;
        decx::utils::frag_manager_gen(&f_mgr, _proc_dims.y, t1D.total_thread);

        const size_t frag_src = (size_t)f_mgr.frag_len * (size_t)_src->Pitch();
        const size_t frag_dst = (size_t)f_mgr.frag_len * (size_t)_dst->Pitch();
        const size_t frag_G = (size_t)f_mgr.frag_len * (size_t)Gmap_dims.x;
        const size_t frag_D = (size_t)f_mgr.frag_len * (size_t)Dmap_dims.x;

        const uint8_t* loc_src = (uint8_t*)_src->Mat.ptr;
        float* loc_G = gradient_info_map.ptr;
        float* loc_D = dir_info_map.ptr;

        for (int i = 0; i < t1D.total_thread - 1; ++i) {
            t1D._async_thread[i] = decx::cpu::register_task_default(_op_ptr,
                loc_src, loc_G, loc_D, Gmap_dims.x, Dmap_dims.x,
                _src->Pitch(), make_uint2(_proc_dims.x / 8, f_mgr.frag_len));

            loc_src += frag_src;
            loc_G += frag_G;
            loc_D += frag_D;
        }
        const uint32_t _L = f_mgr.is_left ? f_mgr.frag_left_over : f_mgr.frag_len;
        t1D._async_thread[t1D.total_thread - 1] = decx::cpu::register_task_default(_op_ptr,
            loc_src, loc_G, loc_D, Gmap_dims.x, Dmap_dims.x,
            _src->Pitch(), make_uint2(_proc_dims.x / 8, _L));

        t1D.__sync_all_threads();

        // post-processing
        loc_src = (uint8_t*)_src->Mat.ptr;
        loc_G = gradient_info_map.ptr;
        loc_D = dir_info_map.ptr;
        uint8_t* loc_dst = (uint8_t*)start_ptr;

        if (decx::alloc::_host_virtual_page_malloc(&cache, t1D.total_thread * 48 * sizeof(float), true)) {
            decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_ALLOCATION,
                ALLOC_FAIL);
            return handle;
        }

        for (int i = 0; i < t1D.total_thread - 1; ++i) {
            t1D._async_thread[i] = decx::cpu::register_task_default(decx::vis::CPUK::_Edge_Detector_Post_processing,
                loc_G, loc_D, cache.ptr + i * 48, (uint64_t*)loc_dst, Gmap_dims.x, Dmap_dims.x,
                _dst->Pitch() / 8, make_uint2(Dmap_dims.x / 8, f_mgr.frag_len), make_float2(powf(_L_threshold, 2), powf(_H_threshold, 2)));

            loc_dst += frag_dst;
            loc_G += frag_G;
            loc_D += frag_D;
        }

        t1D._async_thread[t1D.total_thread - 1] = decx::cpu::register_task_default(decx::vis::CPUK::_Edge_Detector_Post_processing,
            loc_G, loc_D, cache.ptr + (t1D.total_thread - 1) * 48, (uint64_t*)loc_dst, Gmap_dims.x, Dmap_dims.x,
            _dst->Pitch() / 8, make_uint2(Dmap_dims.x / 8, _L), make_float2(powf(_L_threshold, 2), powf(_H_threshold, 2)));

        t1D.__sync_all_threads();
    }
    else {
        if (decx::alloc::_host_virtual_page_malloc(&cache, 48 * sizeof(float), true)) {
            decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_ALLOCATION,
                ALLOC_FAIL);
            return handle;
        }
        switch (method)
        {
        case de::vis::DE_SOBEL:
            decx::vis::CPUK::Sobel_XY_uint8((uint8_t*)_src->Mat.ptr, gradient_info_map.ptr, dir_info_map.ptr, Gmap_dims.x, Dmap_dims.x,
                _src->Pitch(), make_uint2(_proc_dims.x / 8, _proc_dims.y));
            break;

        case de::vis::DE_SCHARR:
            decx::vis::CPUK::Scharr_XY_uint8((uint8_t*)_src->Mat.ptr, gradient_info_map.ptr, dir_info_map.ptr, Gmap_dims.x, Dmap_dims.x,
                _src->Pitch(), make_uint2(_proc_dims.x / 8, _proc_dims.y));
            break;
        default:
            break;
        }
        
        decx::vis::CPUK::_Edge_Detector_Post_processing(gradient_info_map.ptr, dir_info_map.ptr, cache.ptr, (uint64_t*)start_ptr, Gmap_dims.x, Dmap_dims.x,
            _dst->Pitch() / 8, make_uint2(Dmap_dims.x / 8, Dmap_dims.y), make_float2(powf(_L_threshold, 2), powf(_H_threshold, 2)));
    }

    decx::alloc::_host_virtual_page_dealloc(&gradient_info_map);
    decx::alloc::_host_virtual_page_dealloc(&dir_info_map);
    decx::alloc::_host_virtual_page_dealloc(&cache);

    decx::err::Success(&handle);
    return handle;
}