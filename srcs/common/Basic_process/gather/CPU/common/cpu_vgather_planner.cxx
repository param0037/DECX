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

#include "cpu_vgather_planner.h"
#include "../gather_kernels.h"
#include "../../VGT_kernels_LUT_selector.h"

namespace decx
{
    static void* VGT2D_exec_LUT[6][6] = 
    {   // float input
        {   (void*)decx::CPUK::gather2D_fp32_exec_bilinear, // bilinear
            (void*)decx::CPUK::gather2D_fp32_exec_nearest,   // nearest
            // other output types ...
        }, 
        // uint8_t input
        {   NULL,       // type_out = float bilinear
            NULL,       // type_out = float nearest
            (void*)decx::CPUK::gather2D_uint8_exec_bilinear, // bilinear
            NULL,       // nearest
            NULL,
        }, 
        // uchar4 input
        {   NULL,

        }
    };
}


_CRSR_ void decx::cpu_VGT2D_planner::
plan(const uint32_t concurrency,    const uint2 dst_dims_v1, 
     const uint8_t datatype_size,   const de::Interpolate_Types intp_type, 
     const uint2 src_dims_v1,       
     de::DH* handle,                const uint64_t min_thread_proc)
{
    decx::cpu_ElementWise2D_planner::plan(concurrency, dst_dims_v1, datatype_size, datatype_size, min_thread_proc);
    
    this->_interpolate_type = intp_type;
    
    const uint32_t addr_mgr_size = (int32_t)this->_interpolate_type ? sizeof(decx::CPUK::VGT_nearest_addr_mgr) : 
                                        sizeof(decx::CPUK::VGT_bilinear_addr_mgr);

    if (decx::alloc::_host_virtual_page_realloc(&this->_addr_mgrs, this->_concurrency * addr_mgr_size)){
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_ALLOCATION,
            ALLOC_FAIL);
        return;
    }

    if (this->_interpolate_type == de::Interpolate_Types::INTERPOLATE_BILINEAR)
    {
        auto* p_addr_mgr = (decx::CPUK::VGT_bilinear_addr_mgr*)this->_addr_mgrs.ptr;
        for (int32_t i = 0; i < this->_concurrency; ++i){
            p_addr_mgr[i].set_src_dims(src_dims_v1);
        }
    }
    else{
        auto* p_addr_mgr = (decx::CPUK::VGT_nearest_addr_mgr*)this->_addr_mgrs.ptr;
        for (int32_t i = 0; i < this->_concurrency; ++i){
            p_addr_mgr[i].set_src_dims(src_dims_v1);
        }
    }
}


template <typename _type_in, typename _type_out> void 
decx::cpu_VGT2D_planner::run(const _type_in* src,          const float2* map, 
                             _type_out* dst,                const uint32_t pitchmap_v1,    
                             const uint32_t pitchdst_v1,    decx::utils::_thr_1D* t1D)
{
    uint64_t dex_map = 0, dex_dst = 0;
    uint32_t _thr_cnt = 0;

    const uint32_t addr_mgr_size = (int32_t)this->_interpolate_type ? sizeof(decx::CPUK::VGT_nearest_addr_mgr) : 
                                        sizeof(decx::CPUK::VGT_bilinear_addr_mgr);

    uint2 selector = decx::VGT2D_kernel_selector<_type_in, _type_out>(this->_interpolate_type);
    auto* exec_ptr = (decx::CPUK::VGT2D_executor<_type_in, _type_out>*)decx::VGT2D_exec_LUT[selector.x][selector.y];

    for (int32_t i = 0; i < this->_thread_dist.y; ++i)
    {
        dex_map = pitchmap_v1 * i * this->_fmgr_WH[1].frag_len;
        dex_dst = pitchdst_v1 * i * this->_fmgr_WH[1].frag_len;

        for (int32_t j = 0; j < this->_thread_dist.x; ++j){
            uint2 proc_dims_v = 
                make_uint2(j < this->_thread_dist.x - 1 ? this->_fmgr_WH[0].frag_len : this->_fmgr_WH[0].last_frag_len,
                        i < this->_thread_dist.y - 1 ? this->_fmgr_WH[1].frag_len : this->_fmgr_WH[1].last_frag_len);

            t1D->_async_thread[_thr_cnt] = decx::cpu::register_task_default(exec_ptr, src, map + dex_map, dst + dex_dst, 
                proc_dims_v, this->_pitchsrc_v1, pitchmap_v1, pitchdst_v1, (uint8_t*)this->_addr_mgrs.ptr + _thr_cnt * addr_mgr_size);
            
            dex_map += this->_fmgr_WH[0].frag_len * this->_alignment;
            dex_dst += this->_fmgr_WH[0].frag_len * this->_alignment;
            ++_thr_cnt;
        }
    }
    t1D->__sync_all_threads(make_uint2(0, _thr_cnt));
}

template void decx::cpu_VGT2D_planner::run<float>(const float*, const float2*, float*, const uint32_t,    
    const uint32_t, decx::utils::_thr_1D*);

template void decx::cpu_VGT2D_planner::run<uint8_t>(const uint8_t*, const float2*, uint8_t*, const uint32_t,    
    const uint32_t, decx::utils::_thr_1D*);


void decx::cpu_VGT2D_planner::release(decx::cpu_VGT2D_planner* fake_this)
{
    decx::alloc::_host_virtual_page_dealloc(&fake_this->_addr_mgrs);
}
