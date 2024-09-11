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


namespace decx
{
    static void* VGT2D_exec_LUT[] = {
        (void*)decx::CPUK::gather2D_fp32_exec_bilinear, // bilinear
        NULL,   // nearest
    };
}


template <typename _data_type>
void* decx::cpu_VGT2D_planner::find_exec_ptr() const
{
    uint32_t idx = 0;

#if __cplusplus >= 201703L
    if constexpr (std::is_same_v<float, _data_type>){
        idx = 0 + this->_interpolate_type;
    }
#elif __cplusplus >= 201103L
    if (std::is_same<float, _data_type>::value){
        idx = 0 + this->_interpolate_type;
    }
#endif

    return decx::VGT2D_exec_LUT[idx];
}

template void* decx::cpu_VGT2D_planner::find_exec_ptr<float>() const;


_CRSR_ void decx::cpu_VGT2D_planner::
plan(const uint32_t concurrency,    const uint2 dst_dims_v1, 
     const uint8_t datatype_size,   const decx::Interpolate_Types intp_type, 
     const uint2 src_dims_v1,       
     de::DH* handle,                const uint64_t min_thread_proc)
{
    decx::cpu_ElementWise2D_planner::plan(concurrency, dst_dims_v1, datatype_size, datatype_size, min_thread_proc);
    
    this->_interpolate_type = intp_type;
    
    if (decx::alloc::_host_virtual_page_malloc(&this->_addr_mgrs, this->_concurrency * sizeof(decx::CPUK::VGT_addr_mgr))){
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_ALLOCATION,
            ALLOC_FAIL);
        return;
    }

    for (int32_t i = 0; i < this->_concurrency; ++i){
        this->_addr_mgrs.ptr[i].set_src_dims(src_dims_v1);
    }
}


template <typename data_type> void 
decx::cpu_VGT2D_planner::run(const data_type* src,          const float2* map, 
                             data_type* dst,                const uint32_t pitchmap_v1,    
                             const uint32_t pitchdst_v1,    decx::utils::_thr_1D* t1D)
{
    uint64_t dex_map = 0, dex_dst = 0;
    uint32_t _thr_cnt = 0;

    auto* exec_ptr = (decx::CPUK::VGT2D_executor<data_type>*)this->find_exec_ptr<data_type>();

    for (int32_t i = 0; i < this->_thread_dist.y; ++i)
    {
        dex_map = pitchmap_v1 * i * this->_fmgr_WH[1].frag_len;
        dex_dst = pitchdst_v1 * i * this->_fmgr_WH[1].frag_len;

        for (int32_t j = 0; j < this->_thread_dist.x; ++j){
            uint2 proc_dims_v = 
                make_uint2(j < this->_thread_dist.x - 1 ? this->_fmgr_WH[0].frag_len : this->_fmgr_WH[0].last_frag_len,
                        i < this->_thread_dist.y - 1 ? this->_fmgr_WH[1].frag_len : this->_fmgr_WH[1].last_frag_len);

            t1D->_async_thread[_thr_cnt] = decx::cpu::register_task_default(exec_ptr, src, map + dex_map, dst + dex_dst, 
                proc_dims_v, this->_pitchsrc_v1, pitchmap_v1, pitchdst_v1, this->_addr_mgrs.ptr + _thr_cnt);
            
            dex_map += this->_fmgr_WH[0].frag_len * this->_alignment;
            dex_dst += this->_fmgr_WH[0].frag_len * this->_alignment;
            ++_thr_cnt;
        }
    }
    t1D->__sync_all_threads(make_uint2(0, _thr_cnt));
}

template void decx::cpu_VGT2D_planner::run<float>(const float*, const float2*, float*, const uint32_t,    
    const uint32_t, decx::utils::_thr_1D*);


void decx::cpu_VGT2D_planner::release(decx::cpu_VGT2D_planner* fake_this)
{
    decx::alloc::_host_virtual_page_dealloc(&fake_this->_addr_mgrs);
}
