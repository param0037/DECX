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

#ifndef _CPU_FILTER2D_PLANNER_H_
#define _CPU_FILTER2D_PLANNER_H_

#include "../../../../../core/thread_management/thread_pool.h"
#include "../../../../../../common/FMGR/fragment_arrangment.h"
#include "../../../../../core/thread_management/thread_arrange.h"
#include "../../../../../../common/Classes/Matrix.h"
#include "../../../../../../common/Basic_process/extension/extend_flags.h"
#include "../../../../../core/resources_manager/decx_resource.h"


namespace decx {
namespace dsp{
    template <typename _data_type>
    class cpu_Filter2D_planner;


    extern decx::ResourceHandle g_cpu_filter2D_fp32;
    extern decx::ResourceHandle g_cpu_filter2D_64b;
}
}


template <typename _data_type>
class decx::dsp::cpu_Filter2D_planner
{
private:
    decx::_matrix_layout _layout_src, _layout_kernel, _layout_dst;

    decx::utils::_blocking2D_fmgrs _thread_blocking_conf;
    decx::PtrInfo<decx::utils::_blocking2D_fmgrs> _blocking_confs;

    decx::Ptr2D_Info<void> _ext_src;

    de::extend_label _padding_method;

    uint2 _conv_dims_v1;

    uint32_t _concurrency;
    uint2 _thread_dist;

    void filter2D_NB_fp32(decx::_Matrix* src, decx::_Matrix* kernel, decx::_Matrix* dst, decx::utils::_thr_2D* t2D);
    void filter2D_B_fp32(decx::_Matrix* src, decx::_Matrix* kernel, decx::_Matrix* dst, decx::utils::_thr_2D* t2D);

    template <bool _cplxf>
    void filter2D_NB_64b(decx::_Matrix* src, decx::_Matrix* kernel, decx::_Matrix* dst, decx::utils::_thr_2D* t2D);
    template <bool _cplxf>
    void filter2D_B_64b(decx::_Matrix* src, decx::_Matrix* kernel, decx::_Matrix* dst, decx::utils::_thr_2D* t2D);

public:
    cpu_Filter2D_planner() {}


    static uint2 query_dst_dims(const decx::_matrix_layout* src_layout, const decx::_matrix_layout* kernel_layout,
         const de::extend_label padding);


    void _CRSR_ plan(const uint32_t concurrency, const decx::_matrix_layout* src_layout, const decx::_matrix_layout* kernel_layout,
        const decx::_matrix_layout* dst_layout, de::DH* handle, const de::extend_label padding);


    bool changed(const uint32_t conc, decx::_Matrix* src, decx::_Matrix* kernel, decx::_Matrix* dst,
        de::extend_label padding_method) const;

    
    template <bool _cplxf>
    void run(decx::_Matrix* src, decx::_Matrix* kenrel, decx::_Matrix* dst, decx::utils::_thr_2D* t2D);

    
    uint2 get_thread_dist() const;


    static void release(decx::dsp::cpu_Filter2D_planner<_data_type>* _fake_this);
};


#endif