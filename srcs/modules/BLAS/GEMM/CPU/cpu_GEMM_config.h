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


#ifndef _CPU_GEMM_CONFIG_H_
#define _CPU_GEMM_CONFIG_H_

#include "../GEMM_utils.h"
#include "../../../core/resources_manager/decx_resource.h"

namespace decx {
    namespace blas {
        template <typename _data_type>
        class cpu_GEMM_planner;


        struct GEMM_blocking_config;
    }
}


template <typename _data_type>
class decx::blas::cpu_GEMM_planner
{
private:
    uint32_t _concurrency;
    uint2 _proc_dims_v1;

    uint2 _thread_dist_B, _thread_dist_dst;

    const decx::_matrix_layout* _layout_A, 
                               *_layout_B, 
                               *_layout_C;      // = NULL if not applicable
    decx::PtrInfo<decx::blas::GEMM_blocking_config> _thread_config;

    decx::Ptr2D_Info<void> _arranged_B;

    /**
    * Fragment manager for matrix B arrangement
    */
    decx::utils::frag_manager _fmgr_WH_B[2];
    /**
    * ::_fmgr_WH_dst[0] have to be configured by frag_manager_gen_Nx() with N = 2 for multi-threading cases.
    * Since the inner thread block width should be ALIGNED TO _alignment x 2.
    */
    decx::utils::frag_manager _fmgr_WH_dst[2];


    void _CRSR_ _plan_for_B_arrangement(de::DH* handle);


    void _plan_for_exectutors(const bool _cplxf);

public:
    cpu_GEMM_planner() {}


    bool Changed(const uint32_t concurrency, const decx::_matrix_layout* layout_A, 
        const decx::_matrix_layout* layout_B) const;


    void _CRSR_ plan(const uint32_t concurrency, const decx::_matrix_layout* layout_A,
        const decx::_matrix_layout* layout_B, de::DH* handle, const bool _cplxf = false);


    static void _CRSR_ Validate(de::DH* handle, const decx::_matrix_layout* layout_A, const decx::_matrix_layout* layout_B,
        const decx::_matrix_layout* layout_C = NULL);

    template <bool _cplxf>
    void Run(decx::_Matrix* A, decx::_Matrix* B, decx::_Matrix* dst, decx::utils::_thread_arrange_2D* t2D);

    template <bool _cplxf>
    void Run(decx::_Matrix* A, decx::_Matrix* B, decx::_Matrix* C, decx::_Matrix* dst, decx::utils::_thread_arrange_2D* t2D);


    uint2 GetThreadDist_B() const;
    uint2 GetThreadDist_dst() const;


    static void Release(decx::blas::cpu_GEMM_planner<_data_type>* _fake_this);
};


struct decx::blas::GEMM_blocking_config
{
    decx::utils::frag_manager _fmgr_L;
    decx::utils::frag_manager _fmgr_W;  // Aligned to original alignment
    decx::utils::frag_manager _fmgr_H;
};


namespace decx
{
    namespace blas {
        extern decx::ResourceHandle g_cpu_GEMM_fp32_planner;
        extern decx::ResourceHandle g_cpu_GEMM_64b_planner;
        extern decx::ResourceHandle g_cpu_GEMM_cplxd_planner;
    }
}

#endif
