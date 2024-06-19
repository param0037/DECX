/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
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

    // Fragment manager for matrix B arrangement
    decx::utils::frag_manager _fmgr_WH_B[2];
    decx::utils::frag_manager _fmgr_WH_dst[2];


    void _CRSR_ _plan_for_B_arrangement(de::DH* handle);


    void _plan_for_exectutors();

public:
    cpu_GEMM_planner() {}


    void _CRSR_ plan(const uint32_t concurrency, const decx::_matrix_layout* layout_A, 
        const decx::_matrix_layout* layout_B, de::DH* handle);


    void Run(decx::_Matrix* A, decx::_Matrix* B, decx::_Matrix* dst, decx::utils::_thread_arrange_2D* t2D);

    void Run(decx::_Matrix* A, decx::_Matrix* B, decx::_Matrix* C, decx::_Matrix* dst, decx::utils::_thread_arrange_2D* t2D);


    uint2 GetThreadDist_B() const;
    uint2 GetThreadDist_dst() const;


    static void Release(decx::blas::cpu_GEMM_planner<_data_type>* _fake_this);
};


struct decx::blas::GEMM_blocking_config
{
    decx::utils::frag_manager _fmgr_L;
    decx::utils::frag_manager _fmgr_W;
    decx::utils::frag_manager _fmgr_H;
};


namespace decx
{
    namespace blas {
        extern decx::ResourceHandle g_cpu_GEMM_fp32_planner;
    }
}

#endif
