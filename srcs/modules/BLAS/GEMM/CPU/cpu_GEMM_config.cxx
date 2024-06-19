/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#include "cpu_GEMM_config.h"


//#define BL 2
//#define BW 16
//#define BH 32


#define BL 2
#define BW 16
#define BH 32


template <typename _data_type> void _CRSR_
decx::blas::cpu_GEMM_planner<_data_type>::_plan_for_B_arrangement(de::DH* handle)
{
    constexpr uint32_t _alignment = 32 / sizeof(_data_type);

    // Plan the thread distribution
    decx::utils::thread2D_arrangement_advisor(&this->_thread_dist_B, this->_concurrency,
        make_uint2(this->_layout_B->height, this->_layout_B->width));

    this->_arranged_B._dims = make_uint2(this->_layout_B->height * _alignment,
        decx::utils::ceil<uint32_t>(this->_layout_B->pitch, _alignment));

    if (decx::alloc::_host_virtual_page_malloc(&this->_arranged_B._ptr, this->_arranged_B._dims.x * this->_arranged_B._dims.y * sizeof(_data_type))) {
        return;
    }

    if (decx::alloc::_host_virtual_page_malloc(&this->_thread_config, this->_concurrency * sizeof(decx::blas::GEMM_blocking_config))) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_ALLOCATION,
            ALLOC_FAIL);
        return;
    }

    // plan for fragment manager for matrix B arrangement
    decx::utils::frag_manager_gen_Nx(this->_fmgr_WH_B, decx::utils::ceil<uint32_t>(this->_layout_B->pitch, _alignment), this->_thread_dist_B.x, 2);
    decx::utils::frag_manager_gen_Nx(this->_fmgr_WH_B + 1, this->_layout_B->height, this->_thread_dist_B.y, 16);
}

template void _CRSR_ decx::blas::cpu_GEMM_planner<float>::_plan_for_B_arrangement(de::DH*);
template void _CRSR_ decx::blas::cpu_GEMM_planner<double>::_plan_for_B_arrangement(de::DH*);


template <typename _data_type> void
decx::blas::cpu_GEMM_planner<_data_type>::_plan_for_exectutors()
{
    constexpr uint32_t _alignment = 32 / sizeof(_data_type);

    // Plan the thread distribution
    decx::utils::thread2D_arrangement_advisor(&this->_thread_dist_dst, this->_concurrency,
        this->_proc_dims_v1);

    // plan for fragment manager for GEMM
    decx::utils::frag_manager_gen(this->_fmgr_WH_dst, decx::utils::ceil<uint32_t>(this->_proc_dims_v1.x, _alignment), this->_thread_dist_dst.x);
    decx::utils::frag_manager_gen(this->_fmgr_WH_dst + 1, this->_proc_dims_v1.y, this->_thread_dist_dst.y);

    for (uint32_t i = 0; i < this->_thread_dist_dst.y; ++i) 
    {
        uint2 proc_dims_v = make_uint2(this->_fmgr_WH_dst[0].frag_len, 
                                        i < this->_thread_dist_dst.y - 1 ? this->_fmgr_WH_dst[1].frag_len : this->_fmgr_WH_dst[1].last_frag_len);
        for (uint32_t j = 0; j < this->_thread_dist_dst.x - 1; ++j)
        {
            auto* conf_ptr = &((decx::blas::GEMM_blocking_config*)this->_thread_config.ptr)[this->_thread_dist_dst.x * i + j];

            decx::utils::frag_manager_gen_from_fragLen(&conf_ptr->_fmgr_H, proc_dims_v.y, BH);
            decx::utils::frag_manager_gen_from_fragLen(&conf_ptr->_fmgr_W, proc_dims_v.x, BL);
            decx::utils::frag_manager_gen_from_fragLen(&conf_ptr->_fmgr_L, this->_layout_A->pitch, BW);
        }
        proc_dims_v.x = this->_fmgr_WH_dst[0].last_frag_len;

        auto* conf_ptr = &((decx::blas::GEMM_blocking_config*)this->_thread_config.ptr)[this->_thread_dist_dst.x * (i + 1) - 1];

        decx::utils::frag_manager_gen_from_fragLen(&conf_ptr->_fmgr_H, proc_dims_v.y, BH);
        decx::utils::frag_manager_gen_from_fragLen(&conf_ptr->_fmgr_W, proc_dims_v.x, BL);
        decx::utils::frag_manager_gen_from_fragLen(&conf_ptr->_fmgr_L, this->_layout_A->pitch, BW);
    }
}

template void decx::blas::cpu_GEMM_planner<float>::_plan_for_exectutors();
template void decx::blas::cpu_GEMM_planner<double>::_plan_for_exectutors();


template <typename _data_type> void _CRSR_ 
decx::blas::cpu_GEMM_planner<_data_type>::plan(const uint32_t concurrency, 
                                          const decx::_matrix_layout* layout_A,
                                          const decx::_matrix_layout* layout_B, 
                                          de::DH* handle)
{
    this->_concurrency = concurrency;
    this->_layout_A = layout_A;
    this->_layout_B = layout_B;

    this->_proc_dims_v1 = make_uint2(this->_layout_B->width, this->_layout_A->height);

    this->_plan_for_B_arrangement(handle);
    if (handle->error_type != decx::DECX_error_types::DECX_SUCCESS) {
        return;
    }

    this->_plan_for_exectutors();
}

template void _CRSR_ decx::blas::cpu_GEMM_planner<float>::plan(const uint32_t, const decx::_matrix_layout*,
    const decx::_matrix_layout*, de::DH*);

template void _CRSR_ decx::blas::cpu_GEMM_planner<double>::plan(const uint32_t, const decx::_matrix_layout*,
    const decx::_matrix_layout*, de::DH*);


template <typename _data_type>
bool decx::blas::cpu_GEMM_planner<_data_type>::Changed(const uint32_t concurrency, 
                                                       const decx::_matrix_layout* layout_A,
                                                       const decx::_matrix_layout* layout_B) const
{
    uint32_t _conc_matched = this->_concurrency ^ concurrency;
    uint32_t _Adims_matched = 1, _Bdims_matched = 1;
    if (this->_layout_A) {
        _Adims_matched = this->_layout_A->height ^ layout_A->height;
        _Adims_matched ^= (this->_layout_A->width ^ layout_A->width);
    }
    if (this->_layout_B) {
        _Bdims_matched = this->_layout_B->height ^ layout_B->height;
        _Bdims_matched ^= (this->_layout_B->width ^ layout_B->width);
    }
    return _conc_matched ^ _Adims_matched ^ _Bdims_matched;
}

template bool decx::blas::cpu_GEMM_planner<float>::Changed(const uint32_t, const decx::_matrix_layout*,
    const decx::_matrix_layout*) const;

template bool decx::blas::cpu_GEMM_planner<double>::Changed(const uint32_t, const decx::_matrix_layout*,
    const decx::_matrix_layout*) const;


template <typename _data_type>
void _CRSR_ decx::blas::cpu_GEMM_planner<_data_type>::Validate(de::DH* handle,
                                                               const decx::_matrix_layout* layout_A, 
                                                               const decx::_matrix_layout* layout_B,
                                                               const decx::_matrix_layout* layout_C)
{
    if (layout_A->width != layout_B->height) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_DimsNotMatching,
            "The width of matrix A should be consistent to the height of matrix B");
        return;
    }
    if (layout_C) {
        if (layout_C->height != layout_A->height || layout_C->width != layout_B->width) {
            decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_DimsNotMatching,
                "The height and width of the matrix C should be identical to that of matrix A and matrix B, respectively");
            return;
        }
    }
}

template void decx::blas::cpu_GEMM_planner<float>::Validate(de::DH*, const decx::_matrix_layout*,
    const decx::_matrix_layout*, const decx::_matrix_layout*);

template void decx::blas::cpu_GEMM_planner<double>::Validate(de::DH*, const decx::_matrix_layout*,
    const decx::_matrix_layout*, const decx::_matrix_layout*);



template <typename _data_type>
uint2 decx::blas::cpu_GEMM_planner<_data_type>::GetThreadDist_B() const
{
    return this->_thread_dist_B;
}

template uint2 decx::blas::cpu_GEMM_planner<float>::GetThreadDist_B() const;
template uint2 decx::blas::cpu_GEMM_planner<double>::GetThreadDist_B() const;


template <typename _data_type>
uint2 decx::blas::cpu_GEMM_planner<_data_type>::GetThreadDist_dst() const
{
    return this->_thread_dist_dst;
}

template uint2 decx::blas::cpu_GEMM_planner<float>::GetThreadDist_dst() const;
template uint2 decx::blas::cpu_GEMM_planner<double>::GetThreadDist_dst() const;


template <typename _data_type>
void decx::blas::cpu_GEMM_planner<_data_type>::Release(decx::blas::cpu_GEMM_planner<_data_type>* _fake_this)
{
    decx::alloc::_host_virtual_page_dealloc(&_fake_this->_arranged_B._ptr);
    decx::alloc::_host_virtual_page_dealloc(&_fake_this->_thread_config);
}

template void decx::blas::cpu_GEMM_planner<float>::Release(decx::blas::cpu_GEMM_planner<float>* _fake_this);
template void decx::blas::cpu_GEMM_planner<double>::Release(decx::blas::cpu_GEMM_planner<double>* _fake_this);
