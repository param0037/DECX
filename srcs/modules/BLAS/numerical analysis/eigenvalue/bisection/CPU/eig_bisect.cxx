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

#include "eig_bisect.h"
#include <thread_management/thread_pool.h>
#include "eig_utils_kernels.h"


template <typename _data_type> void 
decx::blas::cpu_eig_bisection<_data_type>::Init(const uint32_t conc, 
                                                const decx::_matrix_layout* layout, 
                                                const _data_type max_err, 
                                                de::DH* handle)
{
#if defined(__x86_64__) || defined(__i386__)
    constexpr uint32_t _align_byte = 32;
#endif
#if defined(__aarch64__) || defined(__arm__)
    constexpr uint32_t _align_byte = 16;
#endif
    this->_alignment = _align_byte / sizeof(_data_type);

    this->_layout = *layout;
    this->_concurrency = conc;
    // this->_max_err = max_err;
    this->_iter_scheduler.set_max_err(max_err);

    if (this->_layout.width != this->_layout.width){
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_DimsNotMatching,
            "The input should be a square matrix");
        return;
    }

    this->_aligned_N = decx::utils::align<uint32_t>(this->_layout.width, this->_alignment);
    if (decx::alloc::_host_virtual_page_realloc(&this->_diag, this->_aligned_N * sizeof(_data_type))){
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_ALLOCATION,
            ALLOC_FAIL);
        return;
    }
    // Puls one to fit the process of finding Gerschgorin boundary.
    if (decx::alloc::_host_virtual_page_realloc(&this->_off_diag, (this->_aligned_N + 1) * sizeof(_data_type))){
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_ALLOCATION,
            ALLOC_FAIL);
        return;
    }
    // Allocate the shared memory.
    if (decx::alloc::_host_virtual_page_realloc(&this->_shared_mem, 4 * this->_concurrency * sizeof(_data_type))) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_ALLOCATION,
            ALLOC_FAIL);
        return;
    }

    // Initialize the diagonal extractor
    this->_diag_extractor.plan(this->_concurrency, this->_layout.width, sizeof(_data_type), sizeof(_data_type));

    this->_Gersch_bound_founder.plan(this->_concurrency, this->_layout.width, 
        sizeof(_data_type), sizeof(_data_type), 1);
    this->_Gersch_bound_founder.alloc_shared_mem((this->_concurrency + 1) * sizeof(_data_type) * 200, handle);
}

template void decx::blas::cpu_eig_bisection<float>::Init(const uint32_t, const decx::_matrix_layout*, const float, de::DH*);



template <typename _data_type>
void decx::blas::cpu_eig_bisection<_data_type>::extract_diagonal(const _data_type* src, decx::utils::_thread_arrange_1D* t1D)
{
    using diag_extractor = void(const _data_type*, _data_type*, _data_type*, const uint32_t, const uint32_t);

    const _data_type* loc_ptr = src;
    _data_type* p_diag = this->_diag.ptr;
    _data_type* p_off_diag = this->_off_diag.ptr + 1;       // Shift one entry since b_0 = 0

    const uint32_t& frag_num = this->_diag_extractor.get_fmgr()->frag_num;
    const uint32_t& frag_len = this->_diag_extractor.get_fmgr()->frag_len;

    for (int32_t i = 0; i < frag_num; ++i)
    {
        diag_extractor* f = i < frag_num - 1 ? 
            decx::blas::CPUK::extract_diag_fp32<false> :
            decx::blas::CPUK::extract_diag_fp32<true>;

        const uint32_t proc_len = this->_diag_extractor.get_proc_len_by_id(i);
        
        t1D->_async_thread[i] = decx::cpu::register_task_default(f, loc_ptr, p_diag, p_off_diag, proc_len, this->_layout.pitch);

        loc_ptr += (this->_layout.pitch + 1) * frag_len;
        p_diag += frag_len;
        p_off_diag += frag_len;
    }

    t1D->__sync_all_threads(make_uint2(0, frag_num));
}

template void decx::blas::cpu_eig_bisection<float>::extract_diagonal(const float*, decx::utils::_thread_arrange_1D*);


template <typename _data_type>
void decx::blas::cpu_eig_bisection<_data_type>::calc_Gerschgorin_bound(decx::utils::_thread_arrange_1D* t1D)
{
    const decx::utils::frag_manager* p_dist = this->_Gersch_bound_founder.get_distribution();
    const uint32_t& frag_num = p_dist->frag_num;

    // Reduction between threads
    _data_type* u_ptr = this->_Gersch_bound_founder.get_shared_mem<_data_type>();
    _data_type* l_ptr = u_ptr + frag_num;

    this->_Gersch_bound_founder.caller(decx::blas::CPUK::Gerschgorin_bound_fp32,
        t1D,
        decx::TArg_var<const float*>([this, p_dist](const int32_t i){return this->_diag.ptr + i * p_dist->get_frag_len();}),
        decx::TArg_var<const float*>([this, p_dist](const int32_t i){return this->_off_diag.ptr + i * p_dist->get_frag_len();}),
        decx::TArg_var<float*>      ([u_ptr](const int32_t i)->float*{return u_ptr + i;}),
        decx::TArg_var<float*>      ([l_ptr](const int32_t i)->float*{return l_ptr + i;}),
        decx::TArg_var<uint32_t>    ([p_dist](const int32_t i){return p_dist->get_frag_len_by_id(i);}));
    
    this->_Gerschgorin_L = l_ptr[0];
    this->_Gerschgorin_U = u_ptr[0];
    for (int32_t i = 1; i < frag_num; ++i){
        _data_type val = l_ptr[i];
        if (this->_Gerschgorin_L > val) this->_Gerschgorin_L = val;
        val = u_ptr[i];
        if (this->_Gerschgorin_U < val) this->_Gerschgorin_U = val;
    }
}

template void decx::blas::cpu_eig_bisection<float>::calc_Gerschgorin_bound(decx::utils::_thread_arrange_1D*);


template <typename _data_type>
void decx::blas::cpu_eig_bisection<_data_type>::plan(const decx::_Matrix* mat, decx::utils::_thread_arrange_1D* t1D,
    de::DH* handle)
{
    // Extract diagonal and off-diagonal elements
    this->extract_diagonal((_data_type*)mat->Mat.ptr, t1D);
    // Caclulate the Gerschgorin boundary (the boundary including all possible eigenvalues)
    this->calc_Gerschgorin_bound(t1D);

    this->_iter_scheduler.init((_data_type*)this->_diag.ptr, 
                               (_data_type*)this->_off_diag.ptr, 
                               this->_layout.width,
                               this->_Gerschgorin_L, 
                               this->_Gerschgorin_U, 
                               handle);
    Check_Runtime_Error(handle);
}

template void decx::blas::cpu_eig_bisection<float>::plan(const decx::_Matrix*, decx::utils::_thread_arrange_1D*, de::DH*);


template <>
void decx::blas::eig_bisect_interval<float>::count_violent(const float* diag, const float* off_diag, const uint32_t N)
{
    decx::blas::CPUK::count_eigv_fp32(diag, off_diag, &this->_count_l, this->_l, N);
    decx::blas::CPUK::count_eigv_fp32(diag, off_diag, &this->_count_u, this->_u, N);
}


template <typename _data_type>
void decx::blas::cpu_eig_bisection<_data_type>::iter_bisection()
{
    this->_iter_scheduler.iter((_data_type*)this->_diag.ptr, (_data_type*)this->_off_diag.ptr, this->_layout.width);
}

template void decx::blas::cpu_eig_bisection<float>::iter_bisection();
