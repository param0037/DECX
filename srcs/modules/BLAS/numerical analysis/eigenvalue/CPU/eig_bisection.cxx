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

#include "eigenvalue.h"
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
    this->_max_err = max_err;

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

        const uint32_t proc_len = i < frag_num - 1 ? frag_len : this->_diag_extractor.get_fmgr()->last_frag_len;
        
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
    const uint32_t& frag_num = this->_diag_extractor.get_fmgr()->frag_num;
    const uint32_t& frag_len = this->_diag_extractor.get_fmgr()->frag_len;

    for (int32_t i = 0; i < frag_num; ++i){
        const uint32_t proc_len = i < frag_num - 1 ? frag_len : this->_diag_extractor.get_fmgr()->last_frag_len;

        t1D->_async_thread[i] = decx::cpu::register_task_default(decx::blas::CPUK::Gerschgorin_bound_fp32, 
            this->_diag.ptr + proc_len * i, 
            this->_off_diag.ptr + proc_len * i, 
            this->_shared_mem.ptr + i,
            this->_shared_mem.ptr + i + frag_num,
            proc_len);
    }

    t1D->__sync_all_threads(make_uint2(0, frag_num));

    // Reduction between threads
    _data_type* u_ptr = this->_shared_mem.ptr;
    _data_type* l_ptr = this->_shared_mem.ptr + frag_num;

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

    this->_max_interval_num = (uint32_t)ceilf(
        ((float)this->_Gerschgorin_U - (float)this->_Gerschgorin_L) / (float)this->_max_err);

    if (decx::alloc::_host_virtual_page_malloc_lazy(&this->_stack, 
        2 * this->_max_interval_num * sizeof(decx::blas::cpu_eig_bisection<_data_type>))){
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_ALLOCATION,
            ALLOC_FAIL);
        return;
    }

    this->_double_buffer = decx::utils::double_buffer_manager(this->_stack.ptr, this->_stack.ptr + this->_max_interval_num);
    this->_double_buffer.reset_buffer1_leading();
}

template void decx::blas::cpu_eig_bisection<float>::plan(const decx::_Matrix*, decx::utils::_thread_arrange_1D*, de::DH*);


template <>
void decx::blas::eig_bisect_interval<float>::count_violent(const float* diag, const float* off_diag, const uint32_t N)
{
    decx::blas::CPUK::count_eigv_fp32(diag, off_diag, &this->_count_l, this->_l, N);
    decx::blas::CPUK::count_eigv_fp32(diag, off_diag, &this->_count_u, this->_u, N);
}


template <>
void decx::blas::cpu_eig_bisection<float>::iter_bisection()
{
    float interval_gap = this->_Gerschgorin_U - this->_Gerschgorin_L;
    uint32_t current_stack_vaild_num = 1;

    auto* p_1st_interval = this->_double_buffer.get_buffer1<decx::blas::eig_bisect_interval<float>>();
    p_1st_interval->set(this->_Gerschgorin_L, this->_Gerschgorin_U);
    p_1st_interval->count_violent((float*)this->_diag.ptr, (float*)this->_off_diag.ptr, this->_layout.width);
    
    // printf("p_1st_interval : (%d, %d)\n", p_1st_interval->_count_l, p_1st_interval->_count_u);

    if (p_1st_interval->is_valid())
    {
    while(interval_gap > this->_max_err)
    {
        uint32_t last_valid_num = current_stack_vaild_num;
        current_stack_vaild_num = 0;

        auto* p_read = this->_double_buffer.get_leading_ptr<decx::blas::eig_bisect_interval<float>>();
        auto* p_write = this->_double_buffer.get_lagging_ptr<decx::blas::eig_bisect_interval<float>>();
        uint32_t write_dex = 0;
        
        for (int32_t i = 0; i < last_valid_num; ++i)
        {
            const auto* p_current_interval = p_read + i;
            if (p_current_interval->is_valid()) {
                const float l = p_current_interval->_l;
                const float u = p_current_interval->_u;
                const float _mid = (l + u) / 2.f;

                // Count for the middle point
                uint32_t _count_mid = 0;
                decx::blas::CPUK::count_eigv_fp32((float*)this->_diag.ptr, (float*)this->_off_diag.ptr, &_count_mid, _mid, this->_layout.width);

                (p_write + write_dex)->set(l, _mid);
                (p_write + write_dex)->_count_l = p_current_interval->_count_l;
                (p_write + write_dex)->_count_u = _count_mid;
                ++write_dex;
                
                (p_write + write_dex)->set(_mid, u);
                (p_write + write_dex)->_count_l = _count_mid;
                (p_write + write_dex)->_count_u = p_current_interval->_count_u;
                ++write_dex;
                current_stack_vaild_num += 2;
            }
        }
        interval_gap /= 2.f;
        this->_double_buffer.update_states();
    }
    }
    this->_eig_count_actual = current_stack_vaild_num;
}