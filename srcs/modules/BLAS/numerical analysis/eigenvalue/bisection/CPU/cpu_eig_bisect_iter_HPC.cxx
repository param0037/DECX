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

#include "cpu_eig_bisect_iter_HPC.h"
#include "eig_utils_kernels.h"


template <typename _data_type> void 
decx::blas::cpu_eig_bisect_iter_HPC<_data_type>::
init(const _data_type*  p_diag, 
     const _data_type*  p_off_diag, 
     const uint32_t     N, 
     const _data_type   L, 
     const _data_type   U, 
     de::DH*            handle)
{
    this->_current_interval_gap = U - L;

    this->_max_interval_num = (uint32_t)ceilf((float)this->_current_interval_gap / (float)this->_max_err);

    // Allocate the interval stack
    if (decx::alloc::_host_virtual_page_malloc_lazy(&this->_interval_stack, 
        2 * this->_max_interval_num * sizeof(decx::blas::eig_bisect_interval<_data_type>))){
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_ALLOCATION,
            ALLOC_FAIL);
        return;
    }
    this->_double_buffer = decx::utils::double_buffer_manager(this->_interval_stack.ptr, this->_interval_stack.ptr + this->_max_interval_num);

    // Allocate the count buffer
    const uint32_t max_mid_count_num = decx::utils::align<uint32_t>(this->_max_interval_num - 1, 8);
    if (decx::alloc::_host_virtual_page_malloc_lazy(&this->_count_buffer, this->_max_interval_num * sizeof(_data_type))){
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_ALLOCATION,
            ALLOC_FAIL);
        return;
    }

    // Initialize the double buffer to be buffer1 leading
    this->_double_buffer.reset_buffer1_leading();

    auto* p_1st_interval = this->_double_buffer.get_buffer1<decx::blas::eig_bisect_interval<_data_type>>();
    p_1st_interval->set(L, U);
    p_1st_interval->count_violent(p_diag, p_off_diag, N);

    this->_current_stack_vaild_num = 1;
}

template void decx::blas::cpu_eig_bisect_iter_HPC<float>::init(const float*, const float*, const uint32_t, const float, const float, de::DH*);


template <>
void decx::blas::cpu_eig_bisect_iter_HPC<float>::iter(const float* p_diag, const float* p_off_diag, const uint32_t N)
{
    auto* p_1st_interval = this->_double_buffer.get_buffer1<decx::blas::eig_bisect_interval<float>>();
    
    if (p_1st_interval->is_valid())
    {
    while(this->_current_interval_gap > this->_max_err)
    {
        uint32_t last_valid_num = this->_current_stack_vaild_num;
        this->_current_stack_vaild_num = 0;

        auto* p_read = this->_double_buffer.get_leading_ptr<decx::blas::eig_bisect_interval<float>>();
        auto* p_write = this->_double_buffer.get_lagging_ptr<decx::blas::eig_bisect_interval<float>>();
        uint32_t write_dex = 0;

        uint32_t _now_valid_num = 0;
        
        for (int32_t i = 0; i < last_valid_num; ++i)
        {
            const auto* p_current_interval = p_read + i;
            if (p_current_interval->is_valid()) {
                const float l = p_current_interval->_l;
                const float u = p_current_interval->_u;

                const float _mid = (l + u) / 2.f;
                this->_count_buffer.ptr[_now_valid_num] = _mid;

                this->_current_stack_vaild_num += 2;

                (p_write + write_dex)->set(l, _mid);
                (p_write + write_dex)->_count_l = p_current_interval->_count_l;
                ++write_dex;
                (p_write + write_dex)->set(_mid, u);
                (p_write + write_dex)->_count_u = p_current_interval->_count_u;
                ++write_dex;

                ++_now_valid_num;
            }
        }

        // Count for the middle point
        for (int32_t i = 0; i < decx::utils::ceil<uint32_t>(_now_valid_num, 8); ++i)
        {
            decx::utils::simd::xmm256_reg _count_mid;
            _count_mid._vi = decx::blas::CPUK::count_v8_eigv_fp32(p_diag, p_off_diag, NULL, this->_count_buffer.ptr + i * 8, N);
            
            #pragma unroll
            for (int k = 0; k < 8; ++k){
                if (i * 8 + k < _now_valid_num){
                    (p_write + (i * 8 + k) * 2)->_count_u = _count_mid._arrui[k];
                    (p_write + (i * 8 + k) * 2 + 1)->_count_l = _count_mid._arrui[k];
                }
            }
        }

        this->_current_interval_gap /= 2.f;
        this->_double_buffer.update_states();
    }
    }

    // Final check to filter out the invalid interval(s)
    auto* p_read = this->_double_buffer.get_leading_ptr<decx::blas::eig_bisect_interval<float>>();
    auto* p_write = this->_double_buffer.get_lagging_ptr<decx::blas::eig_bisect_interval<float>>();
    uint32_t STG_dex = 0;
    for (int i = 0; i < this->_current_stack_vaild_num; ++i){
        if (p_read[i].is_valid()){
            _mm_storeu_ps((float*)(p_write + STG_dex), _mm_loadu_ps((float*)(p_read + i)));
            ++STG_dex;
        }
    }
    
    this->_eig_count_actual = STG_dex;
    this->_double_buffer.update_states();
}



template <typename _data_type>
const decx::blas::eig_bisect_interval<_data_type>* 
decx::blas::cpu_eig_bisect_iter_HPC<_data_type>::get_valid_intervals_array()
{
    return this->_double_buffer.get_leading_ptr<decx::blas::eig_bisect_interval<_data_type>>();
}

template const decx::blas::eig_bisect_interval<float>* 
decx::blas::cpu_eig_bisect_iter_HPC<float>::get_valid_intervals_array();