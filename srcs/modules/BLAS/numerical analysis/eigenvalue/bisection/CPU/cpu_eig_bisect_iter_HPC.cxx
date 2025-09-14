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


template <typename _data_type>
void decx::blas::cpu_eig_bisect_count_interval<_data_type>::init(const uint64_t max_intrv_num)
{
    this->_intrv_buf.Allocate(max_intrv_num * 2 * sizeof(T_interval), PAGABLE);

    this->_mid_arr_buf.Allocate(max_intrv_num * 2 * sizeof(_data_type), PAGABLE);
    
    this->_count_buffer.Allocate(1200 * sizeof(int32_t), PAGABLE);
}

template void decx::blas::cpu_eig_bisect_count_interval<float>::init(const uint64_t);


template <typename _data_type>
void decx::blas::cpu_eig_bisect_count_interval<_data_type>::set_count_num(const uint64_t proc_len)
{
    if (this->_total != proc_len) {
        this->_total = proc_len;
        this->_total_v = decx::utils::idiv_ceil<uint64_t>(this->_total, this->_alignment);

        if (this->_total / this->_concurrency > this->_min_thread_proc){
            decx::utils::frag_manager_gen_Nx(&this->_fmgr, this->_total, this->_concurrency, this->_alignment);
        }
        else{
            const uint32_t real_conc = decx::utils::idiv_ceil<uint64_t>(this->_total, this->_min_thread_proc);
            decx::utils::frag_manager_gen_Nx(&this->_fmgr, this->_total, real_conc, this->_alignment);
        }
    }
}

template void decx::blas::cpu_eig_bisect_count_interval<float>::set_count_num(const uint64_t);


template <>
void decx::blas::cpu_eig_bisect_count_interval<float>::count_intervals(uint32_t* p_num, decx::utils::ThreadArrange1D* t1D)
{
    const uint32_t frag_len = this->_fmgr.GetFragLen();

    this->Caller(update_intrv,
        EW_SLOT_ID_MONOTONIC(0),
        decx::TArg_still<decx::blas::cpu_eig_bisect_count_interval<float>*>    (this),
        decx::TArg_var<const T_interval*>  ([&](const int32_t i){return this->_p_interval + i * frag_len;}),
        decx::TArg_var<T_interval*>  ([&](const int32_t i){return this->_intrv_buf.GetRawPtr() + i * frag_len * 2;}),
        decx::TArg_var<const float*>  ([&](const int32_t i){return this->_p_midps + i * frag_len;}),
        decx::TArg_var<float*>        ([&](const int32_t i){return this->_mid_arr_buf.GetRawPtr() + i * frag_len * 2;}),
        decx::TArg_still<uint32_t>    (this->_N),
        decx::TArg_var<uint32_t>      ([&](const int32_t i){return this->get_fmgr()->GetFragLenById(i);}),
        decx::TArg_var<uint32_t*>      ([&](const int32_t i){return this->_count_buffer.GetRawPtr() + i;})
    );

    uint32_t _valid_num = 0;
    uint32_t* _count_res_buf = this->_count_buffer.GetRawPtr<uint32_t>();

    const auto* _loc_pread_intrv = this->_intrv_buf.GetRawPtrConst();
    auto* _loc_pwrite_intrv = this->_p_interval;

    const auto* _loc_pread_midp = this->_mid_arr_buf.GetRawPtrConst();
    auto* _loc_pwrite_midp = this->_p_midps;

    for (int i = 0; i < this->_fmgr.frag_num; ++i){
        const uint32_t _this_count = _count_res_buf[i];
        _valid_num += _this_count;

        memcpy(_loc_pwrite_intrv, _loc_pread_intrv, sizeof(T_interval) * _this_count);
        memcpy(_loc_pwrite_midp, _loc_pread_midp, sizeof(float) * _this_count);

        _loc_pread_intrv += this->get_proc_len_by_id(i) * 2;
        _loc_pwrite_intrv += _this_count;
        _loc_pread_midp += this->get_proc_len_by_id(i) * 2;
        _loc_pwrite_midp += _this_count;
    }
    p_num[0] = _valid_num;
}


template<> _THREAD_FUNCTION_ void 
decx::blas::cpu_eig_bisect_count_interval<float>::
update_intrv(decx::blas::cpu_eig_bisect_count_interval<float>* _fake_this,
             const T_interval*        intrv_outer,
             T_interval*        intrv_buf, 
             const float*       midps_src, 
             float*             midps_dst,
             const uint32_t     N, 
             const uint32_t     proc_len,
             uint32_t*           p_valid_num)
{
    int32_t _next_valid_num = 0;

    for (int32_t i = 0; i < decx::utils::idiv_ceil<uint32_t>(proc_len, 8); ++i)
    {
        decx::utils::simd::xmm256_reg _count_mid;
        _count_mid._vi = decx::blas::CPUK::count_v8_eigv_fp32(
            _fake_this->_p_diag, _fake_this->_p_off_diag, NULL, midps_src + i * 8, N);

#pragma unroll
        for (int k = 0; k < 8; ++k)
        {
        if (i * 8 + k < proc_len)
        {
            const auto* p_current_interval = intrv_outer + i * 8 + k;
            
            if (p_current_interval->is_valid()) {
                const uint32_t count_mid1 = _count_mid._arrui[k];

                const float l = p_current_interval->_l;
                const float u = p_current_interval->_u;

                const float _mid = (l + u) / 2.f;

                if (count_mid1 - p_current_interval->_count_l > 0) {
                    (intrv_buf + _next_valid_num)->set(l, _mid);
                    (intrv_buf + _next_valid_num)->_count_l = p_current_interval->_count_l;
                    (intrv_buf + _next_valid_num)->_count_u = count_mid1;
                    midps_dst[_next_valid_num] = (_mid + l) / 2.f;
                    ++_next_valid_num;
                }
                
                if (p_current_interval->_count_u - count_mid1 > 0){
                    (intrv_buf + _next_valid_num)->set(_mid, u);
                    (intrv_buf + _next_valid_num)->_count_u = p_current_interval->_count_u;
                    (intrv_buf + _next_valid_num)->_count_l = count_mid1;
                    midps_dst[_next_valid_num] = (u + _mid) / 2.f;
                    ++_next_valid_num;
                }
            }
        }
        }
    }
    *p_valid_num = _next_valid_num;
}


template <typename _data_type> void 
decx::blas::cpu_eig_bisect_iter_HPC<_data_type>::
init(const _data_type*  p_diag,         const _data_type* p_off_diag, 
     const uint32_t N,                  const _data_type L, 
     const _data_type U)
{
    int32_t rval = 0;
    this->_current_interval_gap = U - L;

    this->_max_interval_num = (uint32_t)ceilf((float)this->_current_interval_gap / (float)this->_max_err);

    // Allocate the interval stack
    rval |= this->_interval_stack.Allocate(this->_max_interval_num * sizeof(decx::blas::eig_bisect_interval<_data_type>), PAGABLE);
    
    // Allocate the count buffer
    const uint32_t max_mid_count_num = decx::utils::ialign_up<uint32_t>(this->_max_interval_num - 1, 8);
    rval |= this->_mid_points.Reallocate(this->_max_interval_num * sizeof(_data_type), true, true);

    auto* p_1st_interval = this->_interval_stack.GetRawPtr();
    p_1st_interval[0].set(L, U);
    p_1st_interval[0].count_violent(p_diag, p_off_diag, N);
    this->_mid_points[0] = (U + L) / 2.f;

    this->_current_stack_vaild_num = 1;

    this->_count_intervals = decx::blas::cpu_eig_bisect_count_interval<_data_type>(
        p_diag, p_off_diag, this->_mid_points.GetRawPtr(), p_1st_interval, N);
    this->_count_intervals.plan(DecxGetPermitConcurrency(), 1, sizeof(_data_type), sizeof(_data_type), 1);
    this->_count_intervals.init(1024);
}

template void decx::blas::cpu_eig_bisect_iter_HPC<float>::init(const float*, const float*, const uint32_t, const float, const float);


template <>
void decx::blas::cpu_eig_bisect_iter_HPC<float>::iter(const float* p_diag, const float* p_off_diag, const uint32_t N)
{
    decx::utils::Thr1D t1D(DecxGetPermitConcurrency());

    auto* p_1st_interval = this->_interval_stack.GetRawPtr();
    
    //if (p_1st_interval->is_valid())
    // {
    while(this->_current_interval_gap > this->_max_err)
    {
        // printf("now count : %d\n", this->_current_stack_vaild_num);
        // printf("[%d, %d]\n", this->_count_intervals.get_fmgr()->frag_len, this->_count_intervals.get_fmgr()->frag_num);
        this->_count_intervals.set_count_num(this->_current_stack_vaild_num);
        this->_count_intervals.count_intervals(&this->_current_stack_vaild_num, &t1D);

        this->_current_interval_gap /= 2.f;
    }
    //}

    // Final check to filter out the invalid interval(s)
    auto* p_read = this->_interval_stack.GetRawPtr();
    uint32_t STG_dex = 0;
    for (int i = 0; i < this->_current_stack_vaild_num; ++i){
        if (p_read[i].is_valid()){
            // _mm_storeu_ps((float*)(p_write + STG_dex), _mm_loadu_ps((float*)(p_read + i)));
            ++STG_dex;
        }
    }
    
    this->_eig_count_actual = STG_dex;
    // this->_double_buffer.UpdateStatus();
}



template <typename _data_type>
const decx::blas::eig_bisect_interval<_data_type>* 
decx::blas::cpu_eig_bisect_iter_HPC<_data_type>::get_valid_intervals_array()
{
    return this->_interval_stack.GetRawPtr();
}

template const decx::blas::eig_bisect_interval<float>* 
decx::blas::cpu_eig_bisect_iter_HPC<float>::get_valid_intervals_array();
