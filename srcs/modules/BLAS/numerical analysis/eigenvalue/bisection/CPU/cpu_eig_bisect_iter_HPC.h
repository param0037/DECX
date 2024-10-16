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

#ifndef _CPU_EIG_BISECT_ITER_HPC_H_
#define _CPU_EIG_BISECT_ITER_HPC_H_

#include <basic.h>
#include <thread_management/thread_arrange.h>
#include <double_buffer.h>
#include "../common/eig_bisect_interval.h"
#include <allocators.h>
#include <Element_wise/common/cpu_element_wise_planner.h>
#include <Algorithms/reduce/CPU/cpu_reduce_planner.h>


namespace decx
{
namespace blas{
    template <typename _data_type>
    class cpu_eig_bisect_iter_HPC;


    template <typename _data_type>
    class cpu_eig_bisect_count_interval;
}
}


template <typename _data_type>
class decx::blas::cpu_eig_bisect_count_interval : public decx::cpu_ElementWise1D_planner
{
public:
    using T_interval = decx::blas::eig_bisect_interval<_data_type>;
private:
    const _data_type* _p_diag; 
    const _data_type* _p_off_diag;

    decx::PtrInfo<uint32_t> _count_buffer;

    decx::PtrInfo<_data_type> _mid_arr_buf;
    _data_type* _p_midps;            // Outer ptr of which contents should be updated

    T_interval* _p_interval;            // Outer ptr of which contents should be updated
    decx::PtrInfo<T_interval> _intrv_buf;

    uint32_t _N;

public:
    cpu_eig_bisect_count_interval() {}


    cpu_eig_bisect_count_interval(const _data_type* p_diag, 
                                  const _data_type* p_off_diag,
                                  _data_type* p_midps,
                                  T_interval* p_interval,
                                  const uint32_t N) :
    _p_diag(p_diag),
    _p_off_diag(p_off_diag),
    _N(N),
    _p_midps(p_midps),
    _p_interval(p_interval)
    {}


    void init(const uint64_t max_intrv_num, de::DH* handle);


    void set_count_num(const uint64_t proc_len);


    void set_interval_buffers(decx::blas::eig_bisect_interval<_data_type>* p_read) {
        this->_p_interval = p_read;
    }

    void count_intervals(uint32_t* p_num, decx::utils::_thread_arrange_1D* t1D);

    
    _THREAD_FUNCTION_ static
    void update_intrv(decx::blas::cpu_eig_bisect_count_interval<_data_type>* _fake_this,
        const T_interval* intrv_outer, T_interval* intrv_buf, const _data_type* midps_src, _data_type* midps_dst,
        const uint32_t N, const uint32_t proc_len, uint32_t* p_valid_num);
};

template<> void decx::blas::cpu_eig_bisect_count_interval<float>::update_intrv(decx::blas::cpu_eig_bisect_count_interval<float>* _fake_this,
        const T_interval* intrv_outer, T_interval* intrv_buf, const float* midps_src, float* midps_dst,
        const uint32_t N, const uint32_t proc_len, uint32_t* p_valid_num);


template <typename _data_type>
class decx::blas::cpu_eig_bisect_iter_HPC
{
private:
    decx::PtrInfo<decx::blas::eig_bisect_interval<_data_type>> _interval_stack;
    decx::PtrInfo<_data_type> _mid_points;

    uint32_t _eig_count_actual;

    uint32_t _max_interval_num;

    _data_type _max_err;

    uint32_t _current_stack_vaild_num;

    // decx::reduce::cpu_Reduce1D_Planner _update_interval;
    decx::blas::cpu_eig_bisect_count_interval<_data_type> _count_intervals;

public:
    _data_type _current_interval_gap;

    cpu_eig_bisect_iter_HPC() {}


    void set_max_err(const _data_type max_err) {
        this->_max_err = max_err;
    }


    void init(const _data_type* p_diag, const _data_type* p_off_diag, const uint32_t N, const _data_type L, const _data_type U, de::DH* handle);


    void iter(const _data_type* p_diag, const _data_type* p_off_diag, const uint32_t N);


    const decx::blas::eig_bisect_interval<_data_type>* get_valid_intervals_array();


    uint32_t get_eig_count() const{
        return this->_current_stack_vaild_num;
    }


    void reset(const _data_type* p_diag, const _data_type* p_off_diag, const uint32_t N, const _data_type L, const _data_type U) 
    {
        auto* p_1st_interval = this->_interval_stack.ptr;
        p_1st_interval[0].set(L, U);
        p_1st_interval[0].count_violent(p_diag, p_off_diag, N);
        this->_mid_points.ptr[0] = (U + L) / 2.f;

        this->_current_stack_vaild_num = 1;
        this->_current_interval_gap = U - L;
    }
};


#endif
