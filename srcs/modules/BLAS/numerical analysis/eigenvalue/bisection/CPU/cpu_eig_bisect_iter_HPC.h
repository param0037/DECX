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


namespace decx
{
namespace blas{
    template <typename _data_type>
    class cpu_eig_bisect_iter_HPC;
}
}


template <typename _data_type>
class decx::blas::cpu_eig_bisect_iter_HPC
{
private:
    decx::PtrInfo<decx::blas::eig_bisect_interval<_data_type>> _interval_stack;
    decx::PtrInfo<_data_type> _count_buffer;

    decx::utils::double_buffer_manager _double_buffer;
    uint32_t _eig_count_actual;

    uint32_t _max_interval_num;

    _data_type _max_err;

    _data_type _current_interval_gap;
    uint32_t _current_stack_vaild_num;

public:
    cpu_eig_bisect_iter_HPC() {}


    void set_max_err(const _data_type max_err) {
        this->_max_err = max_err;
    }


    void init(const _data_type* p_diag, const _data_type* p_off_diag, const uint32_t N, const _data_type L, const _data_type U, de::DH* handle);


    void iter(const _data_type* p_diag, const _data_type* p_off_diag, const uint32_t N);


    const decx::blas::eig_bisect_interval<_data_type>* get_valid_intervals_array();


    uint32_t get_eig_count() const{
        return this->_eig_count_actual;
    }
};


#endif
