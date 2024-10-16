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

#ifndef _EIGENVALUE_H_
#define _EIGENVALUE_H_

#include <basic.h>
#include <Classes/Matrix.h>
#include <Element_wise/common/cpu_element_wise_planner.h>
#include "cpu_eig_bisect_iter_HPC.h"
#include <Algorithms/reduce/CPU/cpu_reduce_planner.h>


namespace decx
{
namespace blas{
    template <typename _data_type>
    class cpu_eig_bisection;
}
}


template <typename _data_type>
class decx::blas::cpu_eig_bisection
{
private:
    decx::_matrix_layout _layout;

    decx::PtrInfo<_data_type> _diag;
    decx::PtrInfo<_data_type> _off_diag;

    _data_type _Gerschgorin_L, _Gerschgorin_U;

    uint8_t _alignment;

    uint32_t _aligned_N;

    decx::cpu_ElementWise1D_planner _diag_extractor;
    uint32_t _concurrency;

    decx::PtrInfo<_data_type> _shared_mem;

    decx::reduce::cpu_Reduce1D_Planner _Gersch_bound_founder;

    decx::blas::cpu_eig_bisect_iter_HPC<_data_type> _iter_scheduler;

public:
    cpu_eig_bisection() {
        memset(this, 0, sizeof(decx::blas::cpu_eig_bisection<_data_type>));
    }


    void Init(const uint32_t conc, const decx::_matrix_layout* layout, const _data_type max_err, de::DH* handle);


    void extract_diagonal(const _data_type* src, decx::utils::_thread_arrange_1D* t1D);


    void calc_Gerschgorin_bound(decx::utils::_thread_arrange_1D* t1D);


    void plan(const decx::_Matrix* mat, decx::utils::_thread_arrange_1D* t1D, de::DH* handle);


    _data_type* get_diag() const {return this->_diag.ptr;}
    _data_type* get_off_diag() const {return this->_off_diag.ptr;}

    _data_type get_Gerschgorin_L() const { return this->_Gerschgorin_L;}
    _data_type get_Gerschgorin_U() const { return this->_Gerschgorin_U;}


    void iter_bisection();


    const decx::blas::eig_bisect_interval<_data_type>* get_valid_intervals_array()
    {
        return this->_iter_scheduler.get_valid_intervals_array();
    }

    uint32_t get_eig_count() const{
        return this->_iter_scheduler.get_eig_count();
    }

    void reset() {
        this->_iter_scheduler.reset(this->_diag.ptr, this->_off_diag.ptr, this->_layout.width, this->_Gerschgorin_L, this->_Gerschgorin_U);
    }
};

#endif
