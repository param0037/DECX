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

#ifndef _EIG_BISECT_INTERVAL_
#define _EIG_BISECT_INTERVAL_

namespace decx
{
namespace blas{
    template <typename _data_type>
    struct eig_bisect_interval;
}
}

#ifdef _DECX_CPU_PARTS_
#define _EIG_INTERVAL_ALIGN_ __align__(decx::utils::align<uint32_t>(2 * sizeof(_data_type) + 8, 4))
#endif
#ifdef _DECX_CUDA_PARTS_
#define _EIG_INTERVAL_ALIGN_ __align__(decx::utils::align<uint32_t>(2 * sizeof(_data_type) + 8, 16))
#endif


template <typename _data_type> 
struct _EIG_INTERVAL_ALIGN_ decx::blas::eig_bisect_interval
// struct decx::blas::eig_bisect_interval
{
    _data_type _l;
    _data_type _u;
    uint32_t _count_l, _count_u;

#ifdef _DECX_CUDA_PARTS_
    __host__ __device__
#endif
    eig_bisect_interval() : _l(0), _u(0), _count_l(0), _count_u(0) {}


#ifdef _DECX_CUDA_PARTS_
    __host__ __device__
#endif
    void set(const _data_type l, const _data_type u) {
        this->_l = l;
        this->_u = u;
        this->_count_l = 0;
        this->_count_u = 0;
    }


    void count_violent(const _data_type* diag, const _data_type* off_diag, const uint32_t N);


    bool is_valid() const { return (this->_count_u - this->_count_l) > 0; }
};


#endif
