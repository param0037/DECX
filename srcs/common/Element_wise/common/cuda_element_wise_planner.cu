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

#include "cuda_element_wise_planner.h"


void decx::cuda_ElementWise1D_planner::
plan(const uint32_t conc, const uint64_t total, const uint8_t _type_in_size, const uint8_t _type_out_size)
{
    constexpr uint32_t _align_byte = 16;
    
    this->_total = total;
    const uint8_t _ref_size = max(_type_in_size, _type_out_size);

    this->_alignment = _align_byte / _ref_size;

    this->_total_v = decx::utils::ceil<uint64_t>(this->_total, this->_alignment);

    this->_block = _EW_CUDA_BLOCK_SIZE_ == 0 ? decx::cuda::_get_cuda_prop().maxThreadsPerBlock :
        _EW_CUDA_BLOCK_SIZE_;
    
    this->_grid = decx::utils::ceil<uint64_t>(this->_total_v, this->_block);
}
