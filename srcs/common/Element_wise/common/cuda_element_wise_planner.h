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

#ifndef _CUDA_ELEMENT_WISE_PLANNER_H_
#define _CUDA_ELEMENT_WISE_PLANNER_H_

#include "../../basic.h"
#include "../../FMGR/fragment_arrangment.h"
#include "../../../modules/core/configs/config.h"
#include "element_wise_base.h"


namespace decx
{
    class cuda_ElementWise1D_planner;
    class cuda_ElementWise2D_planner;
    
}


#define _EW_CUDA_BLOCK_SIZE_ 256


class decx::cuda_ElementWise1D_planner : decx::element_wise_base_1D
{
private:
    uint32_t _block, _grid;

public:
    cuda_ElementWise1D_planner() {}


    void plan(const uint64_t total, const uint8_t _type_in_size, const uint8_t _type_out_size);


    template <typename FuncType, typename _type_in, typename _type_out, class ...Args>
    inline void caller_unary(FuncType&& f, const _type_in* src, _type_out* dst, decx::cuda_stream* S, Args&& ...additional)
    {
        f(src, dst, this->_total_v, this->_block, this->_grid, S, additional...);
    }


    template <typename FuncType, typename _type_in1, typename _type_in2, typename _type_out, class ...Args>
    inline void caller_binary(FuncType&& f, const _type_in1* A, const _type_in2* B, _type_out* dst, decx::cuda_stream* S, Args&& ...additional)
    {
        f(A, B, dst, this->_total_v, this->_block, this->_grid, S, additional...);
    }

};


class decx::cuda_ElementWise2D_planner : public decx::element_wise_base_2D
{
protected:
    dim3 _block, _grid;

public:
    cuda_ElementWise2D_planner() {}


    void plan(const uint2 proc_dims, const uint8_t type_in_size, const uint8_t type_out_size);


    template <typename FuncType, typename _type_in, typename _type_out, class ...Args>
    inline void caller_unary(FuncType&& f, const _type_in* src, _type_out* dst, const uint32_t Wsrc, const uint32_t Wdst, 
        decx::cuda_stream* S, Args&& ...additional)
    {
        f(src, dst, Wsrc, Wdst, make_uint2(this->_proc_w_v, this->_proc_dims.y), S, additional...);
    }


    template <typename FuncType, typename _type_in1, typename _type_in2, typename _type_out, class ...Args>
    inline void caller_unary(FuncType&& f, const _type_in1* A, const _type_in2* B, _type_out* dst, const uint32_t Wsrc, const uint32_t Wdst, 
        decx::cuda_stream* S, Args&& ...additional)
    {
        f(A, B, dst, Wsrc, Wdst, make_uint2(this->_proc_w_v, this->_proc_dims.y), S, additional...);
    }
};


#endif
