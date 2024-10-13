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

#ifndef _ELEMENT_WISE_BASE_H_
#define _ELEMENT_WISE_BASE_H_

#include "../../basic.h"
#include "../../FMGR/fragment_arrangment.h"
#include "../../../modules/core/configs/config.h"
#ifdef _DECX_CPU_PARTS_
#include "../../../modules/core/thread_management/thread_arrange.h"
#include "../../../modules/core/thread_management/thread_pool.h"
#endif
#ifdef _DECX_CUDA_PARTS_
#include "../../../modules/core/cudaStream_management/cudaEvent_queue.h"
#include "../../../modules/core/cudaStream_management/cudaStream_queue.h"
#endif


namespace decx
{
    class element_wise_base_1D;
    class element_wise_base_2D;

    enum EW_Caller_Argument_Type{
        EW_ARG_UNCHANGED = 0,
        EW_ARG_UPDATED,
    };

    template <decx::EW_Caller_Argument_Type ArgType, 
              typename _data_type, 
              typename FuncType> class EW_Arg;
}


template <decx::EW_Caller_Argument_Type ArgType, 
          typename _data_type, 
          typename FuncType>
class decx::EW_Arg
{
    using T_updator = _data_type(const int32_t);

    _data_type _arg;
    FuncType _updator;

public:
    EW_Arg(_data_type arg, FuncType updator) :
        _arg(arg), _updator(updator) {}


    EW_Arg(FuncType updator) :
        _updator(updator) {}


    _data_type value(const int32_t i) {
        if 
#if __cplusplus >= 201703L
        constexpr 
#endif
        (ArgType == decx::EW_Caller_Argument_Type::EW_ARG_UPDATED){
            return this->_updator(i);
        }else{
            return this->_arg;
        }
    }
};


namespace decx
{
    template <EW_Caller_Argument_Type ArgType, typename _data_type, typename FuncType>
    inline decx::EW_Arg<ArgType, _data_type, FuncType> EW_Arg_helper(_data_type val, FuncType func){
        return decx::EW_Arg<ArgType, _data_type, FuncType>(val, func);
    }


    template <EW_Caller_Argument_Type ArgType, typename _data_type, typename FuncType>
    inline decx::EW_Arg<ArgType, _data_type, FuncType> EW_Arg_helper(FuncType func){
        return decx::EW_Arg<ArgType, _data_type, FuncType>(func);
    }
}


#ifdef _DECX_CPU_PARTS_
#if defined(__x86_64__) || defined(__i386__)
#define _EW_MIN_THREAD_PROC_DEFAULT_CPU_ 256
#endif
#if defined(__aarch64__) || defined(__arm__)
#define _EW_MIN_THREAD_PROC_DEFAULT_CPU_ 128
#endif
#endif

class decx::element_wise_base_1D
{
protected:
    uint64_t _total;

    uint32_t _alignment;

    uint8_t _type_in_size;
    uint8_t _type_out_size;

    uint64_t _total_v;

public:
    void plan_alignment()
    {
#ifdef _DECX_CPU_PARTS_
#if defined(__x86_64__) || defined(__i386__)
        constexpr uint32_t _align_byte = 32;
#endif
#if defined(__aarch64__) || defined(__arm__)
        constexpr uint32_t _align_byte = 16;
#endif
#endif
#ifdef _DECX_CUDA_PARTS_
        constexpr uint32_t _align_byte = 16;
#endif

        const uint8_t _ref_size = max(this->_type_in_size, this->_type_out_size);

        if (_ref_size > sizeof(uint8_t)){
            this->_alignment = _align_byte / _ref_size;
        }
        else{
#ifdef _DECX_CUDA_PARTS_
            this->_alignment = 4;
#endif
#ifdef _DECX_CPU_PARTS_
            this->_alignment = 8;
#endif
        }
    }
};


class decx::element_wise_base_2D : public decx::element_wise_base_1D
{
protected:
    uint2 _proc_dims;
    uint32_t _proc_w_v;
};



#endif
