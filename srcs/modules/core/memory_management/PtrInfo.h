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


#ifndef _PTR_INFO_H_
#define _PTR_INFO_H_


#include "../../../common/include.h"
#include "MemBlock.h"


namespace decx
{
    template <typename _Ty>
    struct PtrInfo;


    template <typename _Ty>
    struct Ptr2D_Info;
}


template <typename _Ty>
struct decx::PtrInfo
{
    decx::MemBlock* block;
    _Ty* ptr;

    PtrInfo() {
        this->block = NULL;
        this->ptr = NULL;
    }

    template <typename _Type_dst>
    decx::PtrInfo<_Type_dst> _type_cast()
    {
        decx::PtrInfo<_Type_dst> _dst;
        _dst.block = this->block;
        _dst.ptr = reinterpret_cast<_Type_dst*>(this->ptr);

        return _dst;
    }
};


template <typename _Ty>
struct decx::Ptr2D_Info
{
    decx::PtrInfo<_Ty> _ptr;
    uint2 _dims;

    Ptr2D_Info() {
        this->_dims = make_uint2(0, 0);
    }

    Ptr2D_Info(decx::PtrInfo<_Ty> _ptr_info, const uint2 dims)
    {
        this->_dims = dims;
        this->_ptr = _ptr_info;
    }
};


#endif