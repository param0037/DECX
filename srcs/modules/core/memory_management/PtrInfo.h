/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/



#ifndef _PTR_INFO_H_
#define _PTR_INFO_H_


#include "../include.h"
#include "MemBlock.h"
#include "store_types.h"


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

    decx::DATA_STORE_TYPE _mem_type;

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
        _dst._mem_type = this->_mem_type;

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