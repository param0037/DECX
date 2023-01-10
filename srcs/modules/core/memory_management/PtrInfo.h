/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/



#ifndef _PTR_INFO_H_
#define _PTR_INFO_H_

#include "MemBlock.h"

namespace decx
{
    template <typename _Ty>
    struct PtrInfo;
}


template <typename _Ty>
struct decx::PtrInfo
{
    decx::MemBlock* block;
    _Ty* ptr;
};


#endif