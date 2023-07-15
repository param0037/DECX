/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#include "_allocator.h"


int decx::alloc::_alloc_D(decx::MemBlock** _ptr, size_t req_size)
{
    decx::mem_pool_D.allocate(req_size, _ptr);

    if ((*_ptr)->_ptr == NULL) {
        return -1;
    }
    return 0;
}


void decx::alloc::_alloc_D_same_place(decx::MemBlock** _ptr)
{
    decx::mem_pool_D.register_reference(*_ptr);
}