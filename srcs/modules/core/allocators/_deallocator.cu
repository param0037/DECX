/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/

#include "_deallocator.h"




void decx::alloc::_dealloc_Hv(decx::MemBlock* _ptr) {
    decx::mem_pool_Hv.deallocate(_ptr);
}


void decx::alloc::_dealloc_Hf(decx::MemBlock* _ptr) {
    decx::mem_pool_Hf.deallocate(_ptr);
}


void decx::alloc::_dealloc_D(decx::MemBlock* _ptr) {
    decx::mem_pool_D.deallocate(_ptr);
}