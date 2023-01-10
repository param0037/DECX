/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/

#include "Memory_pool.h"


decx::MemPool_Hv decx::mem_pool_Hv;

decx::MemPool_Hf decx::mem_pool_Hf;

decx::MemPool_D decx::mem_pool_D;


_DECX_API_ void decx::alloc::release_all_tmp()
{
    decx::mem_pool_Hv.release();
    decx::mem_pool_Hf.release();
    decx::mem_pool_D.release();
}