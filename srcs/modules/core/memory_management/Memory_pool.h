/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/

#ifndef _MEMORY_POOL_H_
#define _MEMORY_POOL_H_


#include "../memory_management/MemoryPool_Hv.h"
#include "../memory_management/MemoryPool_Hf.h"
#include "../memory_management/MemoryPool_D.h"


namespace decx
{
    namespace alloc {
        _DECX_API_ void release_all_tmp();
    }
}


namespace decx 
{
    extern decx::MemPool_Hv mem_pool_Hv;

    extern decx::MemPool_Hf mem_pool_Hf;

    extern decx::MemPool_D mem_pool_D;
}



#endif