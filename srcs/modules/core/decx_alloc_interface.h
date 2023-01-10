/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/

#ifndef _DECX_ALLOC_INTERFACE_H_
#define _DECX_ALLOC_INTERFACE_H_


#include "../core/basic.h"
#include "../core/memory_management/MemBlock.h"


#ifdef Windows
//#pragma comment(lib, "../../../../bin/x64/DECX_allocation.lib")
#endif


namespace decx
{
    namespace alloc
    {
        _DECX_API_ int _alloc_Hv(decx::MemBlock** _ptr, size_t req_size);

        _DECX_API_ int _alloc_Hf(decx::MemBlock** _ptr, size_t req_size);

        /**
        * @return If successed, 0; If failed -1
        */
        _DECX_API_ int _alloc_D(decx::MemBlock** _ptr, size_t req_size);


        /** @return If successed, 0; If failed -1 */
        _DECX_API_ void _alloc_Hv_same_place(decx::MemBlock** _ptr);



        /** @return If successed, 0; If failed -1 */
        _DECX_API_ void _alloc_Hf_same_place(decx::MemBlock** _ptr);

        
        /** @return If successed, 0; If failed -1 */
        _DECX_API_ void _alloc_D_same_place(decx::MemBlock** _ptr);
    }
}


// deallocation

namespace decx
{
    namespace alloc
    {
        _DECX_API_ void _dealloc_Hv(decx::MemBlock* _ptr);


        _DECX_API_ void _dealloc_Hf(decx::MemBlock* _ptr);


        _DECX_API_ void _dealloc_D(decx::MemBlock* _ptr);
    }
}

namespace decx
{
    namespace alloc {
        _DECX_API_ void release_all_tmp();
    }
}

#endif