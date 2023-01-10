/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/


/**
* Memory allocators are defined in this header
*/

#ifndef _ALLOCATOR_H_
#define _ALLOCATOR_H_


#include "../memory_management/Memory_pool.h"
#include "../../handles/decx_handles.h"



namespace decx
{
    namespace alloc
    {
        /** @return If successed, 0; If failed -1 */
        _DECX_API_ int _alloc_Hv(decx::MemBlock** _ptr, size_t req_size);

        /** @return If successed, 0; If failed -1 */
        _DECX_API_ int _alloc_Hf(decx::MemBlock** _ptr, size_t req_size);

        /** @return If successed, 0; If failed -1 */
        _DECX_API_ int _alloc_D(decx::MemBlock** _ptr, size_t req_size);

        /** @return If successed, 0; If failed -1 */
        _DECX_API_ void _alloc_Hv_same_place(decx::MemBlock** _ptr);

        /** @return If successed, 0; If failed -1 */
        _DECX_API_ void _alloc_Hf_same_place(decx::MemBlock** _ptr);

        /** @return If successed, 0; If failed -1 */
        _DECX_API_ void _alloc_D_same_place(decx::MemBlock** _ptr);
    }
}



#endif