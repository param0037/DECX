/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _DECX_ALLOC_INTERFACE_H_
#define _DECX_ALLOC_INTERFACE_H_


#include "../core/basic.h"
#include "../core/memory_management/MemBlock.h"



namespace decx
{
    namespace alloc
    {
        _DECX_API_ int _alloc_Hv(decx::MemBlock** _ptr, size_t req_size);

        _DECX_API_ int _alloc_Hf(decx::MemBlock** _ptr, size_t req_size);

#if defined(_DECX_CUDA_PARTS_)
        /**
        * @return If successed, 0; If failed -1
        */
        _DECX_API_ int _alloc_D(decx::MemBlock** _ptr, size_t req_size);
#endif

        /** @return If successed, 0; If failed -1 */
        _DECX_API_ void _alloc_Hv_same_place(decx::MemBlock** _ptr);



        /** @return If successed, 0; If failed -1 */
        _DECX_API_ void _alloc_Hf_same_place(decx::MemBlock** _ptr);

#if defined(_DECX_CUDA_PARTS_)
        /** @return If successed, 0; If failed -1 */
        _DECX_API_ void _alloc_D_same_place(decx::MemBlock** _ptr);
#endif
    }
}


// deallocation

namespace decx
{
    namespace alloc
    {
        _DECX_API_ void _dealloc_Hv(decx::MemBlock* _ptr);


        _DECX_API_ void _dealloc_Hf(decx::MemBlock* _ptr);

#if defined(_DECX_CUDA_PARTS_)
        _DECX_API_ void _dealloc_D(decx::MemBlock* _ptr);
#endif
    }
}

namespace decx
{
    namespace alloc {
        _DECX_API_ void Memset_H(decx::MemBlock* _ptr, const size_t size, const int value);

#if defined(_DECX_CUDA_PARTS_)
        _DECX_API_ void Memset_D(decx::MemBlock* _ptr, const size_t size, const int value, cudaStream_t *S);
#endif
    }
}

#endif