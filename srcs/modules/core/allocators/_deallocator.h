/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

/**
* Memory deallocators are defined in this header
*/


#ifndef __DEALLOCATORS_H_
#define __DEALLOCATORS_H_


#include "../memory_management/MemoryPool_Hv.h"
#include "../memory_management/MemoryPool_Hf.h"
#ifdef _DECX_CUDA_PARTS_
#include "../memory_management/MemoryPool_D.h"
#endif
#include "../../handles/decx_handles.h"
#include "../memory_management/PtrInfo.h"


namespace decx
{
    namespace alloc
    {
        _DECX_API_ void _dealloc_Hv(decx::MemBlock* _ptr);

        template <typename _Ty>
        static void _host_virtual_page_dealloc(decx::PtrInfo<_Ty>* ptr_info);

        _DECX_API_ void _dealloc_Hf(decx::MemBlock* _ptr);


        template <typename _Ty>
        static void _host_fixed_page_dealloc(decx::PtrInfo<_Ty>* ptr_info);


        _DECX_API_ void _dealloc_D(decx::MemBlock* _ptr);


        template <typename _Ty>
        static void _device_dealloc(decx::PtrInfo<_Ty>* ptr_info);
    }
}

#endif