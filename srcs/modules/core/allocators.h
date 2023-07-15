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
* Memory allocators are defined in this header
*/

#ifndef _ALLOCATORS_APIS_H_
#define _ALLOCATORS_APIS_H_

#include "decx_alloc_interface.h"
#include "memory_management/PtrInfo.h"

#if defined(_DECX_CUDA_PARTS_)
#include "../core/cudaStream_management/cudaStream_queue.h"
#endif


// allocation

namespace decx
{
    namespace alloc
    {
        template<typename T>
        static int _host_virtual_page_malloc(decx::PtrInfo<T>* ptr_info, size_t size, const bool set_zero = true);


        template<typename T>
        static int _host_fixed_page_malloc(decx::PtrInfo<T>* ptr_info, size_t size, const bool set_zero = true);

#if defined(_DECX_CUDA_PARTS_)
        template<typename T>
        static int _device_malloc(decx::PtrInfo<T>* ptr_info, size_t size, bool _set_zero = false, decx::cuda_stream* S = NULL);
#endif

        template<typename T>
        static void _host_virtual_page_malloc_same_place(decx::PtrInfo<T>* ptr_info);


        template<typename T>
        static void _host_fixed_page_malloc_same_place(decx::PtrInfo<T>* ptr_info);

#if defined(_DECX_CUDA_PARTS_)
        template<typename T>
        static void _device_malloc_same_place(decx::PtrInfo<T>* ptr_info);
#endif

        template<typename T>
        static int _host_fixed_page_realloc(decx::PtrInfo<T>* ptr_info, size_t size);


        template<typename T>
        static int _host_virtual_page_realloc(decx::PtrInfo<T>* ptr_info, size_t size);

#if defined(_DECX_CUDA_PARTS_)
        template<typename T>
        static int _device_realloc(decx::PtrInfo<T>* ptr_info, size_t size);
#endif
    }
}




// deallocation

namespace decx
{
    namespace alloc
    {
        template<typename T>
        static void _host_virtual_page_dealloc(decx::PtrInfo<T>* ptr_info);


        template<typename T>
        static void _host_fixed_page_dealloc(decx::PtrInfo<T>* ptr_info);

#if defined(_DECX_CUDA_PARTS_)
        template<typename T>
        static void _device_dealloc(decx::PtrInfo<T>* ptr_info);
#endif
    }
}



template<typename T>
static void decx::alloc::_host_virtual_page_dealloc(decx::PtrInfo<T>* ptr_info) {
    decx::alloc::_dealloc_Hv(ptr_info->block);
    ptr_info->ptr = NULL;
}




template<typename T>
static void decx::alloc::_host_fixed_page_dealloc(decx::PtrInfo<T>* ptr_info) {
    decx::alloc::_dealloc_Hf(ptr_info->block);
    ptr_info->ptr = NULL;
}


#if defined(_DECX_CUDA_PARTS_)
template<typename T>
static void decx::alloc::_device_dealloc(decx::PtrInfo<T>* ptr_info) {
    decx::alloc::_dealloc_D(ptr_info->block);
    ptr_info->ptr = NULL;
}
#endif


template<typename T>
static int decx::alloc::_host_virtual_page_malloc(decx::PtrInfo<T>* ptr_info, size_t size, const bool set_zero)
{
    int ans = decx::alloc::_alloc_Hv(&ptr_info->block, size);
    ptr_info->_mem_type = decx::DATA_STORE_TYPE::Page_Default;
    ptr_info->ptr = reinterpret_cast<T*>(ptr_info->block->_ptr);
    if (set_zero) {
        memset(ptr_info->ptr, 0, size);
    }
    return ans;
}



template<typename T>
static void decx::alloc::_host_virtual_page_malloc_same_place(decx::PtrInfo<T>* ptr_info)
{
    decx::alloc::_alloc_Hv_same_place(&ptr_info->block);
    //ptr_info->_sync_type();
    ptr_info->ptr = reinterpret_cast<T*>(ptr_info->block->_ptr);
}



template<typename T>
static int decx::alloc::_host_fixed_page_malloc(decx::PtrInfo<T>* ptr_info, size_t size, const bool set_zero)
{
    int ans = decx::alloc::_alloc_Hf(&ptr_info->block, size);
    ptr_info->ptr = reinterpret_cast<T*>(ptr_info->block->_ptr);
    ptr_info->_mem_type = decx::DATA_STORE_TYPE::Page_Locked;
    //ptr_info->_sync_type();
    if (set_zero) {
        memset(ptr_info->ptr, 0, size);
    }
    return ans;
}



template<typename T>
static void decx::alloc::_host_fixed_page_malloc_same_place(decx::PtrInfo<T>* ptr_info)
{
    decx::alloc::_alloc_Hf_same_place(&ptr_info->block);
    //ptr_info->_sync_type();
    ptr_info->ptr = reinterpret_cast<T*>(ptr_info->block->_ptr);
}


#if defined(_DECX_CUDA_PARTS_)
template<typename T>
static int decx::alloc::_device_malloc(decx::PtrInfo<T>* ptr_info, size_t size, bool _set_zero, decx::cuda_stream *S)
{
    int ans = decx::alloc::_alloc_D(&ptr_info->block, size);
    ptr_info->ptr = reinterpret_cast<T*>(ptr_info->block->_ptr);
    ptr_info->_mem_type = decx::DATA_STORE_TYPE::Device_Memory;
    if (_set_zero) {
        if (S == NULL) {
            checkCudaErrors(cudaMemset(ptr_info->ptr, 0, size));
        }
        else{
            checkCudaErrors(cudaMemsetAsync(ptr_info->ptr, 0, size, S->get_raw_stream_ref()));
        }
    }

    return ans;
}
#endif


#if defined(_DECX_CUDA_PARTS_)
template<typename T>
static void decx::alloc::_device_malloc_same_place(decx::PtrInfo<T>* ptr_info)
{
    decx::alloc::_alloc_D_same_place(&ptr_info->block);
    //ptr_info->_sync_type();
    ptr_info->ptr = reinterpret_cast<T*>(ptr_info->block->_ptr);
}
#endif


template<typename T>
int decx::alloc::_host_fixed_page_realloc(decx::PtrInfo<T>* ptr_info, size_t size)
{
    if (ptr_info->block != NULL) {
        if (ptr_info->block->_ptr != NULL) {            // if it is previously allocated
            decx::alloc::_dealloc_Hf(ptr_info->block);
        }
    }
    
    // reallocate new memory of new size
    int ans = decx::alloc::_alloc_Hf(&ptr_info->block, size);
    //ptr_info->_sync_type();
    ptr_info->ptr = reinterpret_cast<T*>(ptr_info->block->_ptr);

    return ans;
}


template<typename T>
int decx::alloc::_host_virtual_page_realloc(decx::PtrInfo<T>* ptr_info, size_t size)
{
    if (ptr_info->block != NULL) {
        if (ptr_info->block->_ptr != NULL) {            // if it is previously allocated
            decx::alloc::_dealloc_Hv(ptr_info->block);
        }
    }
    // reallocate new memory of new size
    int ans = decx::alloc::_alloc_Hv(&ptr_info->block, size);
    //ptr_info->_sync_type();
    ptr_info->ptr = reinterpret_cast<T*>(ptr_info->block->_ptr);

    return ans;
}


#if defined(_DECX_CUDA_PARTS_)
template<typename T>
int decx::alloc::_device_realloc(decx::PtrInfo<T>* ptr_info, size_t size)
{
    if (ptr_info->block != NULL) {
        if (ptr_info->block->_ptr != NULL) {            // if it is previously allocated
            decx::alloc::_dealloc_D(ptr_info->block);
        }
    }
    // reallocate new memory of new size
    int ans = decx::alloc::_alloc_D(&ptr_info->block, size);
    //ptr_info->_sync_type();
    ptr_info->ptr = reinterpret_cast<T*>(ptr_info->block->_ptr);

    return ans;
}
#endif  // #if defined(_DECX_CUDA_PARTS_)


#endif