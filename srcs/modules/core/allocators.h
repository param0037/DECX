/**
*   ----------------------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ----------------------------------------------------------------------------------
* 
* This is a part of the open source project named "DECX", a high-performance scientific
* computational library. This project follows the MIT License. For more information 
* please visit https://github.com/param0037/DECX.
* 
* Copyright (c) 2021 Wayne Anderson
* 
* Permission is hereby granted, free of charge, to any person obtaining a copy of this 
* software and associated documentation files (the "Software"), to deal in the Software 
* without restriction, including without limitation the rights to use, copy, modify, 
* merge, publish, distribute, sublicense, and/or sell copies of the Software, and to 
* permit persons to whom the Software is furnished to do so, subject to the following 
* conditions:
* 
* The above copyright notice and this permission notice shall be included in all copies 
* or substantial portions of the Software.
* 
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
* INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR 
* PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE 
* FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR 
* OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
* DEALINGS IN THE SOFTWARE.
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
        static int _host_virtual_page_malloc(decx::PtrInfo<T>* ptr_info, uint64_t size, const bool set_zero = true);

        
        template<typename T>
        static int _host_virtual_page_malloc_lazy(decx::PtrInfo<T>* ptr_info, uint64_t size, const bool set_zero = true);

#if defined(_DECX_CUDA_PARTS_)
        template<typename T>
        static int _device_malloc(decx::PtrInfo<T>* ptr_info, uint64_t size, bool _set_zero = false, decx::cuda_stream* S = NULL);
#endif

        template<typename T>
        static void _host_virtual_page_malloc_same_place(decx::PtrInfo<T>* ptr_info);


#if defined(_DECX_CUDA_PARTS_)
        template<typename T>
        static void _device_malloc_same_place(decx::PtrInfo<T>* ptr_info);
#endif

        template<typename T>
        static int _host_virtual_page_realloc(decx::PtrInfo<T>* ptr_info, uint64_t size, const bool set_zero = true);

#if defined(_DECX_CUDA_PARTS_)
        template<typename T>
        static int _device_realloc(decx::PtrInfo<T>* ptr_info, uint64_t size, bool _set_zero = false, decx::cuda_stream* S = NULL);
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
    ptr_info->block = NULL;
}



#if defined(_DECX_CUDA_PARTS_)
template<typename T>
static void decx::alloc::_device_dealloc(decx::PtrInfo<T>* ptr_info) {
    decx::alloc::_dealloc_D(ptr_info->block);
    ptr_info->ptr = NULL;
    ptr_info->block = NULL;
}
#endif


template<typename T>
static int decx::alloc::_host_virtual_page_malloc(decx::PtrInfo<T>* ptr_info, uint64_t size, const bool set_zero)
{
    int32_t ans = decx::alloc::_alloc_Hv(&ptr_info->block, size);
    ptr_info->ptr = reinterpret_cast<T*>(ptr_info->block->_ptr);
    if (set_zero) {
        memset(ptr_info->ptr, 0, size);
    }
    return ans;
}


/**
 * @brief If the required size is less than the size already allocated, the function WILL NOT allocate
 * any space.
*/
template<typename T>
static int decx::alloc::_host_virtual_page_malloc_lazy(decx::PtrInfo<T>* ptr_info, uint64_t size, const bool set_zero)
{
    if (ptr_info){
        if (ptr_info->block){
            if (ptr_info->block->block_size < size){
                int32_t ans = _host_virtual_page_realloc(ptr_info, size);
                return ans;
            }
        }
    }
    return decx::alloc::_host_virtual_page_malloc(ptr_info, size, set_zero);
}


template<typename T>
static void decx::alloc::_host_virtual_page_malloc_same_place(decx::PtrInfo<T>* ptr_info)
{
    decx::alloc::_alloc_Hv_same_place(&ptr_info->block);
    ptr_info->ptr = reinterpret_cast<T*>(ptr_info->block->_ptr);
}



#if defined(_DECX_CUDA_PARTS_)
template<typename T>
static int decx::alloc::_device_malloc(decx::PtrInfo<T>* ptr_info, uint64_t size, bool _set_zero, decx::cuda_stream *S)
{
    int ans = decx::alloc::_alloc_D(&ptr_info->block, size);
    ptr_info->ptr = reinterpret_cast<T*>(ptr_info->block->_ptr);
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
    ptr_info->ptr = reinterpret_cast<T*>(ptr_info->block->_ptr);
}
#endif



template<typename T>
int decx::alloc::_host_virtual_page_realloc(decx::PtrInfo<T>* ptr_info, uint64_t size, const bool set_zero)
{
    if (ptr_info->block != NULL) {
        if (ptr_info->block->_ptr != NULL) {            // if it is previously allocated
            decx::alloc::_dealloc_Hv(ptr_info->block);
        }
    }
    // reallocate new memory of new size
    int ans = decx::alloc::_alloc_Hv(&ptr_info->block, size);
    ptr_info->ptr = reinterpret_cast<T*>(ptr_info->block->_ptr);
    if (set_zero) {
        memset(ptr_info->ptr, 0, size);
    }

    return ans;
}


#if defined(_DECX_CUDA_PARTS_)
template<typename T>
int decx::alloc::_device_realloc(decx::PtrInfo<T>* ptr_info, uint64_t size, bool _set_zero, decx::cuda_stream* S)
{
    if (ptr_info->block != NULL) {
        if (ptr_info->block->_ptr != NULL) {            // if it is previously allocated
            decx::alloc::_dealloc_D(ptr_info->block);
        }
    }
    // reallocate new memory of new size
    int ans = decx::alloc::_alloc_D(&ptr_info->block, size);
    ptr_info->ptr = reinterpret_cast<T*>(ptr_info->block->_ptr);

    if (_set_zero) {
        if (S == NULL) {
            checkCudaErrors(cudaMemset(ptr_info->ptr, 0, size));
        }
        else {
            checkCudaErrors(cudaMemsetAsync(ptr_info->ptr, 0, size, S->get_raw_stream_ref()));
        }
    }

    return ans;
}
#endif  // #if defined(_DECX_CUDA_PARTS_)


#endif