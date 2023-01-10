/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/


#ifndef __ALIGNED_ALLOCATORS_H_
#define __ALIGNED_ALLOCATORS_H_

#include "../basic.h"



namespace decx
{
    namespace alloc
    {

        static void* aligned_malloc_Hv(size_t size, size_t alignment);

        static void aligned_free_Hv(void* _ptr);

        static void* aligned_malloc_Hf(size_t size, size_t alignment);

        static void aligned_free_Hf(void* _ptr);



        static void* malloc_D(size_t size, size_t alignment);

        static void free_D(void* _ptr);
    }
}




void* decx::alloc::aligned_malloc_Hv(size_t size, size_t alignment)
{
    if (alignment & (alignment - 1)) {
        return NULL;
    }
    else {
        void* raw_ptr = malloc(sizeof(void*) + size + alignment);
        if (raw_ptr)
        {
            void* begin_ptr = (void*)((size_t)(raw_ptr)+sizeof(void*));
            void* real_ptr = (void*)(((size_t)begin_ptr | (alignment - 1)) + 1);
            ((void**)real_ptr)[-1] = raw_ptr;
            return real_ptr;
        }
        else {
            return NULL;
        }
    }
}



void decx::alloc::aligned_free_Hv(void* _ptr)
{
    free((void*)(((void**)_ptr)[-1]));
}


void* decx::alloc::aligned_malloc_Hf(size_t size, size_t alignment)
{
    if (alignment & (alignment - 1)) {
        return NULL;
    }
    else {
        void* raw_ptr = NULL;
        checkCudaErrors(cudaHostAlloc(&raw_ptr, sizeof(void*) + size + alignment, cudaHostAllocDefault));

        if (raw_ptr)
        {
            void* begin_ptr = (void*)((size_t)(raw_ptr)+sizeof(void*));
            void* real_ptr = (void*)(((size_t)begin_ptr | (alignment - 1)) + 1);
            ((void**)real_ptr)[-1] = raw_ptr;
            return real_ptr;
        }
        else {
            return NULL;
        }
    }
}



void decx::alloc::aligned_free_Hf(void* _ptr)
{
    checkCudaErrors(cudaFreeHost((void*)(((void**)_ptr)[-1])));
}




void* decx::alloc::malloc_D(size_t size, size_t alignment)
{
    void* raw_ptr = NULL;
    checkCudaErrors(cudaMalloc(&raw_ptr, size));
    return raw_ptr;
}



void decx::alloc::free_D(void* _ptr)
{
    checkCudaErrors(cudaFree(_ptr));
}


#endif