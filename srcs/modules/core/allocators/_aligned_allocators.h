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


#ifndef __ALIGNED_ALLOCATORS_H_
#define __ALIGNED_ALLOCATORS_H_

#include "../basic.h"


namespace decx
{
    namespace alloc
    {
#ifdef _DECX_CORE_CPU_
        static void* aligned_malloc_Hv(size_t size, size_t alignment);

        static void aligned_free_Hv(void* _ptr);

        static void* aligned_malloc_Hf(size_t size, size_t alignment);

        static void aligned_free_Hf(void* _ptr);
#endif
#ifdef _DECX_CORE_CUDA_
        static void* malloc_D(size_t size, size_t alignment);

        static void free_D(void* _ptr);
#endif
    }
}



#ifdef _DECX_CORE_CPU_
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



// 若是CUDA，成员变量memory_type才有意义，否则空在那里，初始化一个0
// 但是class_creation 函数就会改变，加上cpu::, cuda::的命名空间加以区分
// 用户在both的条件下可以调用任意函数创造主类，Matrix，Vector等，但在只有cpu的情况下只能调用cpu::名下的creation函数创造类。
void* decx::alloc::aligned_malloc_Hf(size_t size, size_t alignment)
{
    if (alignment & (alignment - 1)) {
        return NULL;
    }
    else {
        void* raw_ptr = NULL;
        //checkCudaErrors(cudaHostAlloc(&raw_ptr, sizeof(void*) + size + alignment, cudaHostAllocDefault));
        raw_ptr = malloc(sizeof(void*) + size + alignment);
        //raw_ptr = VirtualAlloc(NULL, sizeof(void*) + size + alignment, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);

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
    //checkCudaErrors(cudaFreeHost((void*)(((void**)_ptr)[-1])));
    free((void*)(((void**)_ptr)[-1]));
    //VirtualFree((void*)(((void**)_ptr)[-1]), 0, MEM_RELEASE);
}




#endif


#ifdef _DECX_CORE_CUDA_


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
#endif  // #ifdef _DECX_CORE_CUDA_

#endif  // #ifndef __ALIGNED_ALLOCATORS_H_