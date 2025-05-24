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


#include "_allocator.h"
#include <decx_alloc_interface.h>
#include <log_console.h>


#define MODULE_TAG "DecxAlloc"


_DECX_API_ int32_t DecxAllocCUDA(void** pMemBlock, uint64_t req_size, void** pRawPtrObtained)
{
    *pMemBlock = nullptr;
    *pRawPtrObtained = nullptr;

    decx::MemBlock** p_MB = (decx::MemBlock**)pMemBlock;

    decx::MemPool_D* _mempool_ptr = decx::MemPool_D::GetInstance();
    _mempool_ptr->allocate(req_size, p_MB);

    if ((*p_MB)->_ptr == NULL) {
        return -1;
    }
    *pRawPtrObtained = (*p_MB)->_ptr;
    return 0;
}


_DECX_API_ int32_t DecxAllocCUDARef(void* pMemBlock, void** pRawPtrObtained)
{
    *pRawPtrObtained = nullptr;
    if (pMemBlock == nullptr){
        DECX_LOG_ERR("Failed to allocate pagable reference, input mempool index is NULL");
        return -1;
    }

    decx::MemBlock* p_MB = (decx::MemBlock*)pMemBlock;

    decx::MemPool_D* _mempool_ptr = decx::MemPool_D::GetInstance();
    _mempool_ptr->register_reference(p_MB);

    *pRawPtrObtained = p_MB->_ptr;
    return 0;
}


_DECX_API_ int32_t DecxFreeCUDA(void* pMemBlock)
{
    decx::MemPool_D* _mempool_ptr = decx::MemPool_D::GetInstance();
    _mempool_ptr->deallocate((decx::MemBlock*)pMemBlock);

    return 0;
}

_DECX_API_ int32_t DecxReallocCUDA(void** pMemBlock, uint64_t new_size, void** pRawPtrObtained)
{
    int32_t rval = 0;
    rval = DecxFreeCUDA(*pMemBlock);
    if (rval){
        return -1;
    }
    rval |= DecxAllocCUDA(pMemBlock, new_size, pRawPtrObtained);
    return rval;
}

_DECX_API_ int32_t DecxCUDAMemset(void* pMemBlock, const uint64_t size, const uint8_t value, decx::cuda_stream* S)
{
    decx::MemBlock* _ptr = (decx::MemBlock*)pMemBlock;
    if (S == nullptr){
        checkCudaErrors(cudaMemset(_ptr->_ptr, value, size));
    }
    else{
        checkCudaErrors(cudaMemsetAsync(_ptr->_ptr, value, size, S->get_raw_stream_ref()));
    }
    return 0;
}
