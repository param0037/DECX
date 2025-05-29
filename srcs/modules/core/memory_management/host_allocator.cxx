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


int32_t _DECX_API_ DecxAllocPagable(void** p_MemBlockIndex, uint64_t req_size, void** pRawPtrObtained)
{
    *p_MemBlockIndex = nullptr;
    *pRawPtrObtained = nullptr;
    
    decx::MemBlock** p_MB = (decx::MemBlock**)p_MemBlockIndex;
    decx::MemPool_Hv* _mempool_ptr = decx::MemPool_Hv::GetInstance();
    _mempool_ptr->allocate(req_size, p_MB);
    if ((*p_MB)->_ptr == NULL) {
        return -1;
    }
    *pRawPtrObtained = (*p_MB)->_ptr;
    return 0;
}


int32_t _DECX_API_ DecxAllocPagableRef(void* p_MemBlockIndex, void** pRawPtrObtained)
{
    *pRawPtrObtained = nullptr;
    if (p_MemBlockIndex == nullptr){
        DECX_LOG_ERR("Failed to allocate pagable reference, input mempool index is NULL");
        return -1;
    }

    decx::MemBlock* p_MB = (decx::MemBlock*)p_MemBlockIndex;
    decx::MemPool_Hv* _mempool_ptr = decx::MemPool_Hv::GetInstance();
    _mempool_ptr->register_reference(p_MB);
    *pRawPtrObtained = p_MB->_ptr;
    return 0;
}


int32_t _DECX_API_ DecxFreePagable(void* pMemBlock)
{
    if (pMemBlock == nullptr){
        DECX_LOG_ERR("Failed to free since the index handler is NULL");
        return -1;
    }
    decx::MemPool_Hv* _mempool_ptr = decx::MemPool_Hv::GetInstance();
    decx::MemBlock* _ptr = (decx::MemBlock*)pMemBlock;
    _mempool_ptr->deallocate(_ptr);

    return 0;
}

_DECX_API_ int32_t DecxReallocPagable(void** pMemBlock, uint64_t new_size, void** pRawPtrObtained)
{
    int32_t rval = 0;
    if (pMemBlock != nullptr) {
        rval |= DecxFreePagable(*pMemBlock);
    }
    if (rval){
        return -1;
    }
    rval |= DecxAllocPagable(pMemBlock, new_size, pRawPtrObtained);
    return rval;
}

int32_t _DECX_API_ DecxMemset(void* pMemBlock, const uint64_t size, const uint8_t value)
{
    if (pMemBlock == nullptr){
        DECX_LOG_ERR("Failed to memset since the index handler is NULL");
        return -1;
    }
    decx::MemBlock* _ptr = (decx::MemBlock*)pMemBlock;
    memset(_ptr->_ptr, value, size);
    return 0;
}


int32_t _DECX_API_ DecxMemIndexGetRawPtr(void* pMemBlock, void** pRawPtrObtained)
{
    if (pMemBlock == nullptr){
        DECX_LOG_ERR("Failed to prase since the index handler is NULL");
        return -1;
    }
    decx::MemBlock* _ptr = (decx::MemBlock*)pMemBlock;
    *pRawPtrObtained = _ptr->_ptr;
    return 0;
}


int32_t _DECX_API_ DecxReallocPagableLazy(void** pMemBlock, uint64_t new_size, void** pRawPtrObtained)
{
    decx::MemBlock* p_idx_handler = (decx::MemBlock*)pMemBlock;
    if (pMemBlock == nullptr){
        return DecxAllocPagable(pMemBlock, new_size, pRawPtrObtained);
    }
    *pRawPtrObtained = p_idx_handler->_ptr;
    if (new_size > p_idx_handler->block_size){
        return DecxReallocPagable(pMemBlock, new_size, pRawPtrObtained);
    }
    return 0;
}
