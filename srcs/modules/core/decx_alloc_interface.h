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


#ifndef _DECX_ALLOC_INTERFACE_H_
#define _DECX_ALLOC_INTERFACE_H_


#include <basic.h>
#ifdef _DECX_CUDA_PARTS_
namespace decx
{
    class cuda_stream;
    class cuda_event;
}
#endif


#ifdef __cplusplus
extern "C"
{
#endif
typedef void* DecxMemoryHandler_t;

typedef enum
{
    PAGABLE = 0,
    PAGELOCKED = 1,
    CUDA_DEVICE = 2,
}DecxMemoryType_e;


static inline const char* DecxPraseMemtypeName(const DecxMemoryType_e mem_type)
{
    switch (mem_type)
    {
    case PAGABLE:
        return "pagable";
    
    case PAGELOCKED:
        return "page-locked";

#ifdef _DECX_CUDA_PARTS_
    case CUDA_DEVICE:
        return "cuda device memory";
#endif
    
    default:
        return "";
    }
}


_DECX_API_ int32_t DecxAllocPagable(void** pMemBlock, uint64_t req_size, void** pRawPtrObtained);
_DECX_API_ int32_t DecxAllocPagableRef(void* pMemBlock, void** pRawPtrObtained);
_DECX_API_ int32_t DecxFreePagable(void* pMemBlock);
_DECX_API_ int32_t DecxMemset(void* pMemBlock, const uint64_t size, const uint8_t value);
_DECX_API_ int32_t DecxReallocPagable(void** pMemBlock, uint64_t new_size, void** pRawPtrObtained);
_DECX_API_ int32_t DecxMemIndexGetRawPtr(void* pMemBlock, void** pRawPtrObtained);

#ifdef _DECX_CUDA_PARTS_
_DECX_API_ int32_t DecxAllocCUDA(void** pMemBlock, uint64_t req_size, void** pRawPtrObtained);
_DECX_API_ int32_t DecxAllocCUDARef(void* pMemBlock, void** pRawPtrObtained);
_DECX_API_ int32_t DecxReallocCUDA(void** pMemBlock, uint64_t new_size, void** pRawPtrObtained);
_DECX_API_ int32_t DecxFreeCUDA(void* pMemBlock);
_DECX_API_ int32_t DecxCUDAMemset(void* pMemBlock, const uint64_t size, const uint8_t value, decx::cuda_stream* S);
#endif

#ifdef __cplusplus
}
#endif

#endif