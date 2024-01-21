/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _MEMSET_H_
#define _MEMSET_H_

#ifdef _DECX_CORE_CPU_
#include "../memory_management/MemoryPool_Hv.h"
#endif
#ifdef _DECX_CORE_CUDA_
#include "../memory_management/MemoryPool_D.h"
#include "../cudaStream_management/cudaStream_queue.h"
#endif

#include "../../handles/decx_handles.h"
#include "../error.h"


namespace decx {
    namespace alloc {
#ifdef _DECX_CORE_CPU_
        _DECX_API_ void Memset_H(decx::MemBlock* _ptr, const size_t size, const int value);
#endif
#ifdef _DECX_CORE_CUDA_
        _DECX_API_ void Memset_D(decx::MemBlock* _ptr, const size_t size, const int value, cudaStream_t* S);
#endif
    }
}


#endif