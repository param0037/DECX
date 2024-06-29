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


#include "../core/basic.h"
#include "../core/memory_management/MemBlock.h"



namespace decx
{
    namespace alloc
    {
        _DECX_API_ int _alloc_Hv(decx::MemBlock** _ptr, size_t req_size);

        //_DECX_API_ int _alloc_Hf(decx::MemBlock** _ptr, size_t req_size);

#if defined(_DECX_CUDA_PARTS_)
        /**
        * @return If successed, 0; If failed -1
        */
        _DECX_API_ int _alloc_D(decx::MemBlock** _ptr, size_t req_size);
#endif

        /** @return If successed, 0; If failed -1 */
        _DECX_API_ void _alloc_Hv_same_place(decx::MemBlock** _ptr);



        /** @return If successed, 0; If failed -1 */
        //_DECX_API_ void _alloc_Hf_same_place(decx::MemBlock** _ptr);

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


        //_DECX_API_ void _dealloc_Hf(decx::MemBlock* _ptr);

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