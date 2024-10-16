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

#ifndef _ALLOCATOR_H_
#define _ALLOCATOR_H_


#include "../memory_management/MemoryPool_Hv.h"
#ifdef _DECX_CUDA_PARTS_
#include "../memory_management/MemoryPool_D.h"
#endif
#include "../../../common/Handle/decx_handle.h"



namespace decx
{
    namespace alloc
    {
        /** @return If successed, 0; If failed -1 */
        _DECX_API_ int _alloc_Hv(decx::MemBlock** _ptr, size_t req_size);

        /** @return If successed, 0; If failed -1 */
        _DECX_API_ int _alloc_Hf(decx::MemBlock** _ptr, size_t req_size);

        /** @return If successed, 0; If failed -1 */
        _DECX_API_ int _alloc_D(decx::MemBlock** _ptr, size_t req_size);

        /** @return If successed, 0; If failed -1 */
        _DECX_API_ void _alloc_Hv_same_place(decx::MemBlock** _ptr);

        /** @return If successed, 0; If failed -1 */
        _DECX_API_ void _alloc_Hf_same_place(decx::MemBlock** _ptr);

#ifdef _DECX_CUDA_PARTS_
        /** @return If successed, 0; If failed -1 */
        _DECX_API_ void _alloc_D_same_place(decx::MemBlock** _ptr);
#endif
    }
}



#endif