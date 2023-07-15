/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#include "_memset.h"

//
//_DECX_API_ void decx::alloc::Memset_H(decx::MemBlock* _ptr, const size_t size, const int value)
//{
//    memset(_ptr->_ptr, value, size);
//}



_DECX_API_ void decx::alloc::Memset_D(decx::MemBlock* _ptr, const size_t size, const int value, cudaStream_t* S)
{
    if (S == NULL) {
        checkCudaErrors(cudaMemset(_ptr->_ptr, value, size));
    }
    else {
        checkCudaErrors(cudaMemsetAsync(_ptr->_ptr, value, size, *S));
    }
}