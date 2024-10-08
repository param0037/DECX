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


#include "cudaStream_queue.h"
#include "../allocators.h"


decx::cudaStream_Queue::cudaStream_Queue()
{
    this->_cuda_stream_num = 0;
    this->true_capacity = _CS_STREAM_Q_INIT_SIZE_;
    
    // allocate host memory (page-locked) for decx::cuda_stream
    if (decx::alloc::_host_virtual_page_malloc(&this->_cuda_stream_arr, _CS_STREAM_Q_INIT_SIZE_ * sizeof(decx::cuda_stream))) {
        Print_Error_Message(4, "Failed to allocate space for cudaStream on host\n");
        exit(-1);
    }
}


decx::cuda_stream* decx::cudaStream_Queue::add_stream_physical(const int flag)
{
    if (this->_cuda_stream_num > this->true_capacity - 1) {
        // assign a temporary pointer
        decx::PtrInfo<decx::cuda_stream> tmp_ptr;
        // physically alloc space for new area
        if (decx::alloc::_alloc_Hv(&(tmp_ptr.block),
            (this->true_capacity + _CS_STREAM_Q_INIT_SIZE_) * sizeof(decx::cuda_stream))) {
            Print_Error_Message(4, "Failed to allocate space for cudaStream on host\n");
            exit(-1);
        }
        tmp_ptr.ptr = reinterpret_cast<decx::cuda_stream*>(tmp_ptr.block->_ptr);

        // copy the old data from this to temp
        memcpy(tmp_ptr.ptr, this->_cuda_stream_arr.ptr, this->true_capacity * sizeof(decx::cuda_stream));
        // refresh this->true_capacity
        this->true_capacity += _CS_STREAM_Q_INIT_SIZE_;
        // deallocate the old memory space
        decx::alloc::_dealloc_Hv(this->_cuda_stream_arr.block);
        // assign the new one to the class
        this->_cuda_stream_arr = tmp_ptr;

        // alloc one from back (push_back())
        new(this->_cuda_stream_arr.ptr + this->_cuda_stream_num) decx::cuda_stream(flag);
        // increament on this->_cuda_stream_num
        ++this->_cuda_stream_num;
    }
    else {
        // alloc one from back (push_back())
        new(this->_cuda_stream_arr.ptr + this->_cuda_stream_num) decx::cuda_stream(flag);
        // increament on this->_cuda_stream_num
        ++this->_cuda_stream_num;
    }

    return (this->_cuda_stream_arr.ptr + this->_cuda_stream_num - 1);
}


bool decx::cudaStream_Queue::_find_idle_stream(uint* res_dex, const int flag)
{
    for (int i = 0; i < this->_cuda_stream_num; ++i) {
        decx::cuda_stream* _tmpS = this->_cuda_stream_arr.ptr + i;
        if (!_tmpS->_is_occupied && _tmpS->_stream_flag == flag) {
            *res_dex = i;
            //_tmpS->attach();
            return true;
        }
    }
    return false;
}


decx::cuda_stream* decx::cudaStream_Queue::stream_accessor_ptr(const int flag)
{
    uint dex = 0;
    decx::cuda_stream* res_ptr = NULL;

    this->_mtx.lock();

    if (this->_find_idle_stream(&dex, flag)) {        // found an idle stream
        res_ptr = this->_cuda_stream_arr.ptr + dex;
    }
    else {          // all the streams are occupied
        res_ptr = this->add_stream_physical(flag);
    }
    res_ptr->attach();

    this->_mtx.unlock();

    return res_ptr;
}


decx::cuda_stream& decx::cudaStream_Queue::stream_accessor_ref(const int flag)
{
    uint dex = 0;
    decx::cuda_stream* res_ptr = NULL;
    if (this->_find_idle_stream(&dex, flag)) {        // found an idle stream
        res_ptr = this->_cuda_stream_arr.ptr + dex;
    }
    else {          // all the streams are occupied
        res_ptr = this->add_stream_physical(flag);
    }
    res_ptr->attach();
    return *res_ptr;
}



void decx::cudaStream_Queue::release()
{
    // call cudaStreamDestroy on each stream
    for (int i = 0; i < this->_cuda_stream_num; ++i) {
        (this->_cuda_stream_arr.ptr + i)->release();
    }
    // deallocte the stream array
    decx::alloc::_dealloc_Hv(this->_cuda_stream_arr.block);
}



decx::cudaStream_Queue::~cudaStream_Queue()
{
    decx::alloc::_dealloc_Hv(this->_cuda_stream_arr.block);
}


_DECX_API_ decx::cuda_stream* decx::cuda::get_cuda_stream_ptr(const int flag)
{
    return decx::CStream.stream_accessor_ptr(flag);
}


_DECX_API_ decx::cuda_stream& decx::cuda::get_cuda_stream_ref(const int flag)
{
    return decx::CStream.stream_accessor_ref(flag);
}