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


#include "cudaEvent_queue.h"
#include "../allocators.h"


decx::cudaEvent_Queue::cudaEvent_Queue()
{
    this->_cuda_event_num = 0;
    this->true_capacity = _CS_STREAM_Q_INIT_SIZE_;
    
    // allocate host memory (page-locked) for decx::cuda_stream
    if (decx::alloc::_host_virtual_page_malloc(&this->_cuda_event_arr, _CS_STREAM_Q_INIT_SIZE_ * sizeof(decx::cuda_stream))) {
        Print_Error_Message(4, "Failed to allocate space for cudaStream on host, cudaEvent_Queue init fail\n");
        exit(-1);
    }
}


decx::cuda_event* decx::cudaEvent_Queue::add_event_physical(const int flag)
{
    if (this->_cuda_event_num > this->true_capacity - 1) {
        // assign a temporary pointer
        decx::PtrInfo<decx::cuda_event> tmp_ptr;
        // physically alloc space for new area
        if (decx::alloc::_alloc_Hv(&(tmp_ptr.block),
            (this->true_capacity + _CS_STREAM_Q_INIT_SIZE_) * sizeof(decx::cuda_stream))) {
            Print_Error_Message(4, "Failed to allocate space for cudaStream on host, cudaEvent_Queue fail to add event\n");
            exit(-1);
        }
        tmp_ptr.ptr = reinterpret_cast<decx::cuda_event*>(tmp_ptr.block->_ptr);

        // copy the old data from this to temp
        memcpy(tmp_ptr.ptr, this->_cuda_event_arr.ptr, this->true_capacity * sizeof(decx::cuda_event));
        // refresh this->true_capacity
        this->true_capacity += _CS_STREAM_Q_INIT_SIZE_;
        // deallocate the old memory space
        decx::alloc::_dealloc_Hv(this->_cuda_event_arr.block);
        // assign the new one to the class
        this->_cuda_event_arr = tmp_ptr;

        // alloc one from back (push_back())
        new(this->_cuda_event_arr.ptr + this->_cuda_event_num) decx::cuda_event(flag);
        // increament on this->_cuda_stream_num
        ++this->_cuda_event_num;
    }
    else {
        // alloc one from back (push_back())
        new(this->_cuda_event_arr.ptr + this->_cuda_event_num) decx::cuda_event(flag);
        // increament on this->_cuda_stream_num
        ++this->_cuda_event_num;
    }

    return (this->_cuda_event_arr.ptr + this->_cuda_event_num - 1);
}


bool decx::cudaEvent_Queue::_find_idle_event(uint* res_dex, const int flag)
{
    for (int i = 0; i < this->_cuda_event_num; ++i) {
        decx::cuda_event* _tmpS = this->_cuda_event_arr.ptr + i;
        if (!_tmpS->_is_occupied && _tmpS->_event_flag == flag) {
            *res_dex = i;
            //_tmpS->attach();
            return true;
        }
    }
    return false;
}


decx::cuda_event* decx::cudaEvent_Queue::event_accessor_ptr(const int flag)
{
    uint dex = 0;

    this->_mtx.lock();

    decx::cuda_event* res_ptr = NULL;
    if (this->_find_idle_event(&dex, flag)) {        // found an idle stream
        res_ptr = this->_cuda_event_arr.ptr + dex;
    }
    else {          // all the streams are occupied
        res_ptr = this->add_event_physical(flag);
    }
    res_ptr->attach();

    this->_mtx.unlock();

    return res_ptr;
}


decx::cuda_event& decx::cudaEvent_Queue::event_accessor_ref(const int flag)
{
    uint dex = 0;
    decx::cuda_event* res_ptr = NULL;
    if (this->_find_idle_event(&dex, flag)) {        // found an idle stream
        res_ptr = this->_cuda_event_arr.ptr + dex;
    }
    else {          // all the streams are occupied
        res_ptr = this->add_event_physical(flag);
    }
    res_ptr->attach();
    return *res_ptr;
}



void decx::cudaEvent_Queue::release()
{
    // call cudaStreamDestroy on each stream
    for (int i = 0; i < this->_cuda_event_num; ++i) {
        (this->_cuda_event_arr.ptr + i)->release();
    }
    // deallocte the stream array
    decx::alloc::_dealloc_Hv(this->_cuda_event_arr.block);
}


decx::cudaEvent_Queue::~cudaEvent_Queue()
{
    decx::alloc::_dealloc_Hv(this->_cuda_event_arr.block);
}


_DECX_API_ decx::cuda_event* decx::cuda::get_cuda_event_ptr(const int flag)
{
    return decx::CEvent.event_accessor_ptr(flag);
}


_DECX_API_ decx::cuda_event& decx::cuda::get_cuda_event_ref(const int flag)
{
    return decx::CEvent.event_accessor_ref(flag);
}