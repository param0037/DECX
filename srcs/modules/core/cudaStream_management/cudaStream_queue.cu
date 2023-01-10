/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/

#include "cudaStream_queue.h"


decx::cudaStream_Queue::cudaStream_Queue()
{
    this->_cuda_stream_num = 0;
    this->true_capacity = _CS_STREAM_Q_INIT_SIZE_;

    // allocate host memory (page-locked) for decx::cuda_stream
    if (decx::alloc::_alloc_Hv(
        &(this->_cuda_stream_arr.block), _CS_STREAM_Q_INIT_SIZE_ * sizeof(decx::cuda_stream))) {
        Print_Error_Message(4, "Failed to allocate space for cudaStream on host\n");
        exit(-1);
    }
    this->_cuda_stream_arr.ptr = reinterpret_cast<decx::cuda_stream*>(this->_cuda_stream_arr.block->_ptr);
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
    if (this->_find_idle_stream(&dex, flag)) {        // found an idle stream
        res_ptr = this->_cuda_stream_arr.ptr + dex;
    }
    else {          // all the streams are occupied
        res_ptr = this->add_stream_physical(flag);
    }
    res_ptr->attach();
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