/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#include "cudaEvent_package.h"


decx::cuda_event::cuda_event(const int flag)
{
    checkCudaErrors(cudaEventCreateWithFlags(&this->_E, flag));
    this->_event_flag = flag;
}


void decx::cuda_event::detach()
{
    this->_is_occupied = false;
}


void decx::cuda_event::attach()
{
    this->_is_occupied = true;
}


void decx::cuda_event::synchronize()
{
    checkCudaErrors(cudaEventSynchronize(this->_E));
}


void decx::cuda_event::event_record(decx::cuda_stream* attached_stream)
{
    checkCudaErrors(cudaEventRecord(this->_E, attached_stream->get_raw_stream_ref()));
}


cudaEvent_t& decx::cuda_event::get_raw_event_ref()
{
    return this->_E;
}


cudaEvent_t* decx::cuda_event::get_raw_event_ptr()
{
    return &(this->_E);
}


void decx::cuda_event::release()
{
    checkCudaErrors(cudaEventDestroy(this->_E));
}