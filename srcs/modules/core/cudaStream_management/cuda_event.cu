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