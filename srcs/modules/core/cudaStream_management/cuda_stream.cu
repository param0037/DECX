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


#include "cudaStream_package.h"


decx::cuda_stream::cuda_stream(const int flag)
{
    checkCudaErrors(cudaStreamCreateWithFlags(&this->_S, flag));
    this->_stream_flag = flag;
}


void decx::cuda_stream::detach()
{
    this->_is_occupied = false;
}


void decx::cuda_stream::attach()
{
    this->_is_occupied = true;
}


void decx::cuda_stream::synchronize()
{
    checkCudaErrors(cudaStreamSynchronize(this->_S));
}


cudaStream_t& decx::cuda_stream::get_raw_stream_ref()
{
    return this->_S;
}


cudaStream_t* decx::cuda_stream::get_raw_stream_ptr()
{
    return &(this->_S);
}


void decx::cuda_stream::release()
{
    checkCudaErrors(cudaStreamDestroy(this->_S));
}


decx::cuda_stream::~cuda_stream()
{
    checkCudaErrors(cudaStreamDestroy(this->_S));
}