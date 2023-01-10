/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
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


void decx::cuda_stream::this_stream_sync()
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
