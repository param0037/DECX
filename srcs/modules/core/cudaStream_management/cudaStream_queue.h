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


#ifndef _CUDASTREAM_QUEUE_CUH_
#define _CUDASTREAM_QUEUE_CUH_


#include "../basic.h"
#include "cudaStream_package.h"
#include "../decx_alloc_interface.h"
#include "../memory_management/PtrInfo.h"


#define _CS_STREAM_Q_INIT_SIZE_ 10


#ifdef _DECX_CORE_CUDA_

namespace decx
{
    class cudaStream_Queue;
}


class decx::cudaStream_Queue
{
private:
    std::mutex _mtx;

    size_t true_capacity;

    decx::PtrInfo<decx::cuda_stream> _cuda_stream_arr;

    /*
    * This function will not label the found cuda_stream as occupied
    * @param uint *res_dex : This is the pointer of result index
    * @param const int flag : This is the flag of cudaStream
    */
    bool _find_idle_stream(uint *res_dex, const int flag);


    decx::cuda_stream* add_stream_physical(const int flag);

public:
    size_t _cuda_stream_num;


    cudaStream_Queue();


    decx::cuda_stream* stream_accessor_ptr(const int flag);


    decx::cuda_stream& stream_accessor_ref(const int flag);


    void release();


    ~cudaStream_Queue();
};




namespace decx
{
    extern decx::cudaStream_Queue CStream;
}

#endif

namespace decx
{
    namespace cuda
    {
        _DECX_API_ decx::cuda_stream* get_cuda_stream_ptr(const int flag);


        _DECX_API_ decx::cuda_stream& get_cuda_stream_ref(const int flag);
    }
}

#endif