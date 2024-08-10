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


#ifndef _CUDASTREAM_PACKAGE_CUH_
#define _CUDASTREAM_PACKAGE_CUH_


#include "../../../common/basic.h"


namespace decx
{
    class cuda_stream;
}


class _DECX_API_ decx::cuda_stream
{
private:
    cudaStream_t _S;

public:
    int _stream_flag;
    bool _is_occupied;

    cuda_stream(const int flag);


    void detach();


    void attach();

    /* Call cudaStreamSynchronize() and the parameter is this->_S */
    void synchronize();


    /* Return a referance of cudaStream_t object */
    cudaStream_t& get_raw_stream_ref();

    /* Return a pointer of cudaStream_t object */
    cudaStream_t* get_raw_stream_ptr();


    void release();


    ~cuda_stream();
};

namespace decx
{
    namespace cuda
    {
        _DECX_API_ decx::cuda_stream* get_cuda_stream_ptr(const int flag);


        _DECX_API_ decx::cuda_stream& get_cuda_stream_ref(const int flag);
    }
}


#endif