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


#ifndef _CUDAEVENT_PACKAGE_CUH_
#define _CUDAEVENT_PACKAGE_CUH_


#include "../../../common/basic.h"
#include "cudaStream_package.h"


namespace decx
{
    class cuda_event;
}


class _DECX_API_ decx::cuda_event
{
private:
    cudaEvent_t _E;

public:
    int _event_flag;
    bool _is_occupied;

    cuda_event(const int flag);

    /**
     * Release the cuda_event
    */
    void detach();


    void attach();

    /* Call cudaStreamSynchronize() and the parameter is this->_S */
    void synchronize();


    void event_record(decx::cuda_stream* attached_stream);


    /* Return a referance of cudaStream_t object */
    cudaEvent_t& get_raw_event_ref();

    /* Return a pointer of cudaStream_t object */
    cudaEvent_t* get_raw_event_ptr();


    void release();
};




#endif