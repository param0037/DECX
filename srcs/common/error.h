/**
* ----------------------------------------------------------------------------------
* Author : Wayne Anderson
* Date : 2021.04.16
* ----------------------------------------------------------------------------------
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


#ifndef _ERROR_H_
#define _ERROR_H_

#include <basic.h>
#include <log_console.h>

#define decx_assert(__predicate, ...) {                                     \
    if_opt ((__predicate)) {                                                \
        DECX_LOG(LOG_ERROR, "runtime_assert", __FUNCTION__, __VA_ARGS__);   \
        exit(EXIT_FAILURE);                                                 \
    }                                                                       \
}                                                                           \

#ifdef _DECX_CUDA_PARTS_
static inline const char* _cudaGetErrorEnum(cudaError_t error) noexcept
{
    return cudaGetErrorName(error);
}


template <typename T>
void check(T result, char const* const func, const char* const file, int const line)
{
    if (result) {
        DECX_LOG(LOG_ERROR, "CUDA_SDK", __FUNCTION__, "CUDA error at %s:%d code=%d(%s) \"%s\"", file, line,
        static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
        exit(EXIT_FAILURE);
    }
}


#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)
#endif


#endif