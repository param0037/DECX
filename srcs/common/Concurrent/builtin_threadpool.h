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

#ifndef _BUILTIN_THREADPOOL_H_
#define _BUILTIN_THREADPOOL_H_

#include <basic.h>
#include <vector_defines.h>


#ifdef _MSC_VER
#define _THREAD_FUNCTION_  // represents a function that only runs on threads
#define _THREAD_CALL_      // represents a function that is only called by a thread function
#define _THREAD_GENERAL_   // represents a function that can be called within threads and called as a thread function
#endif
#if defined(__GNUC__) || defined(__clang__)
#define _THREAD_FUNCTION_   __attribute__((hot)) // represents a function that only runs on threads
#define _THREAD_CALL_       __attribute__((hot)) // represents a function that is only called by a thread function
#define _THREAD_GENERAL_    __attribute__((hot)) // represents a function that can be called within threads and called as a thread function
#endif


namespace decx
{
namespace core 
{
    enum class TaskQueueUsage_e
    {
        TaskQueue_Generic = 0,
        TaskQueue_CalcLoad = 1,
        TaskQueue_Nodes = 2,
        taskQueue_DispWin = 3,
    };


    enum class TaskQueueBehaviour_e
    {
        TaskQueue_FIFO = 0,
        TaskQueue_LIFO = 1,
        TaskQueue_Priority = 2,
    };


    _DECX_API_ uint64_t GetOptimalThreadID();


    _DECX_API_ uint64_t GetOptimalThreadID_Ranged(const uint2 range);


    _DECX_API_ uint64_t GetCurrentThreadNum();


    _DECX_API_ uint64_t AppendThread();
}
}



#endif