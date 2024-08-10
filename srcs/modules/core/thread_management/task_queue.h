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


#ifndef _TASK_QUEUE_H_
#define _TASK_QUEUE_H_

#if defined(_DECX_CPU_PARTS_)

#include "../../../common/basic.h"
#include "../../../common/Array/Dynamic_Array.h"


#define MAX_THREAD_NUM 16



namespace decx
{
#ifdef _DECX_CORE_CPU_
    /**
    * This threadpool is designed for intensive tasks. That is, system will exit when the number of actual thread
    * exceeds that of the maximum thread. There is no other object that is corresponding to each thread.
    * Each thread can be used repeatedly as long as it is sleeping
    */
    class ThreadPool;
#endif

    class ThreadTaskQueue;
}


typedef std::packaged_task<void()> Task;


class decx::ThreadTaskQueue
{
public:
    // private variables for each thread
    std::mutex _mtx;
    std::condition_variable _cv;

    decx::utils::Dynamic_Array<Task> _task_queue;

    bool _shutdown;

    ThreadTaskQueue();
};



namespace decx 
{
    template <class FuncType, class ...Args>
    static std::future<void> emplace_back(decx::ThreadTaskQueue* _tq, FuncType&& f, Args&& ...args) {
        _tq->_task_queue.emplace_back(std::bind(std::forward<FuncType>(f), std::forward<Args>(args)...));

        return _tq->_task_queue.back()->get_future();
    }
}


#endif      // if defined(_DECX_CPU_PARTS_)


#endif      // ifndef _TASK_QUEUE_H_