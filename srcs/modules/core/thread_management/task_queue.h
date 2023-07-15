/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   For More information please visit https://github.com/param0037/DECX
*/


#ifndef _TASK_QUEUE_H_
#define _TASK_QUEUE_H_

#if defined(_DECX_CPU_PARTS_)

#include "../basic.h"
#include "../utils/Dynamic_Array.h"


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


#endif      // if defined(_DECX_CPU_CODES_)


#endif      // ifndef _TASK_QUEUE_H_