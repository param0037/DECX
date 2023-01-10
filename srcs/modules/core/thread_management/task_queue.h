/**
*	---------------------------------------------------------------------
*	Author : Wayne Anderson
*   Date   : 2021.04.16
*	---------------------------------------------------------------------
*	This is a part of the open source program named "DECX", copyright c Wayne,
*	2021.04.16
*/


#ifndef _TASK_QUEUE_H_
#define _TASK_QUEUE_H_

#ifdef _DECX_CPU_CODES_

#include "../basic.h"
#include <queue>

#define MAX_THREAD_NUM 16



// 用一个mutex管理一个task_queue，用另一个mutex来wait()
namespace decx
{
    class ThreadPool;

    class ThreadTaskQueue;
}


typedef std::packaged_task<void()> Task;


class decx::ThreadTaskQueue
{
private:
    void move_ahead();

public:
    // private variables for each thread
    std::mutex _mtx;
    std::condition_variable _cv;
    Task* _task_queue,
        * begin_ptr,
        * end_ptr;

    int _task_num;
    bool _shutdown;

    ThreadTaskQueue();

    template <class FuncType, class ...Args>
    std::future<void> emplace_back(FuncType&& f, Args&& ...args);

    void pop_back();

    void pop_front();


    ~ThreadTaskQueue();
};




template <class FuncType, class ...Args>
static std::future<void> emplace_back(decx::ThreadTaskQueue* _tq, FuncType&& f, Args&& ...args) {
    new (_tq->begin_ptr + _tq->_task_num)Task(
        std::bind(std::forward<FuncType>(f), std::forward<Args>(args)...));

    ++_tq->_task_num;
    return (_tq->begin_ptr + _tq->_task_num - 1)->get_future();
}



#endif


#endif