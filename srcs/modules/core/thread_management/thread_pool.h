/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/


#ifndef _THREAD_POOL_H_
#define _THREAD_POOL_H_

#ifdef _DECX_CPU_CODES_

#include "task_queue.h"

#define MAX_THREAD_NUM 16



class decx::ThreadPool
{
    /*template <class FuncType, class ...Args>
    friend std::future<void> register_task(decx::ThreadPool* _tp, FuncType&& f, Args&& ...args);*/
public:

    std::thread* _thr_list;

    decx::ThreadTaskQueue* _task_schd;
    bool _internal_sync_enable;
    std::mutex _mtx;

    size_t _max_thr_num, current_thread_num;
    bool _all_shutdown;

    void _find_task_queue_id(size_t* id);

    // main_loop callback function running on each thread
    void _thread_main_loop(const size_t pool_id);


public:
    size_t _hardware_concurrent;

    uint _sync_label;
    std::mutex _mtx_for_sync;

    void Start();

    ThreadPool(const int thread_num, const bool start_at_begin);

    /*template <class FuncType, class ...Args>
    std::future<void> register_task(FuncType&& f, Args&& ...args);*/


    /*template <class FuncType, class ...Args>
    std::future<void> register_task_by_id(size_t id, FuncType&& f, Args&& ...args);*/


    void add_thread(const int add_thread_num);


    void TerminateAllThreads();


    ~ThreadPool();
};


namespace decx {
    namespace cpu {
        template <class FuncType, class ...Args>
        static std::future<void> register_task(decx::ThreadPool* _tp, FuncType&& f, Args&& ...args)
        {
            size_t id;
            _tp->_find_task_queue_id(&id);

            decx::ThreadTaskQueue* tmp_task_que = &(_tp->_task_schd[id]);
            tmp_task_que->_mtx.lock();
            std::future<void> fut = emplace_back(tmp_task_que, std::forward<FuncType>(f), std::forward<Args>(args)...);
            tmp_task_que->_mtx.unlock();
            tmp_task_que->_cv.notify_one();

            return fut;
        }



        template <class FuncType, class ...Args>
        std::future<void> register_task_by_id(decx::ThreadPool* _tp, size_t id, FuncType&& f, Args&& ...args)
        {
            decx::ThreadTaskQueue* tmp_task_que = &(_tp->_task_schd[id]);
            std::future<void> fut = tmp_task_que->emplace_back(std::forward<FuncType>(f), std::forward<Args>(args)...);
            tmp_task_que->_cv.notify_one();

            return fut;
        }
    }
}



namespace decx
{
    extern decx::ThreadPool thread_pool;        // shared variable
}


#endif

// represents a function that only runs on threads
#define _THREAD_FUNCTION_ _NO_ALIAS_ 
// represents a function that is only called by a thread function
#define _THREAD_CALL_ _NO_ALIAS_ _NO_THROW_
// represents a function that can be called within threads and called as a thread function
#define _THREAD_GENERAL_ _NO_ALIAS_

#endif