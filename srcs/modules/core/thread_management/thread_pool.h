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


#ifndef _THREAD_POOL_H_
#define _THREAD_POOL_H_

#ifdef _DECX_CPU_PARTS_

#include "task_queue.h"
#include "../configs/config.h"

#define MAX_THREAD_NUM 1024

#ifdef _MSC_VER
#define _THREAD_FUNCTION_  // represents a function that only runs on threads
#define _THREAD_CALL_      // represents a function that is only called by a thread function
#define _THREAD_GENERAL_   // represents a function that can be called within threads and called as a thread function
#endif
#if defined(__GNUC__) || defined(__clang__)
#define _THREAD_FUNCTION_ __attribute__((hot)) // represents a function that only runs on threads
#define _THREAD_CALL_ __attribute__((hot))     // represents a function that is only called by a thread function
#define _THREAD_GENERAL_ __attribute__((hot))  // represents a function that can be called within threads and called as a thread function
#endif


#ifdef _DECX_CORE_CPU_
class decx::ThreadPool
{
public:
    std::thread* _thr_list;

    decx::ThreadTaskQueue* _task_schd;
    bool _internal_sync_enable;
    std::mutex _mtx;

    uint64_t _max_thr_num, current_thread_num;
    bool _all_shutdown;

    void FindOptimalTaskQueueID(uint64_t* id);


    void FindOptimalTaskQueueID_Ranged(uint64_t* id, const uint2 _range);

    // main_loop callback function running on each thread
    _THREAD_FUNCTION_ void ThreadMainLoop(const uint64_t pool_id);


public:
    // The actual number of concurrent thread this processor supports
    uint64_t _hardware_concurrent;

    uint _sync_label;
    std::mutex _mtx_for_sync;

    void Start();

    ThreadPool(const int thread_num, const bool start_at_begin);


    void AppendThreads(const int add_thread_num);


    void TerminateAllThreads();


    ~ThreadPool();
};
#endif


namespace decx
{
    namespace cpu {
        _DECX_API_ decx::ThreadTaskQueue* GetTaskQueueByID(const uint64_t _idx);


        _DECX_API_ uint64_t GetOptimalThreadID();


        _DECX_API_ uint64_t GetOptimalThreadID_Ranged(const uint2 range);


        _DECX_API_ uint64_t GetCurrentThreadNum();


        _DECX_API_ uint64_t AppendThread();
    }
}



namespace decx {
namespace cpu 
{
    template <class FuncType, class ...Args>
    static std::future<void> RegisterTaskLoadBalanced(FuncType&& f, Args&& ...args)
    {
        uint64_t id = decx::cpu::GetOptimalThreadID_Ranged(
            make_uint2(0, decx::utils::clamp_max<uint64_t>(decx::cpu::_get_permitted_concurrency(), decx::cpu::GetCurrentThreadNum())));
        
        decx::ThreadTaskQueue* tmp_task_que = decx::cpu::GetTaskQueueByID(id);
        tmp_task_que->_mtx.lock();
        std::future<void> fut = decx::InsertTaskBack(tmp_task_que, std::forward<FuncType>(f), std::forward<Args>(args)...);
        tmp_task_que->_mtx.unlock();
        tmp_task_que->_cv.notify_one();

        return fut;
    }


    template <class FuncType, class ...Args>
    static std::future<void> RegisterTaskByID(FuncType&& f, const uint32_t id, Args&& ...args)
    {
        decx::ThreadTaskQueue* tmp_task_que = decx::cpu::GetTaskQueueByID(id);
        tmp_task_que->_mtx.lock();
        std::future<void> fut = decx::InsertTaskBack(tmp_task_que, std::forward<FuncType>(f), std::forward<Args>(args)...);
        tmp_task_que->_mtx.unlock();
        tmp_task_que->_cv.notify_one();

        return fut;
    }


    template <class FuncType, class ...Args>
    static std::future<void> RegisterTaskAppened(FuncType&& f, Args&& ...args)
    {
        uint64_t tid = decx::cpu::AppendThread();
        return decx::cpu::RegisterTaskByID(f, tid, args...);
    }
}
}


#ifdef _DECX_CORE_CPU_
namespace decx
{
    _DECX_API_ extern decx::ThreadPool* thread_pool;        // shared variable
}
#endif

#endif

#ifndef _THREAD_FUNCTION_
#define _THREAD_FUNCTION_
#endif


#ifndef _THREAD_CALL_
#define _THREAD_CALL_
#endif

#ifndef _THREAD_GENERAL_
#define _THREAD_GENERAL_
#endif

#endif