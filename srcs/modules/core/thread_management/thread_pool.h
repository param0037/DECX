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

#include "task_queue.h"
#include <configs/config.h>
#include <vector_defines.h>
#include <Concurrent/builtin_threadpool.h>


#define MAX_THREAD_NUM 1024


namespace decx
{
namespace core
{
    /**
    * This threadpool is designed for intensive tasks. That is, system will exit when the number of actual thread
    * exceeds that of the maximum thread. There is no other object that is corresponding to each thread.
    * Each thread can be used repeatedly as long as it is sleeping
    */
    class ThreadPool;
}
}


class decx::core::ThreadPool
{
public:
    std::thread*            _thr_list;

    decx::core::ThreadTaskQueue*  _task_schd;
    bool                    _internal_sync_enable;
    std::mutex              _mtx;

    uint32_t                _max_thr_num, 
                            current_thread_num;
    bool                    _all_shutdown;

    void FindOptimalTaskQueueID(int32_t* id, const TaskQueueUsage_e usage);


    void FindOptimalTaskQueueID_Ranged(int32_t* id, const uint2 _range, const TaskQueueUsage_e usage);


    // main_loop callback function running on each thread
    _THREAD_FUNCTION_ void __TPMgrTask(const uint32_t pool_id);


public:
    // The actual number of concurrent thread this processor supports
    uint32_t _hardware_concurrent;

    uint _sync_label;
    std::mutex _mtx_for_sync;

    void Start();

    ThreadPool(const int thread_num, const bool start_at_begin);


    int32_t AppendThread(const TaskQueueInfo_t* p_init_params);


    void TerminateAllThreads();


    int32_t TaskQueueMatchedQuery(const TaskQueueInfo_t* p_match);


    ~ThreadPool();
};


namespace decx
{
namespace core {
    extern decx::core::ThreadPool* thread_pool;        // shared variable
}
}


namespace decx {
namespace cpu 
{
    template <class FuncType, class ...Args>
    static std::future<void> RegisterTaskLoadBalanced(FuncType&& f, Args&& ...args)
    {
        // uint32_t id = decx::cpu::GetOptimalThreadID_Ranged(
        //     make_uint2(0, decx::utils::clamp_max<uint32_t>(DecxGetPermitConcurrency(), decx::cpu::GetCurrentThreadNum())));
        
        // decx::core::ThreadTaskQueue* tmp_task_que = decx::cpu::GetTaskQueueByID(id);
        // tmp_task_que->_mtx.lock();
        // std::future<void> fut = decx::InsertTaskBack(tmp_task_que, std::forward<FuncType>(f), std::forward<Args>(args)...);
        // tmp_task_que->_mtx.unlock();
        // tmp_task_que->_cv.notify_one();

        // return fut;
        return std::future<void>();
    }


    template <class FuncType, class ...Args>
    static std::future<void> RegisterTaskByID(FuncType&& f, const uint32_t id, Args&& ...args)
    {
        // decx::core::ThreadTaskQueue* tmp_task_que = decx::cpu::GetTaskQueueByID(id);
        // tmp_task_que->_mtx.lock();
        // std::future<void> fut = decx::InsertTaskBack(tmp_task_que, std::forward<FuncType>(f), std::forward<Args>(args)...);
        // tmp_task_que->_mtx.unlock();
        // tmp_task_que->_cv.notify_one();

        // return fut;
        return std::future<void>();
    }


    template <class FuncType, class ...Args>
    static std::future<void> RegisterTaskAppened(FuncType&& f, Args&& ...args)
    {
        // uint32_t tid = decx::cpu::AppendThread();
        // return decx::cpu::RegisterTaskByID(f, tid, args...);
        return std::future<void>();
    }


    // template <class FuncType, class ...Args>
    // static std::future<void> RegisterTask(FuncType&& f, const decx::cpu::ThreadDispatchMethod_e method, const uint32_t id, Args&& ...args)
    // {
    //     switch (method)
    //     {
    //     case decx::cpu::ThreadDispatchMethod_e::Dispatch_NewSlot:
    //         return RegisterTaskAppened(f, args...);

    //     case decx::cpu::ThreadDispatchMethod_e::Dispatch_ByID:
    //         return RegisterTaskByID(f, id, args...);
        
    //     case decx::cpu::ThreadDispatchMethod_e::Dispatch_LoadBalanced:
    //     default:
    //         return RegisterTaskLoadBalanced(f, args...);
    //     }
    // }
}
}

#endif