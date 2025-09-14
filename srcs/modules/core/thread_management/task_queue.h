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


#include <basic.h>
#include <Array/Dynamic_Array.h>
#include <Concurrent/task_handle.h>
#include <Concurrent/lock.h>
#include "utils/loockfree_ringbuffer.h"
#include <Concurrent/lock.h>
#include <Concurrent/semaphore.h>

namespace decx
{
namespace core
{
    class ThreadTaskQueue;

    struct TaskQueueCtx_t;


    enum TQ_TaskShareStatus_e : uint8_t
    {
        TQ_TaskShareIdle = 0,
        TQ_TaskWaiting = 1,
        TQ_TaskSharing = 2,
    };
}
}

class decx::core::ThreadTaskQueue
{
private:
    // private variables for each thread
    DecxLock_t _lock;
    decx::utils::Lockfree_RingBuffer<decx::core::TaskImplHandle_t> _task_queue;

    decx::core::TaskQueueCtx_t*         _external_queue;
    const uint32_t*                     _external_tq_num;

    std::atomic<uint8_t>                _shutdown;

    decx::core::TaskQueueBehaviour_e    _behaviour;
    decx::core::TaskQueueUsage_e        _usage;
    uint8_t                             _has_init = 0;
    uint32_t                            _slot_id;

    // Added: synchronized task count used as wait predicate
    
    DecxBinarySemaphore_t _sem;

private:
    int32_t TaskFinder(decx::core::TaskImplHandle_t** p_task, const decx::core::TaskQueueUsage_e target_usage);

public:
    std::atomic<TQ_TaskShareStatus_e> _share_flag;

    ThreadTaskQueue(const TaskQueueInfo_t* p_init_param);


    void Switch(const TaskQueueSwitch switch_stage);


    _THREAD_GENERAL_ void __TQMainLoop();


    uint8_t IsRunning() const
    {
        return this->_shutdown.load();
    }


    void SetSlotID(const uint32_t slot_id) {
        this->_slot_id = slot_id;
    }


    const TaskQueueUsage_e& GetUsage() const
    {
        return this->_usage;
    }

    const TaskQueueBehaviour_e& GetBehaviour() const
    {
        return this->_behaviour;
    }


    uint32_t GetCurrentTaskNum() const;


    uint8_t HasInit() const {return this->_has_init;}


    int32_t RegisterTask(decx::core::TaskImplHandle_t task_hdlr);


    int32_t ExternalTaskQueueHook(decx::core::TaskQueueCtx_t* p_task_arr_ext, const uint32_t* p_tq_cnt);


    int32_t TaskFinder(decx::core::TaskImplHandle_t* p_task);
};

struct decx::core::TaskQueueCtx_t
{
    decx::core::ThreadTaskQueue _task_schd;
    std::thread                 _worker;
};


#endif      // ifndef _TASK_QUEUE_H_
