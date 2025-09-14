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
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
* INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR 
* PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDES BE LIABLE 
* FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR 
* OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
* DEALINGS IN THE SOFTWARE.
*/


#include "task_queue.h"


decx::core::ThreadTaskQueue::ThreadTaskQueue(const TaskQueueInfo_t* p_init_param)
{
    this->_shutdown.store(TaskQueueSwitch::TaskQueue_OFF == p_init_param->_switch ? 1 : 0);
    this->_behaviour = p_init_param->_behaviour;
    this->_usage = p_init_param->_usage;

    DecxCore_BinarySemaphoreCreate(&this->_sem);

    this->_task_queue.Allocate(10);
    this->_share_flag.store(TQ_TaskShareIdle, std::memory_order_relaxed);
    this->_has_init = 1;
}


_THREAD_GENERAL_ void decx::core::ThreadTaskQueue::__TQMainLoop()
{
    const DecxWaitSettings_t tq_sem_wait_settings = {
        ._option = DecxWaitOpt_Relaxed,
        ._max_spin_cnt = 1000,
        ._spin_factor_exp = 3,
        ._timeout_msec = DECX_WAIT_FOREVER
    };

    while (!this->_shutdown.load(std::memory_order_relaxed))
    {
        decx::core::TaskImplHandle_t task;
        decx::utils::LFQ_Status_e rval = decx::utils::LFQ_Status_e::LFQ_Success;

        DecxCore_BinarySemaphoreWait(&this->_sem, &tq_sem_wait_settings);
        rval = this->_task_queue.PopBack(&task);
        
        // Execute task if we got one
        if (decx::utils::LFQ_Status_e::LFQ_Success == rval) {
            task->Execute();
        }

        // decx::core::TaskImplHandle_t stolen_task;
        // if (TaskFinder(&stolen_task) == 0){
        //     stolen_task->Execute();
        // }
    }
    return;
}


void decx::core::ThreadTaskQueue::Switch(const TaskQueueSwitch switch_stage)
{
    this->_shutdown.store((TaskQueueSwitch::TaskQueue_OFF == switch_stage) ? 1 : 0);
    DecxCore_BinarySemaphorePost(&this->_sem);
}


uint32_t decx::core::ThreadTaskQueue::GetCurrentTaskNum() const
{
#if _TQ_USE_LOCKQ_
    return (uint32_t)this->_task_queue.size();
#else
    return (uint32_t)this->_task_queue.Size();
#endif
}


int32_t decx::core::ThreadTaskQueue::RegisterTask(decx::core::TaskImplHandle_t task_hdlr)
{
    this->_task_queue.Enqueue(task_hdlr);
    DecxCore_BinarySemaphorePost(&this->_sem);
    return 0;
}


int32_t decx::core::ThreadTaskQueue::
ExternalTaskQueueHook(decx::core::TaskQueueCtx_t* p_task_arr_ext, const uint32_t* p_tq_cnt)
{
    if (nullptr == p_task_arr_ext){
        return -1;
    }
    this->_external_queue = p_task_arr_ext;
    this->_external_tq_num = p_tq_cnt;
    return 0;
}


int32_t decx::core::ThreadTaskQueue::
TaskFinder(decx::core::TaskImplHandle_t* p_task)
{
    const uint32_t num = *this->_external_tq_num;
    decx::core::TaskImplHandle_t task;
    decx::utils::LFQ_Status_e rval = decx::utils::LFQ_Status_e::LFQ_Success;
    for (uint32_t i = 0; i < num; ++i){
        if (i == this->_slot_id){
            continue;
        }
        auto* p_tq = &this->_external_queue[i]._task_schd;
        if (p_tq->HasInit()) 
        {
            if (p_tq->_task_queue.Size() == 0){
                continue;
            }

            if (p_tq->_share_flag.load(std::memory_order_acquire) != TQ_TaskShareIdle) {
                continue;
            }

            // if (this->_futex.load(std::memory_order_relaxed) > 0){
            //     return 2;
            // }

            TQ_TaskShareStatus_e share_flag_exp = TQ_TaskShareIdle;
            // Attempt to transition the target queue into sharing state only once.
            // If it's not idle, skip to the next queue to avoid spinning indefinitely.
            if (!p_tq->_share_flag.compare_exchange_strong(share_flag_exp, TQ_TaskSharing, std::memory_order_release, std::memory_order_relaxed)) {
                continue;
            }
            rval = p_tq->_task_queue.TryDequeue(&task);
            // _mm_pause();
            if (decx::utils::LFQ_Status_e::LFQ_Success == rval) {
                *p_task = task;
                // this->_share_flag.store(0, std::memory_order_release);
                p_tq->_share_flag.store(TQ_TaskShareIdle, std::memory_order_release);
                return 0;
            }
            p_tq->_share_flag.store(TQ_TaskShareIdle, std::memory_order_release);
        }
    }
    // this->_share_flag.store(0, std::memory_order_release);
    return 1;
}
