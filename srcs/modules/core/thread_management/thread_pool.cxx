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


#include "thread_pool.h"
#define MODULE_TAG "Builtin_Threadpool"


void decx::core::ThreadPool::FindOptimalTaskQueueID(uint64_t* id)
{
    uint64_t task_que_len = this->current_thread_num;
    uint64_t res_id = 0,
        least_len = (this->_task_schd)->GetCurrentTaskNum();

    if (least_len != 0) {
        for (uint64_t i = 1; i < task_que_len; ++i)
        {
            decx::core::ThreadTaskQueue* tmp_iter = this->_task_schd + i;

            uint64_t current_len = tmp_iter->GetCurrentTaskNum();

            if (current_len != 0) {
                if (current_len < least_len)
                    least_len = current_len;
            }
            else {
                least_len = i;
                break;
            }
        }
        *id = least_len;
    }
    else {
        *id = res_id;
    }
}


void decx::core::ThreadPool::FindOptimalTaskQueueID_Ranged(uint64_t* id, const uint2 _range)
{
    uint64_t task_que_len = this->current_thread_num;
    uint64_t res_id = 0,
        least_len = (this->_task_schd + _range.x)->GetCurrentTaskNum();

    if (least_len != 0) {
        for (uint64_t i = _range.x + 1; i < _range.y; ++i)
        {
            decx::core::ThreadTaskQueue* tmp_iter = this->_task_schd + i;

            uint64_t current_len = tmp_iter->GetCurrentTaskNum();

            if (current_len != 0) {
                if (current_len < least_len)
                    least_len = current_len;
            }
            else {
                least_len = i;
                break;
            }
        }
        *id = least_len;
    }
    else {
        *id = res_id;
    }
}


_THREAD_FUNCTION_
void decx::core::ThreadPool::ThreadMainLoop(const uint64_t queue_id)
{
    decx::core::ThreadTaskQueue* thread_unit = &(this->_task_schd[queue_id]);
    thread_unit->ThreadMainLoop();
    return;
}


void decx::core::ThreadPool::Start()
{
    this->_all_shutdown = false;

    for (int i = 0; i < this->current_thread_num; ++i) {
        new(this->_task_schd + i) decx::core::ThreadTaskQueue();
    }
    for (uint64_t i = 0; i < this->current_thread_num; ++i) {
        new(this->_thr_list + i) std::thread(&decx::core::ThreadPool::ThreadMainLoop, this, i);
    }
}


decx::core::ThreadPool::ThreadPool(const int thread_num, const bool start_at_begin)
{
    this->_all_shutdown = true;
    this->_max_thr_num = MAX_THREAD_NUM;
    this->current_thread_num = thread_num;

    this->_hardware_concurrent = std::thread::hardware_concurrency();

    this->_task_schd = (decx::core::ThreadTaskQueue*)malloc(this->_max_thr_num * sizeof(decx::core::ThreadTaskQueue));
    this->_thr_list = (std::thread*)malloc(this->_max_thr_num * sizeof(std::thread));

    this->_sync_label = 0;
    this->_internal_sync_enable = false;

    if (start_at_begin) {
        Start();
    }
}


void decx::core::ThreadPool::AppendThreads(const int add_thread_num)
{
    if (this->current_thread_num + add_thread_num > this->_max_thr_num) {
        DECX_LOG_ERR("Thread number is excessive");
        return;
    }
    else {
        for (int32_t i = 0; i < add_thread_num; ++i) {
            new(this->_task_schd + this->current_thread_num + i) decx::core::ThreadTaskQueue();
        }
        for (int32_t i = 0, idx = this->current_thread_num; i < add_thread_num; ++i, ++idx) {
            new(this->_thr_list + idx) std::thread(&decx::core::ThreadPool::ThreadMainLoop, this, idx);
        }
        this->current_thread_num += add_thread_num;
    }
}


void decx::core::ThreadPool::TerminateAllThreads()
{
    for (int i = 0; i < this->current_thread_num; ++i) {
        std::thread* _iter = this->_thr_list + i;
        decx::core::ThreadTaskQueue* Tschd_iter = this->_task_schd + i;
        {
            std::unique_lock<std::mutex> lck(Tschd_iter->GetMutex());
            Tschd_iter->Switch(TaskQueueSwitch::TaskQueue_OFF);
        }
        Tschd_iter->GetCondVar().notify_one();
        _iter->join();
    }

    this->_all_shutdown = true;
}



decx::core::ThreadPool::~ThreadPool() {
    if (!this->_all_shutdown) {
        TerminateAllThreads();
    }
    for (int i = 0; i < this->current_thread_num; ++i) {
        std::thread* _iter = this->_thr_list + i;
        _iter->~thread();
    }

    free(this->_task_schd);
    free(this->_thr_list);
}


decx::core::ThreadPool* decx::core::thread_pool;


_DECX_API_ uint64_t decx::core::GetOptimalThreadID()
{
    uint64_t res_id;
    decx::core::thread_pool->FindOptimalTaskQueueID(&res_id);
    return res_id;
}


_DECX_API_ uint64_t decx::core::GetOptimalThreadID_Ranged(const uint2 range)
{
    uint64_t res_id;
    decx::core::thread_pool->FindOptimalTaskQueueID_Ranged(&res_id, range);
    return res_id;
}


_DECX_API_ uint64_t decx::core::GetCurrentThreadNum()
{
    return decx::core::thread_pool->current_thread_num;
}


_DECX_API_ uint64_t decx::core::ThreadpoolAddSot(const TaskQueueUsage_e usage, const TaskQueueBehaviour_e behaviour)
{
    decx::core::thread_pool->AppendThreads(1);
    return decx::core::GetCurrentThreadNum() - 1;
}


_DECX_API_ int32_t decx::cpu::InsertTaskByID(decx::core::TaskImplHandle_t task, const uint32_t id)
{
    auto* p_task_queue = decx::core::thread_pool->_task_schd + id;
    p_task_queue->GetMutex().lock();
    p_task_queue->RegisterTask(task);
    p_task_queue->GetMutex().unlock();
    p_task_queue->GetCondVar().notify_one();
    return 0;
}
