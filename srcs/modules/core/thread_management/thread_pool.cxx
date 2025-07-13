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


void decx::core::ThreadPool::FindOptimalTaskQueueID_Ranged(int32_t* id, const uint2 _range, const TaskQueueUsage_e usage)
{
    uint32_t task_que_len = this->current_thread_num;
    uint32_t least_len = 0xFFFFFFFFU;

    for (uint32_t i = _range.x; i < _range.y; ++i)
    {
        decx::core::ThreadTaskQueue* tmp_iter = this->_task_schd + i;
        if (usage != tmp_iter->GetUsage()){
            continue;
        }
        uint32_t current_len = tmp_iter->GetCurrentTaskNum();

        if (current_len != 0) {
            if (current_len < least_len)
                least_len = current_len;
        }
        else {
            least_len = i;
            break;
        }
    }
    *id = least_len;        // If not found, *id = -1
}


void decx::core::ThreadPool::FindOptimalTaskQueueID(int32_t* id, const TaskQueueUsage_e usage)
{
    this->FindOptimalTaskQueueID_Ranged(id, make_uint2(0, this->current_thread_num), usage);
}


_THREAD_FUNCTION_
void decx::core::ThreadPool::__TPMgrTask(const uint32_t queue_id)
{
    decx::core::ThreadTaskQueue* thread_unit = &(this->_task_schd[queue_id]);
    thread_unit->__TQMainLoop();
    return;
}


void decx::core::ThreadPool::Start()
{
    this->_all_shutdown = false;

    for (int i = 0; i < this->current_thread_num; ++i) {
        new(this->_task_schd + i) decx::core::ThreadTaskQueue();
    }
    for (uint32_t i = 0; i < this->current_thread_num; ++i) {
        new(this->_thr_list + i) std::thread(&decx::core::ThreadPool::__TPMgrTask, this, i);
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


int32_t decx::core::ThreadPool::AppendThread(const TaskQueueInfo_t* p_init_params)
{
    if (this->current_thread_num + 1 > this->_max_thr_num) {
        DECX_LOG_ERR("Thread number is excessive");
        return -1;
    }
    else {
        new(this->_task_schd + this->current_thread_num) decx::core::ThreadTaskQueue(p_init_params);
        int32_t slot_id = this->current_thread_num;
        new(this->_thr_list + slot_id) std::thread(&decx::core::ThreadPool::__TPMgrTask, this, slot_id);
        this->current_thread_num++;
        return slot_id;
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


int32_t decx::core::ThreadPool::TaskQueueMatchedQuery(const TaskQueueInfo_t* p_match)
{
    this->_mtx.lock();
    int32_t rval = -1;
    for (int32_t slot_id = 0; slot_id < this->current_thread_num; ++slot_id)
    {
        const auto* p_taskqueue = this->_task_schd + slot_id;
        if (p_taskqueue->IsRunning() == 0){
            continue;
        }

        uint32_t pred = ((uint32_t)p_taskqueue->GetBehaviour() ^ (uint32_t)p_match->_behaviour);
        pred |= ((uint32_t)p_taskqueue->GetUsage() ^ (uint32_t)p_match->_usage);
        pred |= (p_taskqueue->GetCurrentTaskNum() ^ p_match->_tsak_num);
        if (pred == 0){
            rval = (int32_t)slot_id;
        }
    }
    this->_mtx.unlock();
    return rval;
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


_DECX_API_ int32_t decx::core::GetOptimalThreadID(const TaskQueueUsage_e usage)
{
    int32_t res_id;
    decx::core::thread_pool->FindOptimalTaskQueueID(&res_id, usage);
    return res_id;
}


_DECX_API_ int32_t decx::core::GetOptimalThreadID_Ranged(const uint2 range, const TaskQueueUsage_e usage)
{
    int32_t res_id;
    decx::core::thread_pool->FindOptimalTaskQueueID_Ranged(&res_id, range, usage);
    return res_id;
}


_DECX_API_ int32_t decx::core::GetCurrentThreadNum()
{
    return decx::core::thread_pool->current_thread_num;
}


_DECX_API_ int32_t decx::core::ThreadpoolAddSot(const TaskQueueInfo_t* p_init_params)
{
    return decx::core::thread_pool->AppendThread(p_init_params);
}


_DECX_API_ int32_t decx::core::InsertTaskByID(decx::core::TaskImplHandle_t task, const uint32_t id)
{
    auto* p_task_queue = decx::core::thread_pool->_task_schd + id;
    p_task_queue->GetMutex().lock();
    p_task_queue->RegisterTask(task);
    p_task_queue->GetMutex().unlock();
    p_task_queue->GetCondVar().notify_one();
    return 0;
}


_DECX_API_ int32_t decx::core::TaskQueueQuery(const TaskQueueInfo_t* p_match)
{
    return decx::core::thread_pool->TaskQueueMatchedQuery(p_match);
}