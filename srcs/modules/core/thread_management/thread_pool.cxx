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
#include <sched.h>

namespace decx
{
namespace core
{
static TP_InitInfo_t g_tp_init_info[(int)TaskQueueUsage_e::TaskQueue_UsageNum] = 
{
    {._usage = TaskQueueUsage_e::TaskQueue_Generic,  ._thread_cnt = 0,   ._start_initially = 0},
    {._usage = TaskQueueUsage_e::TaskQueue_CalcLoad, ._thread_cnt = -1,  ._start_initially = 1},
    {._usage = TaskQueueUsage_e::TaskQueue_ResMgr,   ._thread_cnt = 1,   ._start_initially = 1},
    {._usage = TaskQueueUsage_e::TaskQueue_Nodes,    ._thread_cnt = 0,   ._start_initially = 0},
    {._usage = TaskQueueUsage_e::TaskQueue_DispWin,  ._thread_cnt = 0,   ._start_initially = 0},
};
}
}

void decx::core::ThreadPool::FindOptimalTaskQueueID_Ranged(int32_t* id, const uint2 _range, const TaskQueueUsage_e usage)
{
    // uint32_t task_que_len = this->current_thread_num;
    // uint32_t least_len = 0xFFFFFFFFU;

    // for (uint32_t i = _range.x; i < _range.y; ++i)
    // {
    //     decx::core::ThreadTaskQueue* tmp_iter = this->_task_schd + i;
    //     if (usage != tmp_iter->GetUsage()){
    //         continue;
    //     }
    //     uint32_t current_len = tmp_iter->GetCurrentTaskNum();

    //     if (current_len != 0) {
    //         if (current_len < least_len)
    //             least_len = current_len;
    //     }
    //     else {
    //         least_len = i;
    //         break;
    //     }
    // }
    // *id = least_len;        // If not found, *id = -1
    *id = 0;
}


void decx::core::ThreadPool::FindOptimalTaskQueueID(int32_t* id, const TaskQueueUsage_e usage)
{
    this->FindOptimalTaskQueueID_Ranged(id, make_uint2(0, this->current_thread_num), usage);
}


_THREAD_FUNCTION_
void decx::core::ThreadPool::__TPMgrTask(const TaskQueueUsage_e usage, const uint32_t slot_id)
{
    // // Set affinity
    // cpu_set_t cpuset;
    // CPU_ZERO(&cpuset);
    // CPU_SET(slot_id, &cpuset);
    // int rval = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    // if (rval){
    //     DECX_LOG_WARN("pthread_setaffinity_np failed, ret=%d", rval);
    // }

    decx::core::ThreadTaskQueue* thread_unit = &(this->_TQ_ctx[(uint32_t)usage][slot_id]._task_schd);
    thread_unit->__TQMainLoop();
    return;
}


void decx::core::ThreadPool::Start()
{
    this->_all_shutdown = false;

    const uint32_t conc_num = std::thread::hardware_concurrency();

    TaskQueueInfo_t tq_init_info = {
        ._switch    = TaskQueueSwitch::TaskQueue_ON,
        ._behaviour = TaskQueueBehaviour_e::TaskQueue_LIFO,
        ._usage     = TaskQueueUsage_e::TaskQueue_Generic,
        ._task_num  = 0};

    for (uint32_t i = 0; i < (uint32_t)TaskQueueUsage_e::TaskQueue_UsageNum; ++i)
    {
        auto* p_init_info = sFindTQInitINfo(g_tp_init_info, (int)TaskQueueUsage_e::TaskQueue_UsageNum, (TaskQueueUsage_e)i);
        if (p_init_info) 
        {
            tq_init_info._usage = (TaskQueueUsage_e)i;
            tq_init_info._switch = (TaskQueueSwitch)p_init_info->_start_initially;
            this->_TQ_valid_nums[i] = (p_init_info->_thread_cnt < 0) ? conc_num : p_init_info->_thread_cnt;
            // if (TaskQueueUsage_e::TaskQueue_CalcLoad == p_init_info->_usage){
                // printf("yes\n");
            // }
            
            for (uint32_t j = 0; j < this->_TQ_valid_nums[i]; ++j)
            {
                new(&this->_TQ_ctx[i][j]._task_schd) ThreadTaskQueue(&tq_init_info);
                
                if (this->_TQ_ctx[i][j]._task_schd.HasInit()) {
                    this->_TQ_ctx[i][j]._task_schd.ExternalTaskQueueHook(this->_TQ_ctx[i].GetRawPtr(), this->_TQ_valid_nums + i);
                    this->_TQ_ctx[i][j]._task_schd.SetSlotID(j);
                }

                // Start the threads
                // if (p_init_info->_start_initially) {
                    new(&this->_TQ_ctx[i][j]._worker) std::thread(&decx::core::ThreadPool::__TPMgrTask, this, tq_init_info._usage, j);
                // }
            }
        }
        
    }
}


decx::core::ThreadPool::ThreadPool(const int thread_num, const bool start_at_begin)
{
    this->_all_shutdown = true;
    this->_max_thr_num = MAX_THREAD_NUM;
    this->current_thread_num = thread_num;

    this->_hardware_concurrent = std::thread::hardware_concurrency();

    for (uint32_t i = 0; i < (uint32_t)TaskQueueUsage_e::TaskQueue_UsageNum; ++i){
        decx_assert(this->_TQ_ctx[i].Allocate(this->_max_thr_num * sizeof(TaskQueueCtx_t), PAGABLE), ALLOC_FAIL);
    }

    this->_sync_label = 0;
    this->_internal_sync_enable = false;

    if (start_at_begin) {
        Start();
    }
}


int32_t decx::core::ThreadPool::AppendThread(const TaskQueueInfo_t* p_init_params)
{
    if (nullptr == p_init_params) {
        DECX_LOG_ERR("Invalid pointer to init params, since it is null");
        return -1;
    }
    else {
        const uint32_t current_worker_num = this->_TQ_valid_nums[(uint32_t)p_init_params->_usage];
        if (current_worker_num == this->_max_thr_num - 1){
            DECX_LOG_ERR("Append thread failed, queue number reaches max:%d", this->_max_thr_num);
            return -1;
        }
        auto* p_target = &this->_TQ_ctx[(uint32_t)p_init_params->_usage][current_worker_num];

        new(&p_target->_task_schd) decx::core::ThreadTaskQueue(p_init_params);
        if (TaskQueueSwitch::TaskQueue_ON == p_init_params->_switch) {
            new(&p_target->_worker) std::thread(&decx::core::ThreadPool::__TPMgrTask, this, p_init_params->_usage, current_worker_num);
        }
        ++this->_TQ_valid_nums[(uint32_t)p_init_params->_usage];
        return current_worker_num;
    }
}


void decx::core::ThreadPool::TerminateAllThreads()
{
    for (uint32_t i = 0; i < (uint32_t)TaskQueueUsage_e::TaskQueue_UsageNum; ++i){
        for (uint32_t j = 0; j < this->_TQ_valid_nums[i]; ++j){
            std::thread* p_worker = &this->_TQ_ctx[i][j]._worker;
            decx::core::ThreadTaskQueue* p_tschd = &this->_TQ_ctx[i][j]._task_schd;

            p_tschd->Switch(TaskQueueSwitch::TaskQueue_OFF);
            if (p_worker->joinable())
                p_worker->join();
        }
        this->_TQ_valid_nums[i] = 0;
    }

    this->_all_shutdown = true;
}

decx::core::TP_InitInfo_t* decx::core::ThreadPool::
sFindTQInitINfo(decx::core::TP_InitInfo_t* p_info_array, const uint32_t search_depth, const TaskQueueUsage_e usage)
{
    for (int32_t i = 0; i < search_depth; ++i) {
        auto* p_info = p_info_array + i;
        if (p_info->_usage == usage) {
            return p_info;
        }
    }
    return nullptr;
}

int32_t decx::core::ThreadPool::TaskQueueMatchedQuery(const TaskQueueInfo_t* p_match)
{
    this->_mtx.lock();
    // int32_t rval = -1;
    // for (int32_t slot_id = 0; slot_id < this->current_thread_num; ++slot_id)
    // {
    //     const auto* p_taskqueue = this->_task_schd + slot_id;
    //     if (p_taskqueue->IsRunning() == 0){
    //         continue;
    //     }

    //     uint32_t pred = ((uint32_t)p_taskqueue->GetBehaviour() ^ (uint32_t)p_match->_behaviour);
    //     pred |= ((uint32_t)p_taskqueue->GetUsage() ^ (uint32_t)p_match->_usage);
    //     pred |= (p_taskqueue->GetCurrentTaskNum() ^ p_match->_task_num);
    //     if (pred == 0){
    //         rval = (int32_t)slot_id;
    //     }
    // }
    this->_mtx.unlock();
    // return rval;
    return 0;
}


decx::core::ThreadPool::~ThreadPool() {
    if (!this->_all_shutdown) {
        TerminateAllThreads();
    }
    for (uint32_t i = 0; i < (uint32_t)TaskQueueUsage_e::TaskQueue_UsageNum; ++i){
        for (uint32_t j = 0; j < this->_TQ_valid_nums[i]; ++j){
            std::thread* p_worker = &this->_TQ_ctx[i][j]._worker;
            decx::core::ThreadTaskQueue* p_tschd = &this->_TQ_ctx[i][j]._task_schd;
            if (p_tschd->HasInit()){
                delete p_tschd;
                delete p_worker;
            }
        }
        this->_TQ_ctx[i].Free();
    }
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


_DECX_API_ int32_t decx::core::
InsertTaskByID(decx::core::TaskImplHandle_t task, const decx::core::TaskQueueUsage_e usage, const uint32_t id)
{
    auto* p_task_queue = &decx::core::thread_pool->_TQ_ctx[(uint32_t)usage][id]._task_schd;
    p_task_queue->RegisterTask(task);
    return 0;
}


_DECX_API_ int32_t decx::core::TaskQueueQuery(const TaskQueueInfo_t* p_match)
{
    return decx::core::thread_pool->TaskQueueMatchedQuery(p_match);
}