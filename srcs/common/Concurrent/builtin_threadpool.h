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

#ifndef _BUILTIN_THREADPOOL_H_
#define _BUILTIN_THREADPOOL_H_

#include <basic.h>
#include <vector_defines.h>
#include "task_impl.h"

namespace decx
{
namespace core 
{
    enum class TaskQueueUsage_e : uint8_t
    {
        TaskQueue_Generic = 0,
        TaskQueue_CalcLoad = 1,
        TaskQueue_ResMgr = 2,
        TaskQueue_Nodes = 3,
        TaskQueue_DispWin = 4,
    };


    enum class TaskQueueBehaviour_e : uint8_t
    {
        TaskQueue_LIFO = 0,
        TaskQueue_FIFO = 1,
        TaskQueue_Priority = 2,
    };


    enum class TaskQueueSwitch : uint8_t
    {
        TaskQueue_OFF = 0,
        TaskQueue_ON = 1,
    };


    struct TaskQueueInfo_t
    {
        TaskQueueSwitch         _switch;
        TaskQueueBehaviour_e    _behaviour;
        TaskQueueUsage_e        _usage;
        uint32_t                _tsak_num;
    };


    enum class ThreadDispatchMethod_e
    {
        // Always create a new thread in the threadpool for the task unconditionally.
        Dispatch_NewSlot = 0,

        // Find the task queue that holds the least tasks and push the task to it, load balanced.
        Dispatch_LoadBalanced = 1,

        // Push the task to the task queue by indicated slot ID.
        Dispatch_ByID = 2,
    };


    _DECX_API_ int32_t GetOptimalThreadID(const TaskQueueUsage_e usage);


    _DECX_API_ int32_t GetOptimalThreadID_Ranged(const uint2 range, const TaskQueueUsage_e usage);


    _DECX_API_ int32_t GetCurrentThreadNum();


    /**
     * @brief Append one task queue to the thread pool
     * @param p_init_param Initialize parameters for the created task queue
     * @return Slot id of the created task queue
     */
    _DECX_API_ int32_t ThreadpoolAddSot(const TaskQueueInfo_t* p_init_params);


    /**
     * @brief Find the first taskqueue slot id that matches all the requirement described in p_match
     * @param p_match Match info
     * @return -1 for not found; otherwise for found slot id
     */
    _DECX_API_ int32_t TaskQueueQuery(const TaskQueueInfo_t* p_match);


    _DECX_API_ int32_t InsertTaskByID(decx::core::TaskImplHandle_t task, const uint32_t id);
}
}



#endif