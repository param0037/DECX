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


#ifndef _TASK_HANDLE_H_
#define _TASK_HANDLE_H_

#include "task_impl.h"
#include "builtin_threadpool.h"

#define TASK_IMPL_SIZE_CALC \
    (TASK_PACK_MAX_SIZE - sizeof(int32_t) - sizeof(ThreadDispatchMethod_e) - sizeof(decx::core::TaskQueueUsage_e))

namespace decx
{
namespace core
{
    struct __align__(TASK_PACK_MAX_SIZE) TaskHandle_t
    {
        alignas(TASK_PACK_MAX_SIZE) uint8_t _task_impl[TASK_IMPL_SIZE_CALC];
        int32_t _slot_id;
        ThreadDispatchMethod_e _dispatch_method;
        decx::core::TaskQueueUsage_e _usage;
    };
}
}


#define PACK_CPY(data) (data)
#define PACK_REF(data) std::ref(data)


namespace decx
{
namespace core
{
	template <typename FuncType, typename ... ArgTypes> static inline
	int32_t TaskCreate(const ThreadDispatchMethod_e method, const decx::core::TaskQueueUsage_e usage, int32_t slot_id, 
                       TaskHandle_t* task_hdlr, FuncType task_entry, ArgTypes ... args)
	{
        if (nullptr == task_hdlr) {
            return -1;
        }
		using TaskType = decx::core::Task<FuncType, ArgTypes...>;
		static_assert(sizeof(TaskType) <= TASK_IMPL_SIZE_CALC, "Task size is too large");
		new(task_hdlr->_task_impl) decx::core::Task<FuncType, ArgTypes...>(std::forward<FuncType>(task_entry), std::forward<ArgTypes>(args)...);
        task_hdlr->_dispatch_method = method;
        task_hdlr->_slot_id = slot_id;
        task_hdlr->_usage = usage;
        return 0;
	}


    static int32_t TaskDestroy(TaskHandle_t* task_hdlr)
    {
        if (nullptr == task_hdlr) {
            return -1;
        }
        auto* p_task = reinterpret_cast<decx::core::TaskImplHandle_t>(task_hdlr->_task_impl);
        p_task->~TaskBase();
        return 0;
    }


    static int32_t TaskRun(TaskHandle_t* task_hdlr)
    {
        if (nullptr == task_hdlr) {
            return -1;
        }
        auto* p_task = reinterpret_cast<decx::core::TaskImplHandle_t>(task_hdlr->_task_impl);
        decx::core::InsertTaskByID(p_task, task_hdlr->_usage, task_hdlr->_slot_id);
        return 0;
    }


    static int32_t TaskPostProcSet(TaskHandle_t* task_hdlr, const decx::core::TaskPostProcHandle_t* p_hdlr)
    {
        if (nullptr == task_hdlr) {
            return -1;
        }
        auto* p_task = reinterpret_cast<decx::core::TaskImplHandle_t>(task_hdlr->_task_impl);
        p_task->SetPostProcCb(p_hdlr);
        return 0;
    }
}
}

#endif