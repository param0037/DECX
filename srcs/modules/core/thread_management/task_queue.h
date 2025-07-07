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


namespace decx
{
namespace core
{
    class ThreadTaskQueue;


    enum class TaskQueueSwitch : uint8_t
    {
        TaskQueue_OFF = 0,
        TaskQueue_ON = 1,
    };
}
}


class decx::core::ThreadTaskQueue
{
private:
    // private variables for each thread
    std::mutex _mtx;
    std::condition_variable _cv;

    decx::utils::Dynamic_Array<decx::core::TaskImplHandle_t> _task_queue;

    uint8_t _shutdown;

    decx::core::TaskQueueBehaviour_e _behaviour;
    decx::core::TaskQueueUsage_e     _usage;

public:
    ThreadTaskQueue();


    void Switch(const TaskQueueSwitch switch_stage);


    _THREAD_GENERAL_ void ThreadMainLoop();


    std::mutex& GetMutex() {
        return this->_mtx;
    }


    std::condition_variable& GetCondVar() {
        return this->_cv;
    }


    uint64_t GetCurrentTaskNum();


    int32_t RegisterTask(decx::core::TaskImplHandle_t task_hdlr);
};



#endif      // ifndef _TASK_QUEUE_H_