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

#ifndef _COMPUTE_LOADS_MGR_H_
#define _COMPUTE_LOADS_MGR_H_

#include <basic.h>
#include "task_handle.h"
#include <PtrInfo.h>
#include <Concurrent/semaphore.h>

namespace decx
{
namespace utils
{
    class ComputeLoadsMgr;


    class ComputeLoadsMgr2D;
}
}


class _DECX_API_ decx::utils::ComputeLoadsMgr
{
protected:
    decx::PtrInfo<decx::core::TaskHandle_t> _task_arr;
    uint32_t                                _max_thread;
    uint32_t                                _valid_thread_num;
    decx::core::ThreadDispatchMethod_e      _dispatch_method;
    DecxCountingSemaphore_t                 _barrier_sem;
    decx::core::TaskPostProcHandle_t        _postproc_hdlr;

private:
    static void PostBarrierCallback(const int32_t argc, void* p_argv_list);

public:
    int32_t RunAll();


    int32_t SynchronizeAll();

public:
    int32_t SetMaxThreadNum(const uint32_t max_thread_num);


    ComputeLoadsMgr();


    ComputeLoadsMgr(const int32_t max_thread_num);


    void SetDispatchMethod(const decx::core::ThreadDispatchMethod_e method);


    template <typename FuncType, typename ... ArgTypes> inline
    int32_t AppendTask(const int32_t slot_id, FuncType&& f, ArgTypes&& ...args)
    {
        int32_t rval = decx::core::TaskCreate(this->_dispatch_method, decx::core::TaskQueueUsage_e::TaskQueue_CalcLoad, slot_id,
            this->_task_arr + this->_valid_thread_num, std::forward<FuncType>(f), std::forward<ArgTypes>(args)...);
        
        decx::core::TaskPostProcSet(this->_task_arr + this->_valid_thread_num, &this->_postproc_hdlr);
        ++this->_valid_thread_num;
        return rval;
    }


    int32_t Run(const uint2& range);


    int32_t Synchronize(const uint2& range);


    int32_t ClearAll();


    decx::core::TaskHandle_t* Back();


    ~ComputeLoadsMgr();
};


class _DECX_API_ decx::utils::ComputeLoadsMgr2D : public decx::utils::ComputeLoadsMgr
{
private:
    uint2 _thread_dist;

public:
    int32_t RunAll();


    int32_t SynchronizeAll();

public:
    ComputeLoadsMgr2D() {}


    ComputeLoadsMgr2D(const uint2& dist) : ComputeLoadsMgr(dist.x * dist.y) 
    {
        this->_thread_dist = dist;
    }


    int32_t Reshape(const uint2& new_dist);


    int32_t Run(const uint2& range_x, const uint2& range_y);


    int32_t Synchronize(const uint2& range_x, const uint2& range_y);


    const uint2& GetDist() const
    {
        return this->_thread_dist;
    }


    int32_t AdvisedReshape(const uint32_t total_thr_num, const uint2 proc_dims);


    ~ComputeLoadsMgr2D();
};


#endif
