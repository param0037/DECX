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


namespace decx
{
namespace utils
{
    class ComputeLoadsMgr;


    class ComputeLoadsMgr2D;
}
}


class decx::utils::ComputeLoadsMgr
{
private:
    decx::PtrInfo<decx::core::TaskHandle_t> _task_arr;
    uint32_t _max_thread;
    uint32_t _valid_thread_num;
    decx::core::ThreadDispatchMethod_e _dispatch_method;

public:
    int32_t SetMaxThreadNum(const uint32_t max_thread_num)
    {
        int32_t rval = 0;
        if (this->_task_arr.IsValid()){
            rval |= this->_task_arr.Free();
        }
        rval |= this->_task_arr.Allocate(max_thread_num, PAGABLE);
        this->_max_thread = max_thread_num;
        this->_valid_thread_num = 0;
        return rval;
    }


    int32_t Resize(const uint32_t max_thread_num)
    {
        if (max_thread_num > this->_max_thread){
            return this->SetMaxThreadNum(max_thread_num);
        }
        return 0;
    }


    ComputeLoadsMgr() {}


    ComputeLoadsMgr(const int32_t max_thread_num)
    {
        this->SetMaxThreadNum(max_thread_num);
    }


    void SetDispatchMethod(const decx::core::ThreadDispatchMethod_e method)
    {
        this->_dispatch_method = method;
    }


    template <typename FuncType, typename ... ArgTypes> inline
    int32_t AppendTask(const int32_t slot_id, FuncType&& f, ArgTypes&& ...args)
    {
        int32_t rval = decx::core::TaskCreate(this->_dispatch_method, slot_id,
            this->_task_arr + this->_valid_thread_num, std::forward<FuncType>(f), std::forward<ArgTypes>(args)...);
        ++this->_valid_thread_num;
        return rval;
    }


    int32_t Run(const uint2& range)
    {
        if (range.y > this->_valid_thread_num){
            return -1;
        }
        int32_t rval = 0;
        for (int32_t i = range.x; i < range.y; ++i){
            rval |= decx::core::TaskRun(&this->_task_arr[i]);
        }
        return rval;
    }


    int32_t RunAll()
    {
        return this->Run(make_uint2(0, this->_valid_thread_num));
    }


    int32_t Synchronize(const uint2& range)
    {
        int32_t rval = 0;
        if (range.y > this->_valid_thread_num){
            return -1;
        }
        for (int32_t i = range.x; i < range.y; ++i){
            rval |= decx::core::TaskSync(&this->_task_arr[i]);
        }
        return rval;
    }


    int32_t SynchronizeAll()
    {
        return this->Synchronize(make_uint2(0, this->_valid_thread_num));
    }


    int32_t ClearAll()
    {
        int32_t rval = 0;
        for (int32_t i = 0; i < this->_valid_thread_num; ++i){
            rval |= decx::core::TaskDestroy(&this->_task_arr[i]);
        }
        return rval;
    }


    decx::core::TaskHandle_t* Back()
    {
        return this->_task_arr + this->_valid_thread_num;
    }


    ~ComputeLoadsMgr()
    {
        this->_task_arr.Free();
        this->_valid_thread_num = 0;
        this->_max_thread = 0;
    }
};


class decx::utils::ComputeLoadsMgr2D : public decx::utils::ComputeLoadsMgr
{
private:
    uint2 _thread_dist;

public:
    ComputeLoadsMgr2D() {}


    ComputeLoadsMgr2D(const uint2& dist) : ComputeLoadsMgr(dist.x * dist.y) 
    {
        this->_thread_dist = dist;
    }


    int32_t Reshape(const uint2& new_dist)
    {
        this->_thread_dist = new_dist;
        return ComputeLoadsMgr::Resize(new_dist.x * new_dist.y);
    }


    int32_t Run(const uint2& range_x, const uint2& range_y)
    {
        int32_t rval = 0;
        for (int32_t i = range_y.x; i < range_y.y; ++i){
            rval |= ComputeLoadsMgr::Run(make_uint2(i * this->_thread_dist.x + range_x.x, i * this->_thread_dist.x + range_x.y));
        }
        return rval;
    }


    int32_t Synchronize(const uint2& range_x, const uint2& range_y)
    {
        int32_t rval = 0;
        for (int32_t i = range_y.x; i < range_y.y; ++i){
            rval |= ComputeLoadsMgr::Synchronize(make_uint2(i * this->_thread_dist.x + range_x.x, i * this->_thread_dist.x + range_x.y));
        }
        return rval;
    }


    const uint2& GetDist() const
    {
        return this->_thread_dist;
    }
};


#endif
