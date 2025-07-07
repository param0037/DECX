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
#include <Array/Fixed_Length_Array.h>


namespace decx
{
namespace utils
{
    class ComputeLoadsMgr;
}
}


class decx::utils::ComputeLoadsMgr
{
private:
    decx::utils::Fixed_Length_Array<decx::core::TaskHandle_t> _task_arr;

public:
    void SetMaxThreadNum(const uint32_t max_thread_num)
    {
        this->_task_arr.PreMalloc(max_thread_num);
    }


    ComputeLoadsMgr() {}


    ComputeLoadsMgr(const int32_t max_thread_num)
    {
        this->SetMaxThreadNum(max_thread_num);
    }


    template <typename FuncType, typename ... ArgTypes> inline
    int32_t AppendTask(FuncType&& f, ArgTypes&& ...args)
    {
        this->_task_arr.Space();
        return decx::core::TaskCreate(this->_task_arr.back(), std::forward<FuncType>(f), std::forward<ArgTypes>(args)...);
    }


    int32_t Run(const uint2& range)
    {
        for (int32_t i = range.x; i < range.y; ++i){
            decx::core::TaskRun(&this->_task_arr[i]);
        }
        return 0;
    }


    int32_t RunAll()
    {
        return this->Run(make_uint2(0, this->_task_arr.size()));
    }


    int32_t Synchronize(const uint2& range)
    {
        int32_t rval = 0;
        for (int32_t i = range.x; i < range.y; ++i){
            rval |= decx::core::TaskSync(&this->_task_arr[i]);
        }
        return rval;
    }


    int32_t SynchronizeAll()
    {
        return this->Synchronize(make_uint2(0, this->_task_arr.size()));
    }


    int32_t ClearAll()
    {
        int32_t rval = 0;
        for (int32_t i = 0; i < this->_task_arr.size(); ++i){
            rval |= decx::core::TaskDestroy(&this->_task_arr[i]);
        }
        this->_task_arr.clear();
        return rval;
    }


    decx::core::TaskHandle_t* Back()
    {
        return this->_task_arr.back();
    }


    ~ComputeLoadsMgr()
    {
        this->_task_arr.clear();
    }
};


#endif
