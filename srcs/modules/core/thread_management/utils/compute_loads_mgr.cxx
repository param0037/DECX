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

#include <Concurrent/compute_loads_mgr.h>
#define MODULE_TAG "TaskMgr"


int32_t decx::utils::ComputeLoadsMgr::SetMaxThreadNum(const uint32_t max_thread_num)
{
    int32_t rval = 0;
    if (max_thread_num > this->_max_thread){
        if (this->_task_arr.IsValid()){
            rval |= this->_task_arr.Free();
        }
        rval |= this->_task_arr.Allocate(max_thread_num * sizeof(decx::core::TaskHandle_t), PAGABLE);
        this->_max_thread = max_thread_num;
        this->_valid_thread_num = 0;
    }
    return rval;
}

void decx::utils::ComputeLoadsMgr::PostBarrierCallback(const int32_t argc, void* p_argv_list)
{
    auto* p_mgr = (decx::utils::ComputeLoadsMgr*)p_argv_list;
    if (p_mgr) {
        DecxCore_CountingSemaphorePost(&p_mgr->_barrier_sem, p_mgr->_valid_thread_num);
    }
}

decx::utils::ComputeLoadsMgr::ComputeLoadsMgr()
{
    this->_max_thread = 0;
    this->_valid_thread_num = 0;
    this->_postproc_hdlr = {
        ._p_cb_entry = PostBarrierCallback,
        ._argc = 0,
        ._p_args_list = (void*)this
    };
    if (DecxCore_CountingSemaphoreCreate(&this->_barrier_sem)){
        DECX_LOG_ERR("Semaphore create failed");
    }
}

decx::utils::ComputeLoadsMgr::ComputeLoadsMgr(const int32_t max_thread_num)
{
    this->_postproc_hdlr = {
        ._p_cb_entry = PostBarrierCallback,
        ._argc = 0,
        ._p_args_list = (void*)this
    };
    this->SetMaxThreadNum(max_thread_num);
    this->_valid_thread_num = 0;
    this->_dispatch_method = decx::core::ThreadDispatchMethod_e::Dispatch_ByID;
    DecxCore_CountingSemaphoreCreate(&this->_barrier_sem);
}


void decx::utils::ComputeLoadsMgr::SetDispatchMethod(const decx::core::ThreadDispatchMethod_e method)
{
    this->_dispatch_method = method;
}


int32_t decx::utils::ComputeLoadsMgr::Run(const uint2& range)
{
    DecxCore_CountingSemaphoreReset(&this->_barrier_sem);
    if (range.y > this->_valid_thread_num){
        DECX_LOG_ERR("Range out of boundary: range.y=%d, valid thread num=%d", range.y, this->_valid_thread_num);
        return -1;
    }
    int32_t rval = 0;
    const uint32_t num = range.y - range.x;
    
    uint8_t need_reset_notify_cnt = (num == this->_valid_thread_num);
    for (int32_t i = range.x; i < range.y; ++i){
        rval |= decx::core::TaskRun(&this->_task_arr[i]);
    }
    return rval;
}


int32_t decx::utils::ComputeLoadsMgr::RunAll()
{
    return this->Run(make_uint2(0, this->_valid_thread_num));
}


int32_t decx::utils::ComputeLoadsMgr::Synchronize(const uint2& range)
{
    int32_t rval = 0;
    if (range.y > this->_valid_thread_num){
        return -1;
    }

    int32_t num = range.y - range.x;
    
    DecxWaitSettings_t wait_settings = {
        ._option = DecxWaitOpt_Hybrid,
        ._max_spin_cnt = 1000,
        ._spin_factor_exp = 4,
        ._timeout_msec = DECX_WAIT_FOREVER
    };
    rval = (int32_t)DecxCore_CountingSemaphoreWait(&this->_barrier_sem, &wait_settings, num);
    return rval;
}


int32_t decx::utils::ComputeLoadsMgr::SynchronizeAll()
{
    return this->Synchronize(make_uint2(0, this->_valid_thread_num));
}


int32_t decx::utils::ComputeLoadsMgr::ClearAll()
{
    int32_t rval = 0;
    for (int32_t i = 0; i < this->_valid_thread_num; ++i){
        rval |= decx::core::TaskDestroy(&this->_task_arr[i]);
    }
    this->_valid_thread_num = 0;
    return rval;
}


decx::core::TaskHandle_t* decx::utils::ComputeLoadsMgr::Back()
{
    return this->_task_arr + this->_valid_thread_num;
}


decx::utils::ComputeLoadsMgr::~ComputeLoadsMgr()
{
    this->_task_arr.Free();
    this->_valid_thread_num = 0;
    this->_max_thread = 0;
}


int32_t decx::utils::ComputeLoadsMgr2D::Reshape(const uint2& new_dist)
{
    this->_thread_dist = new_dist;
    return ComputeLoadsMgr::SetMaxThreadNum(new_dist.x * new_dist.y);
}


int32_t decx::utils::ComputeLoadsMgr2D::Run(const uint2& range_x, const uint2& range_y)
{
    int32_t rval = 0;
    for (int32_t i = range_y.x; i < range_y.y; ++i){
        rval |= ComputeLoadsMgr::Run(make_uint2(i * this->_thread_dist.x + range_x.x, i * this->_thread_dist.x + range_x.y));
    }
    return rval;
}


int32_t decx::utils::ComputeLoadsMgr2D::Synchronize(const uint2& range_x, const uint2& range_y)
{
    int32_t rval = 0;
    int32_t num = (range_x.y - range_x.x) * (range_y.y - range_y.x);
    
    DecxWaitSettings_t wait_settings = {
        ._option = DecxWaitOpt_Hybrid,
        ._max_spin_cnt = 1000,
        ._spin_factor_exp = 4,
        ._timeout_msec = DECX_WAIT_FOREVER
    };
    rval = (int32_t)DecxCore_CountingSemaphoreWait(&this->_barrier_sem, &wait_settings, num);
    return rval;
}


int32_t decx::utils::ComputeLoadsMgr2D::RunAll()
{
    int32_t rval = 0;
    for (int32_t i = 0; i < this->_thread_dist.x * this->_thread_dist.y; ++i){
        rval |= decx::core::TaskRun(&this->_task_arr[i]);
    }
    return rval;
}


int32_t decx::utils::ComputeLoadsMgr2D::SynchronizeAll()
{
    int32_t rval = 0;
    this->Synchronize(make_uint2(0, this->_thread_dist.x), make_uint2(0, this->_thread_dist.y));
    return rval;
}


int32_t decx::utils::ComputeLoadsMgr2D::AdvisedReshape(const uint32_t total_thr_num, const uint2 proc_dims)
{
    uint2 advised_dist = make_uint2(0, 0);
    if (total_thr_num > 1) {
        const float k = (float)proc_dims.x / (float)proc_dims.y;
        const uint32_t base = roundf(sqrtf((float)total_thr_num / k));

        if (base < 2) {
            advised_dist = make_uint2(total_thr_num, 1);
        }
        else if (base > total_thr_num) {
            advised_dist = make_uint2(1, total_thr_num);
        }
        else {
            uint32_t base_var_ni = base;
            uint32_t base_var_nz = base;
            // Towards infinity
            while (total_thr_num % base_var_ni) {
                ++base_var_ni;
            }
            // Towards zero
            while (total_thr_num % base_var_nz) {
                --base_var_nz;
            }
            // Get the differences and find the minimal
            uint32_t diff_ni = fabs(base - base_var_ni);
            uint32_t diff_nz = fabs(base - base_var_nz);
            uint32_t candidate_base = diff_ni > diff_nz ? base_var_nz : base_var_ni;
            advised_dist.y = candidate_base;
            advised_dist.x = total_thr_num / candidate_base;
        }
    }
    else {
        advised_dist = make_uint2(1, 1);
    }

    return this->Reshape(advised_dist);
}


decx::utils::ComputeLoadsMgr2D::~ComputeLoadsMgr2D()
{
    // ComputeLoadsMgr::~ComputeLoadsMgr();
}