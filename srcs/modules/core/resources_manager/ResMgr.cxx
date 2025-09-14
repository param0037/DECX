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


#include "decx_resource.h"


decx::ResMgr::ResMgr()
{
    this->_run = true;
    this->_last_res_num = 0;
    this->_shortest_wait_period = 0x7fffffffffffffff;

    decx::core::TaskQueueInfo_t desired_form = {
        ._switch    = decx::core::TaskQueueSwitch::TaskQueue_ON,
        ._behaviour = decx::core::TaskQueueBehaviour_e::TaskQueue_LIFO,
        ._usage     = decx::core::TaskQueueUsage_e::TaskQueue_ResMgr,
        ._task_num  = 0,
    };

    DecxCore_BinarySemaphoreCreate(&this->_sem);
    DecxCore_LockCreate(&this->_lock);

    int32_t slot_id = decx::core::TaskQueueQuery(&desired_form);
    if (slot_id == -1){
        slot_id = decx::core::ThreadpoolAddSot(&desired_form);
    }

    decx::core::TaskCreate(decx::core::ThreadDispatchMethod_e::Dispatch_ByID, desired_form._usage, slot_id, &this->_task, decx::ResMgr::__ResMgrTask, this);
    decx::core::TaskRun(&this->_task);

    this->_wp._outer_info = this;
}


_THREAD_FUNCTION_
void decx::ResMgr::__ResMgrTask(decx::ResMgr* fake_this)
{
    fake_this->_shortest_wait_period = (long long)100;
    fake_this->_last_res_num = fake_this->_res_arr.size();

    DecxWaitSettings_t sem_wst = {
        ._option = DecxWaitOpt_Relaxed,
        ._max_spin_cnt = 1000,
        ._spin_factor_exp = 4,
        ._timeout_msec = DECX_WAIT_FOREVER,
    };

    DecxWaitSettings_t lock_wst = {
        ._option = DecxWaitOpt_Hybrid,
        ._max_spin_cnt = 1000,
        ._spin_factor_exp = 4,
        ._timeout_msec = DECX_WAIT_FOREVER,
    };

    while (fake_this->_run)
    {
        time_t _current;
        time(&_current);

        fake_this->_shortest_wait_period = 0x7fffffffffffffff;
        for (uint32_t i = 0; i < fake_this->_res_arr.size(); ++i) 
        {
            decx::Resource* res_ptr = fake_this->_res_arr + i;
            
            if (res_ptr->exceeded_lifespan(_current)) {
                DecxCore_LockAcquire(&fake_this->_lock, &lock_wst);
                if (res_ptr->Delete()) {
                    fake_this->_res_arr.del(i);
                }
                DecxCore_LockRelease(&fake_this->_lock);
            }
            else{
                fake_this->_shortest_wait_period = min(fake_this->_shortest_wait_period,
                    res_ptr->get_lifespan() - _current + res_ptr->get_last_used_instant());
            }
        }

        fake_this->_last_res_num = fake_this->_res_arr.size();

        sem_wst._timeout_msec = fake_this->_shortest_wait_period * 1000;
        DecxCore_BinarySemaphoreWait(&fake_this->_sem, &sem_wst);
    }
}


uint64_t decx::ResMgr::checkin(void** exposed_ptr, const time_t lifespan, 
    res_release_fn _decon)
{
    DecxWaitSettings_t lock_wst = {
        ._option = DecxWaitOpt_Hybrid,
        ._max_spin_cnt = 1000,
        ._spin_factor_exp = 4,
        ._timeout_msec = DECX_WAIT_FOREVER,
    };

    DecxCore_LockAcquire(&this->_lock, &lock_wst);
    this->_res_arr.emplace_back(exposed_ptr, lifespan, _decon);
    DecxCore_LockRelease(&this->_lock);
    DecxCore_BinarySemaphorePost(&this->_sem);
    return this->_res_arr.size() - 1;
}


void decx::ResMgr::lock_resource(const uint64_t res_id)
{
    DecxWaitSettings_t lock_wst = {
        ._option = DecxWaitOpt_Hybrid,
        ._max_spin_cnt = 1000,
        ._spin_factor_exp = 4,
        ._timeout_msec = DECX_WAIT_FOREVER,
    };

    DecxCore_LockAcquire(&this->_lock, &lock_wst);
    this->_res_arr[res_id].lock();
    DecxCore_LockRelease(&this->_lock);
}


void decx::ResMgr::unlock_resource(const uint64_t res_id)
{
    DecxWaitSettings_t lock_wst = {
        ._option = DecxWaitOpt_Hybrid,
        ._max_spin_cnt = 1000,
        ._spin_factor_exp = 4,
        ._timeout_msec = DECX_WAIT_FOREVER,
    };

    DecxCore_LockAcquire(&this->_lock, &lock_wst);
    this->_res_arr[res_id].unlock();
    DecxCore_LockRelease(&this->_lock);
}



void decx::ResMgr::checkout(const uint64_t res_id)
{
    DecxWaitSettings_t lock_wst = {
        ._option = DecxWaitOpt_Hybrid,
        ._max_spin_cnt = 1000,
        ._spin_factor_exp = 4,
        ._timeout_msec = DECX_WAIT_FOREVER,
    };

    DecxCore_LockAcquire(&this->_lock, &lock_wst);
    this->_res_arr.del(res_id);
    DecxCore_LockRelease(&this->_lock);
    DecxCore_BinarySemaphorePost(&this->_sem);
}


decx::ResMgr::~ResMgr()
{
    DecxWaitSettings_t lock_wst = {
        ._option = DecxWaitOpt_Hybrid,
        ._max_spin_cnt = 1000,
        ._spin_factor_exp = 4,
        ._timeout_msec = DECX_WAIT_FOREVER,
    };

    DecxCore_LockAcquire(&this->_lock, &lock_wst);
    this->_run = false;
    DecxCore_LockRelease(&this->_lock);
    
    DecxCore_BinarySemaphorePost(&this->_sem);

    decx::core::TaskDestroy(&this->_task);
}


_DECX_API_ uint64_t decx::ResourceCheckIn(void** exposed_ptr, const time_t lifespan_sec,
    res_release_fn _decon)
{
    return decx::_res_mgr->checkin(exposed_ptr, lifespan_sec, _decon);
}


_DECX_API_ void decx::ResourceLock(const uint64_t res_id)
{
    decx::_res_mgr->lock_resource(res_id);
}


_DECX_API_ void decx::ResourceUnlock(const uint64_t res_id)
{
    decx::_res_mgr->unlock_resource(res_id);
}


_DECX_API_ void decx::ResourceCheckOut(const uint64_t res_id)
{
    decx::_res_mgr->checkout(res_id);
}


decx::ResMgr* decx::_res_mgr;
