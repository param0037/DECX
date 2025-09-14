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
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
* INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR 
* PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDES BE LIABLE 
* FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR 
* OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
* DEALINGS IN THE SOFTWARE.
*/

#include "decx_semaphore_impl.h"
#include <linux/futex.h>
#include <sys/syscall.h>
#include <unistd.h>


namespace decx
{
namespace core
{
    static DecxWaitResult_e WaitGeneric(std::atomic<int32_t> &sem, const DecxWaitSettings_t* p_settings, const int32_t count)
    {
        DecxWaitResult_e rval = DecxWait_Success;

        switch (p_settings->_option)
        {
        case DecxWaitOpt_Spin:
            rval = decx::core::WaitSpin<DecxWaitTolerance_e::WaitTol_Equal>(sem, count, p_settings->_timeout_msec,
                p_settings->_max_spin_cnt, p_settings->_spin_factor_exp);
            break;

        case DecxWaitOpt_Relaxed:
            rval = decx::core::WaitRelaxed<DecxWaitTolerance_e::WaitTol_Equal>(sem, count, p_settings->_timeout_msec);
            break;

        case DecxWaitOpt_Hybrid:
            rval = decx::core::WaitHybrid<DecxWaitTolerance_e::WaitTol_Equal>(sem, count, p_settings->_timeout_msec, 
                p_settings->_max_spin_cnt, p_settings->_spin_factor_exp);
            break;
        
        default:
            break;
        }

        int32_t sem_exp = count;
        sem.compare_exchange_strong(sem_exp, 0, std::memory_order_release, std::memory_order_relaxed);

        return rval;
    }
}
}



DecxWaitResult_e
decx::core::SemaphoreImpl_Binary::Wait(const DecxWaitSettings_t* p_settings)
{
    this->_wait_num.fetch_add(1, std::memory_order_release);
    auto rval = decx::core::WaitGeneric(this->_sem, p_settings, 1);
    this->_wait_num.fetch_sub(1, std::memory_order_release);
    return rval;
}


DecxWaitResult_e
decx::core::SemaphoreImpl_Counting::Wait(const DecxWaitSettings_t* p_settings, const int32_t count)
{
    this->_wait_num.fetch_add(1, std::memory_order_release);
    auto rval = decx::core::WaitGeneric(this->_sem, p_settings, count);
    this->_wait_num.fetch_sub(1, std::memory_order_release);
    return rval;
}


int32_t decx::core::SemaphoreImpl_Binary::Post()
{
    int32_t flag_exp = 0;
    if (this->_sem.compare_exchange_strong(flag_exp, 
        1, 
        std::memory_order_release,
        std::memory_order_relaxed)) {
        syscall(SYS_futex, &this->_sem, FUTEX_WAKE, this->_wait_num.load(std::memory_order_acquire), nullptr, nullptr, 0);
    }
    return 0;
}


int32_t decx::core::SemaphoreImpl_Counting::Post(const int32_t notify_cnt)
{
    int32_t old_val = this->_sem.fetch_add(1, std::memory_order_release);
    int32_t new_val = old_val + 1;
    // Wake the waiting thread when we reach the target count
    if (new_val >= notify_cnt) {
        syscall(SYS_futex, &this->_sem, FUTEX_WAKE, this->_wait_num.load(std::memory_order_acquire), nullptr, nullptr, 0);
    }
    
    return 0;
}


_DECX_API_ int32_t DecxCore_BinarySemaphoreCreate(DecxBinarySemaphore_t* p_sem)
{
    if (nullptr == p_sem){
        DECX_LOG_ERR("Invalid pointer to lock, since it is NULL");
        return -1;
    }
    auto* p_impl = (decx::core::SemaphoreImpl_Binary*)p_sem->_impl;
    static_assert(sizeof(decx::core::SemaphoreImpl_Binary) <= DecxSemImplSizeMax, 
        "Static size check failed, binary semaphore size is too large");
    new (p_sem->_impl) decx::core::SemaphoreImpl_Binary();
    return 0;
}


_DECX_API_ int32_t DecxCore_CountingSemaphoreCreate(DecxCountingSemaphore_t* p_sem)
{
    if (nullptr == p_sem){
        DECX_LOG_ERR("Invalid pointer to lock, since it is NULL");
        return -1;
    }
    auto* p_impl = (decx::core::SemaphoreImpl_Counting*)p_sem->_impl;
    static_assert(sizeof(decx::core::SemaphoreImpl_Counting) <= DecxSemImplSizeMax, 
        "Static size check failed, countingg semaphore size is too large");
    new (p_sem->_impl) decx::core::SemaphoreImpl_Counting();
    return 0;
}


_DECX_API_ int32_t DecxCore_BinarySemaphoreDestroy(DecxBinarySemaphore_t* p_sem)
{
    if (nullptr == p_sem){
        DECX_LOG_ERR("Invalid pointer to semaphore, since it is NULL");
        return -1;
    }
    auto* p_impl = (decx::core::SemaphoreImpl_Binary*)p_sem->_impl;
    p_impl->~SemaphoreImpl_Binary();
    return 0;
}


_DECX_API_ int32_t DecxCore_CountingSemaphoreDestroy(DecxCountingSemaphore_t* p_sem)
{
    if (nullptr == p_sem){
        DECX_LOG_ERR("Invalid pointer to semaphore, since it is NULL");
        return -1;
    }
    auto* p_impl = (decx::core::SemaphoreImpl_Counting*)p_sem->_impl;
    p_impl->~SemaphoreImpl_Counting();
    return 0;
}


_DECX_API_ DecxWaitResult_e 
DecxCore_BinarySemaphoreWait(DecxBinarySemaphore_t* p_sem, const DecxWaitSettings_t* p_settings)
{
    if (nullptr == p_sem){
        DECX_LOG_ERR("Invalid pointer to semaphore, since it is NULL");
        return DecxWait_UnexpectedError;
    }
    auto* p_impl = (decx::core::SemaphoreImpl_Binary*)p_sem->_impl;
    auto rval = p_impl->Wait(p_settings);
    return rval;
}


_DECX_API_ DecxWaitResult_e 
DecxCore_CountingSemaphoreWait(DecxCountingSemaphore_t* p_sem, const DecxWaitSettings_t* p_settings, int32_t desired_cnt)
{
    if (nullptr == p_sem){
        DECX_LOG_ERR("Invalid pointer to semaphore, since it is NULL");
        return DecxWait_UnexpectedError;
    }
    auto* p_impl = (decx::core::SemaphoreImpl_Counting*)p_sem->_impl;
    auto rval = p_impl->Wait(p_settings, desired_cnt);
    return rval;
}


_DECX_API_ int32_t DecxCore_BinarySemaphorePost(DecxBinarySemaphore_t* p_sem)
{
    if (nullptr == p_sem){
        DECX_LOG_ERR("Invalid pointer to semaphore, since it is NULL");
        return DecxWait_UnexpectedError;
    }
    auto* p_impl = (decx::core::SemaphoreImpl_Binary*)p_sem->_impl;
    auto rval = p_impl->Post();
    return rval;
}


_DECX_API_ int32_t DecxCore_CountingSemaphorePost(DecxCountingSemaphore_t* p_sem, int32_t notify_cnt)
{
    if (nullptr == p_sem){
        DECX_LOG_ERR("Invalid pointer to semaphore, since it is NULL");
        return DecxWait_UnexpectedError;
    }
    auto* p_impl = (decx::core::SemaphoreImpl_Counting*)p_sem->_impl;
    auto rval = p_impl->Post(notify_cnt);
    return rval;
}


_DECX_API_ int32_t DecxCore_BinarySemaphoreReset(DecxBinarySemaphore_t* p_sem)
{
    if (nullptr == p_sem){
        DECX_LOG_ERR("Invalid pointer to semaphore, since it is NULL");
        return DecxWait_UnexpectedError;
    }
    auto* p_impl = (decx::core::SemaphoreImpl_Binary*)p_sem->_impl;
    p_impl->Reset();
    return 0;
}


_DECX_API_ int32_t DecxCore_CountingSemaphoreReset(DecxCountingSemaphore_t* p_sem)
{
    if (nullptr == p_sem){
        DECX_LOG_ERR("Invalid pointer to semaphore, since it is NULL");
        return DecxWait_UnexpectedError;
    }
    auto* p_impl = (decx::core::SemaphoreImpl_Counting*)p_sem->_impl;
    p_impl->Reset();
    return 0;
}