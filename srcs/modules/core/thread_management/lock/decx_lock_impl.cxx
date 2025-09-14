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

#include "decx_lock_impl.h"
#include "decx_utils_functions.h"
#include <log_console.h>


#define Tolerance_Predicator(Tol, current_val, target, action) {        \
    if_opt ((Tol) == DecxWaitTolerance_e::WaitTol_GreaterEqual){        \
        if ((current_val) >= (target)) {action;}                        \
    }                                                                   \
    else if_opt ((Tol) == DecxWaitTolerance_e::WaitTol_LessEqual) {     \
        if ((current_val) <= (target)) {action;}                        \
    }                                                                   \
    else if_opt ((Tol) == DecxWaitTolerance_e::WaitTol_Equal) {         \
        if ((current_val) == (target)) {action;}                        \
    }                                                                   \
}


namespace decx
{
namespace core
{
template <DecxWaitTolerance_e Tol> DecxWaitResult_e 
decx::core::WaitSpin(std::atomic<int32_t>&  predicate, 
                    const int32_t           desired,
                    const uint64_t          timeout_msec, 
                    const uint32_t          max_spin_num,
                    const uint32_t          spin_factor_exp)
{
    auto start_inst = std::chrono::steady_clock::now();
    uint32_t spin_num = 0;
    const uint32_t spin_gap = (1 << spin_factor_exp) - 1;
    for (;;)
    {
        if ((spin_num & spin_gap) == 0) {
            auto now_inst = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now_inst - start_inst);
            if (duration.count() > timeout_msec){
                return DecxWait_Timeout;
            }
        }

        if (spin_num > max_spin_num){
            return DecxWait_SpinOver;
        }

        int32_t current_val = predicate.load(std::memory_order_relaxed);
        Tolerance_Predicator(Tol, current_val, desired, break);

        _mm_pause();
        ++spin_num;
    }
    return DecxWait_Success;
}

template DecxWaitResult_e decx::core::WaitSpin<DecxWaitTolerance_e::WaitTol_GreaterEqual>(std::atomic<int32_t>&, const int32_t, 
    const uint64_t, const uint32_t, const uint32_t);

template DecxWaitResult_e decx::core::WaitSpin<DecxWaitTolerance_e::WaitTol_LessEqual>(std::atomic<int32_t>&, const int32_t, 
    const uint64_t, const uint32_t, const uint32_t);

template DecxWaitResult_e decx::core::WaitSpin<DecxWaitTolerance_e::WaitTol_Equal>(std::atomic<int32_t>&, const int32_t, 
    const uint64_t, const uint32_t, const uint32_t);


template <DecxWaitTolerance_e Tol> DecxWaitResult_e 
decx::core::WaitRelaxed(std::atomic<int32_t>& predicate, 
                        const int32_t         desired,
                        const uint64_t        timeout_msec)
{
    timespec timeout = {
        .tv_sec = long(timeout_msec / 1000),
        .tv_nsec = long(timeout_msec % 1000) * 1000000L};
    
    // Check first
    int32_t current_val = predicate.load(std::memory_order_acquire);
    if (current_val >= desired){
        return DecxWait_Success;
    }

    int rval = 0;
    timespec* p_timeout = (timeout_msec < DECX_WAIT_FOREVER) ? &timeout : nullptr;
    
    for (;;){
        // Re-check the predicate before each futex call
        current_val = predicate.load(std::memory_order_acquire);
        Tolerance_Predicator(Tol, current_val, desired, return DecxWait_Success);
        
        rval = syscall(SYS_futex, &predicate, FUTEX_WAIT, current_val, p_timeout, nullptr, 0);
        
        switch (rval)
        {
        case 0:
            // Futex call succeeded, check if condition is met
            current_val = predicate.load(std::memory_order_acquire);
            Tolerance_Predicator(Tol, current_val, desired, return DecxWait_Success);
            // Condition not met, continue waiting
            break;
            
        case -1:
            switch (errno)
            {
            case ETIMEDOUT:
                return DecxWait_Timeout;
                
            case EAGAIN:
                // Value changed between load and futex call, retry
                current_val = predicate.load(std::memory_order_acquire);
                Tolerance_Predicator(Tol, current_val, desired, return DecxWait_Success);
                // Continue waiting with new value
                break;
            
            default:
                return DecxWait_UnexpectedError;
            }
            break;
            
        default:
            return DecxWait_UnexpectedError;
        }
    }
}

template DecxWaitResult_e decx::core::WaitRelaxed<DecxWaitTolerance_e::WaitTol_GreaterEqual>(std::atomic<int32_t>&, const int32_t, 
    const uint64_t);

template DecxWaitResult_e decx::core::WaitRelaxed<DecxWaitTolerance_e::WaitTol_LessEqual>(std::atomic<int32_t>&, const int32_t, 
    const uint64_t);

template DecxWaitResult_e decx::core::WaitRelaxed<DecxWaitTolerance_e::WaitTol_Equal>(std::atomic<int32_t>&, const int32_t, 
    const uint64_t);


template <DecxWaitTolerance_e Tol> DecxWaitResult_e 
decx::core::WaitHybrid(std::atomic<int32_t>& predicate,
                       const int32_t         desired,
                       const uint64_t        timeout_msec, 
                       const uint32_t        max_spin_num,
                       const uint32_t        spin_factor_exp)
{
    timespec timeout = {
        .tv_sec = long(timeout_msec / 1000),
        .tv_nsec = long(timeout_msec % 1000) * 1000000L};

    const uint32_t spin_gap = (1 << spin_factor_exp) - 1;
    
    // Fast path: check if already reached target
    int32_t current_val = predicate.load(std::memory_order_acquire);
    Tolerance_Predicator(Tol, current_val, desired, return DecxWait_Success);

    int rval = 0;
    timespec* p_timeout = (timeout_msec < DECX_WAIT_FOREVER) ? &timeout : nullptr;
    
    // Use a more efficient waiting strategy
    int32_t last_checked = current_val;
    
    for (;;){
        // Spin for a while before using futex
        for (int32_t i = 0; i < max_spin_num; ++i) {
            current_val = predicate.load(std::memory_order_acquire);
            Tolerance_Predicator(Tol, current_val, desired, return DecxWait_Success);
            if ((i & spin_gap) == 0) {
                _mm_pause();  // CPU pause instruction
            }
        }
        
        // Only use futex if we haven't made progress
        if (current_val == last_checked) {
            rval = syscall(SYS_futex, &predicate, FUTEX_WAIT, current_val, p_timeout, nullptr, 0);
            
            switch (rval)
            {
            case 0:
                // Futex call succeeded, check if condition is met
                current_val = predicate.load(std::memory_order_acquire);
                Tolerance_Predicator(Tol, current_val, desired, return DecxWait_Success);
                last_checked = current_val;
                break;
                
            case -1:
                switch (errno)
                {
                case ETIMEDOUT:
                    return DecxWait_Timeout;
                    
                case EAGAIN:
                    // Value changed, continue with new value
                    current_val = predicate.load(std::memory_order_acquire);
                    Tolerance_Predicator(Tol, current_val, desired, return DecxWait_Success);
                    last_checked = current_val;
                    break;
                
                default:
                    return DecxWait_UnexpectedError;
                }
                break;
                
            default:
                return DecxWait_UnexpectedError;
            }
        } 
        else {
            last_checked = current_val;
        }
    }
}

template DecxWaitResult_e decx::core::WaitHybrid<DecxWaitTolerance_e::WaitTol_GreaterEqual>(std::atomic<int32_t>&, const int32_t, 
    const uint64_t, const uint32_t, const uint32_t);

template DecxWaitResult_e decx::core::WaitHybrid<DecxWaitTolerance_e::WaitTol_LessEqual>(std::atomic<int32_t>&, const int32_t, 
    const uint64_t, const uint32_t, const uint32_t);

template DecxWaitResult_e decx::core::WaitHybrid<DecxWaitTolerance_e::WaitTol_Equal>(std::atomic<int32_t>&, const int32_t, 
    const uint64_t, const uint32_t, const uint32_t);


decx::core::LockImpl::LockImpl()
{
    this->_init = 1;
    this->_flag.store((int32_t)LockStatus_e::LockStatus_Released, std::memory_order_release);
    std::atomic_thread_fence(std::memory_order_seq_cst);
}


DecxWaitResult_e decx::core::LockImpl::Acquire(const DecxWaitSettings_t* p_settings)
{
    DecxWaitResult_e rval = DecxWait_Success;
    switch (p_settings->_option)
    {
    case DecxWaitOpt_Spin:
        rval = decx::core::WaitSpin<DecxWaitTolerance_e::WaitTol_Equal>(this->_flag, (int32_t)LockStatus_e::LockStatus_Released, 
            p_settings->_timeout_msec, p_settings->_max_spin_cnt, p_settings->_spin_factor_exp);
        break;

    case DecxWaitOpt_Relaxed:
        rval = decx::core::WaitRelaxed<DecxWaitTolerance_e::WaitTol_Equal>(this->_flag, (int32_t)LockStatus_e::LockStatus_Released, 
            p_settings->_timeout_msec);
        break;

    case DecxWaitOpt_Hybrid:
        rval = decx::core::WaitHybrid<DecxWaitTolerance_e::WaitTol_Equal>(this->_flag, (int32_t)LockStatus_e::LockStatus_Released, 
            p_settings->_timeout_msec, p_settings->_max_spin_cnt, p_settings->_spin_factor_exp);
        break;
    
    default:
        break;
    }

    if (DecxWait_Success == rval){
        int32_t flag_exp = (int32_t)LockStatus_e::LockStatus_Released;
        if (this->_flag.compare_exchange_strong(flag_exp, 
                                                (int32_t)LockStatus_e::LockStatus_Acquired, 
                                                std::memory_order_release,
                                                std::memory_order_relaxed)) {
            return DecxWait_Success;
        }
        else{
            return DecxWait_RaceFail;
        }
    }
    return rval;
}


decx::core::LockImpl::~LockImpl()
{
    this->_init = 0;
    this->_flag.store(0, std::memory_order_seq_cst);
}


int32_t decx::core::LockImpl::Release()
{
    this->_flag.store((int32_t)LockStatus_e::LockStatus_Released, std::memory_order_release);
    syscall(SYS_futex, &this->_flag, FUTEX_WAKE, 1, nullptr, nullptr, 0);
    return 0;
}

}       // namespace core
}       // namespace decx


_DECX_API_ int32_t DecxCore_LockCreate(DecxLock_t* p_lock)
{
    static_assert(sizeof(DecxLock_t) <= DecxLockImplSizeMax, "Static size check failed, Lock size is too large");
    if (nullptr == p_lock){
        DECX_LOG_ERR("Invalid pointer to lock, since it is NULL");
        return -1;
    }
    new (p_lock->_impl) decx::core::LockImpl();
    return 0;
}


_DECX_API_ int32_t DecxCore_LockDestroy(DecxLock_t* p_lock)
{
    if (nullptr == p_lock){
        DECX_LOG_ERR("Invalid pointer to lock, since it is NULL");
        return -1;
    }
    auto* p_impl = (decx::core::LockImpl*)p_lock->_impl;
    p_impl->~LockImpl();
    return 0;
}


_DECX_API_ DecxWaitResult_e DecxCore_LockAcquire(DecxLock_t* p_lock, const DecxWaitSettings_t* p_settings)
{
    if (nullptr == p_lock){
        DECX_LOG_ERR("Invalid pointer to lock, since it is NULL");
        return DecxWait_UnexpectedError;
    }
    auto* p_impl = (decx::core::LockImpl*)p_lock->_impl;
    return p_impl->Acquire(p_settings);
}


_DECX_API_ int32_t DecxCore_LockRelease(DecxLock_t* p_lock)
{
    if (nullptr == p_lock){
        DECX_LOG_ERR("Invalid pointer to lock, since it is NULL");
        return -1;
    }
    auto* p_impl = (decx::core::LockImpl*)p_lock->_impl;
    return p_impl->Release();
}