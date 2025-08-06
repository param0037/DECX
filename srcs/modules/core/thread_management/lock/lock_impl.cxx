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

#include "lock_impl.h"


decx::core::SpinLock::SpinLock()
{
    this->_occupied.store(0);
    this->_lock_type = DecxLockType_e::LockType_Spin;
    this->_init = 1;
}


DecxAsync_WaitResult_e decx::core::SpinLock::Lock_Timeout(const uint64_t timeout_msec)
{
    auto start = std::chrono::steady_clock::now();
    uint8_t expected = 0;
    while (!this->_occupied.compare_exchange_strong(expected, 1, std::memory_order_acquire)) {
        // Reset expected to 0 if compare_exchange_strong failed
        expected = 0;
        // Busy-wait loop with pause instruction
        while (this->_occupied.load(std::memory_order_relaxed) == 1) {
            _mm_pause(); // Use this intrinsic for x86 architecture
            auto current_tick = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(current_tick - start);
            if (duration.count() > timeout_msec){
                return DecxAsync_WaitResult_e::Wait_Timeout;
            }
        }
        
    }
    return DecxAsync_WaitResult_e::Wait_Success;
}


DecxAsync_WaitResult_e decx::core::SpinLock::Lock()
{
    uint8_t expected = 0;
    while (!this->_occupied.compare_exchange_strong(expected, 1, std::memory_order_acquire)) {
        // Reset expected to 0 if compare_exchange_strong failed
        expected = 0;
        // Busy-wait loop with pause instruction
        while (this->_occupied.load(std::memory_order_relaxed) == 1) {
            _mm_pause(); // Use this intrinsic for x86 architecture
        }
    }
    return DecxAsync_WaitResult_e::Wait_Success;
}


int32_t decx::core::SpinLock::Unlock()
{
    this->_occupied.store(0);
    return 0;
}


decx::core::TimedMutexLock::TimedMutexLock()
{
    this->_lock_type = DecxLockType_e::LockType_Mutex;
    this->_init = 1;
}


DecxAsync_WaitResult_e decx::core::TimedMutexLock::Lock()
{
    this->_mtx.lock();
    return DecxAsync_WaitResult_e::Wait_Success;
}


DecxAsync_WaitResult_e decx::core::TimedMutexLock::Lock_Timeout(const uint64_t timeout_msec)
{
    if (this->_mtx.try_lock_for(std::chrono::milliseconds(timeout_msec))){
        return DecxAsync_WaitResult_e::Wait_Success;
    }
    return DecxAsync_WaitResult_e::Wait_Timeout;
}


int32_t decx::core::TimedMutexLock::Unlock()
{
    this->_mtx.unlock();
    return 0;
}



int32_t DecxLockCreate(DecxLock_t* p_lock, const DecxLockType_e lock_type)
{
    switch (lock_type)
    {
    case DecxLockType_e::LockType_Spin:
        new (p_lock->_impl) decx::core::SpinLock;
        break;

    case DecxLockType_e::LockType_Mutex:
        new (p_lock->_impl) decx::core::TimedMutexLock;
        break;
    
    default:
        break;
    }
    return 0;
}

DecxAsync_WaitResult_e DecxLockAcquire(DecxLock_t* p_lock, const DecxAsync_WaitOption_e option, const uint64_t timeout_msec)
{
    if (nullptr == p_lock){
        return DecxAsync_WaitResult_e::Wait_UnknownErr;
    }
    decx::core::LockBase* p_lock_impl = (decx::core::LockBase*)p_lock->_impl;
    switch (option)
    {
    case DecxAsync_WaitOption_e::Wait_Forever:
        p_lock_impl->Lock();
        break;

    case DecxAsync_WaitOption_e::Wait_Timeout:
        p_lock_impl->Lock_Timeout(timeout_msec);
        break;
    
    default:
        break;
    }
    return DecxAsync_WaitResult_e::Wait_Success;
}

int32_t DecxLockRelease(DecxLock_t* p_lock)
{
    if (nullptr == p_lock){
        return -1;
    }
    decx::core::LockBase* p_lock_impl = (decx::core::LockBase*)p_lock->_impl;
    return p_lock_impl->Unlock();
}

int32_t DecxLockDestroy(DecxLock_t* p_lock)
{
    if (nullptr == p_lock){
        return -1;
    }
    decx::core::LockBase* p_lock_impl = (decx::core::LockBase*)p_lock->_impl;
    delete p_lock_impl;
    return 0;
}

