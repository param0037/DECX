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

#ifndef _LOCK_IMPL_H_
#define _LOCK_IMPL_H_

#include <basic.h>
#include <atomic>
#include <Concurrent/lock.h>
#include <chrono>
#include <mutex>


namespace decx
{
namespace core
{
    class LockBase;

    class SpinLock;

    class TimedMutexLock;
}
}


class decx::core::LockBase
{
protected:
    DecxLockType_e _lock_type;
    uint8_t _init;

public:
    LockBase() {}


    virtual DecxAsync_WaitResult_e Lock() {
        return DecxAsync_WaitResult_e::Wait_Success;
    }
    // virtual DecxAsync_WaitResult_e Lock() = 0;


    virtual DecxAsync_WaitResult_e Lock_Timeout(const uint64_t timeout_msec) {
        return DecxAsync_WaitResult_e::Wait_Success;
    }
    // virtual DecxAsync_WaitResult_e Lock_Timeout(const uint64_t timeout_msec) = 0;


    virtual int32_t Unlock() {return 0;}
    // virtual int32_t Unlock() = 0;


    virtual ~LockBase() {}
};


class decx::core::SpinLock : public decx::core::LockBase
{
private:
    std::atomic<uint8_t> _occupied;

public:
    SpinLock();


    virtual DecxAsync_WaitResult_e Lock_Timeout(const uint64_t timeout_msec);


    virtual DecxAsync_WaitResult_e Lock();


    virtual int32_t Unlock();
};


class decx::core::TimedMutexLock : public decx::core::LockBase
{
private:
    std::timed_mutex _mtx;

public:
    TimedMutexLock();


    virtual DecxAsync_WaitResult_e Lock();


    virtual DecxAsync_WaitResult_e Lock_Timeout(const uint64_t timeout_msec);


    virtual int32_t Unlock();
};


#endif