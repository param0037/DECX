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

#ifndef _DECX_SEMAPHORE_IMPL_H_
#define _DECX_SEMAPHORE_IMPL_H_

#include <Concurrent/semaphore.h>
#include "../lock/decx_lock_impl.h"

namespace decx
{
namespace core
{
    class SemaphoreImpl_Binary;

    class SemaphoreImpl_Counting;
}
}


class _DECX_API_ decx::core::SemaphoreImpl_Binary
{
private:
    std::atomic<int32_t> _sem;
    std::atomic<uint32_t> _wait_num;

public:
    int32_t Reset()
    {
        this->_wait_num.store(0, std::memory_order_seq_cst);
        this->_sem.store(0, std::memory_order_seq_cst);
        return 0;
    }


    SemaphoreImpl_Binary() {
        this->Reset();
    }


    DecxWaitResult_e Wait(const DecxWaitSettings_t* p_settings);


    int32_t Post();


    ~SemaphoreImpl_Binary() {
        this->Reset();
    }
};


class _DECX_API_ decx::core::SemaphoreImpl_Counting
{
private:
    std::atomic<int32_t> _sem;
    std::atomic<uint32_t> _wait_num;

public:
    int32_t Reset()
    {
        this->_wait_num.store(0, std::memory_order_seq_cst);
        this->_sem.store(0, std::memory_order_seq_cst);
        return 0;
    }


    SemaphoreImpl_Counting() {
        this->Reset();
    }


    DecxWaitResult_e Wait(const DecxWaitSettings_t* p_settings, const int32_t count);


    int32_t Post(const int32_t notify_cnt);


    ~SemaphoreImpl_Counting() {
        this->Reset();
    }
};



#endif
