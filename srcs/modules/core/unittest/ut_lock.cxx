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

#include <Concurrent/lock.h>
#include "ut_decx_core_cpu.h"

DecxLock_t g_lock;

char g_container[128];

static void print_test(const uint32_t id)
{
    for (int i = 0; i < 1000; ++i) {
        DecxLockAcquire(&g_lock, DecxAsync_WaitOption_e::Wait_Forever, 0);
        snprintf(g_container, 128, "Hello from id=%d\n", id);
        std::this_thread::sleep_for(std::chrono::microseconds(100));
        snprintf(g_container, 128, "Hello from id=%d After sleep(10)\n", id);
        DecxLockRelease(&g_lock);
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
}

int32_t _DECX_API_ DecxUT_LockTest()
{
    DecxLockCreate(&g_lock, DecxLockType_e::LockType_Spin);
    // DecxLockCreate(&g_lock, DecxLockType_e::LockType_Mutex);

    auto s = std::chrono::steady_clock::now();
    std::thread thr0(print_test, 0);
    std::thread thr1(print_test, 1);
    thr0.join();
    thr1.join();
    auto e = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(e - s);
    printf("time spent : %llu us\n", duration.count());
    return 0;
}