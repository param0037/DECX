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

#ifndef _SPIN_LOCK_H_
#define _SPIN_LOCK_H_

#include <basic.h>
#define _LOCK_IMPL_SIZE_ 64

#ifdef __cplusplus
extern "C"
{
#endif
    enum class DecxLockType_e : uint8_t {
        LockType_Mutex = 0,
        LockType_Spin = 1,
    };

    enum class DecxAsync_WaitOption_e : uint8_t {
        Wait_Forever = 0,
        Wait_Timeout = 1,
    };


    enum class DecxAsync_WaitBehaviour_e : uint8_t {
        Wait_Lowpower = 0,
        Wait_Spin = 1,
    };

    enum class DecxAsync_WaitResult_e : uint8_t {
        Wait_Success = 0,
        Wait_Timeout = 1,
        Wait_UnknownErr = 255,
    };


    typedef struct DecxLock_t
    {
        uint8_t _impl[_LOCK_IMPL_SIZE_];
    };

    int32_t _DECX_API_ DecxLockCreate(DecxLock_t* p_lock, const DecxLockType_e lock_type);


    DecxAsync_WaitResult_e _DECX_API_ 
    DecxLockAcquire(DecxLock_t* p_lock, const DecxAsync_WaitOption_e option, const uint64_t timeout_msec);


    int32_t _DECX_API_ DecxLockRelease(DecxLock_t* p_lock);


    int32_t _DECX_API_ DecxLockDestroy(DecxLock_t* p_lock);

#ifdef __cplusplus
}
#endif


#endif
