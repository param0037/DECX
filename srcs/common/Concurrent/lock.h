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

#ifndef _DECX_LOCK_H_
#define _DECX_LOCK_H_


#include <basic.h>

#ifdef __cplusplus
extern "C" {
#endif


#define DecxLockImplSizeMax 64

typedef enum
{
    DecxWaitOpt_Hybrid  = 0,
    DecxWaitOpt_Spin    = 1,
    DecxWaitOpt_Relaxed = 2,
} DecxWaitOption_e;


typedef enum
{
    DecxWait_Success            = 0,        // Successfully waited
    DecxWait_Timeout            = 1,        // Wait but timeout occurs
    DecxWait_SpinOver           = 2,        // Spin count exceeds set maximal value
    DecxWait_RaceFail           = 3,        // 
    DecxWait_UnexpectedError    = -1,
} DecxWaitResult_e;


typedef struct
{
    DecxWaitOption_e    _option;            // Wait option
    uint32_t            _max_spin_cnt;      // Maximum spin count
    uint32_t            _spin_factor_exp;   // Spin count downscale factor (power of 2)
    uint64_t            _timeout_msec;      // Wait timeout (in millisecond)
} DecxWaitSettings_t;


typedef struct __align__(DecxLockImplSizeMax)
{
    uint8_t _impl[DecxLockImplSizeMax];
} DecxLock_t;

#define DECX_WAIT_FOREVER   0xFFFFFFFFFFFFFFFFU
#define DECX_WAIT_IMMIDIATE 0x0
#define DECX_WAIT_1MS       1
#define DECX_WAIT_10MS      10


_DECX_API_ int32_t DecxCore_LockCreate(DecxLock_t* p_lock);

_DECX_API_ int32_t DecxCore_LockDestroy(DecxLock_t* p_lock);

_DECX_API_ DecxWaitResult_e DecxCore_LockAcquire(DecxLock_t* p_lock, const DecxWaitSettings_t* p_settings);

_DECX_API_ int32_t DecxCore_LockRelease(DecxLock_t* p_lock);

#ifdef __cplusplus
}
#endif


#endif
