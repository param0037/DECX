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

#ifndef _DECX_SEMAPHORE_H_
#define _DECX_SEMAPHORE_H_

#include <basic.h>
#include "lock.h"


#ifdef __cplusplus
extern "C" {
#endif

#define DecxSemImplSizeMax 64


typedef __align__(DecxSemImplSizeMax) struct 
{
    uint8_t _impl[DecxSemImplSizeMax];
} DecxBinarySemaphore_t;


typedef __align__(DecxSemImplSizeMax) struct 
{
    uint8_t _impl[DecxSemImplSizeMax];
} DecxCountingSemaphore_t;


/**
 * @brief Create semaphore according to the given type
 * @param p_sem Pointer to the semaphore instance to be allocated on
 * @return 0 for OK; otherwise NG
 */
_DECX_API_ int32_t DecxCore_BinarySemaphoreCreate(DecxBinarySemaphore_t* p_sem);
_DECX_API_ int32_t DecxCore_CountingSemaphoreCreate(DecxCountingSemaphore_t* p_sem);

/**
 * @brief Destroy a semaphore
 * @param p_sem Pointer to the semaphore instance to be deleted
 * @return 0 for OK; otherwise NG
 */
_DECX_API_ int32_t DecxCore_BinarySemaphoreDestroy(DecxBinarySemaphore_t* p_sem);
_DECX_API_ int32_t DecxCore_CountingSemaphoreDestroy(DecxCountingSemaphore_t* p_sem);

/**
 * @brief Wait for a signal on the semaphore. If it's binary semaphore, wait for any post operation related; if it's 
 *        counting semaphore, wait for the post operation that notifies at the same count
 * @param p_sem Pointer to the semaphore
 * @param p_settings Pointer to the waiting settings
 * @param (desired_cnt) Which count specifically is waiting for. If it's binary semaphore, this value will not be used
 * @return See DecxWaitResult_e enumerator for more information
 */
_DECX_API_ DecxWaitResult_e 
DecxCore_BinarySemaphoreWait(DecxBinarySemaphore_t* p_sem, const DecxWaitSettings_t* p_settings);
_DECX_API_ DecxWaitResult_e 
DecxCore_CountingSemaphoreWait(DecxCountingSemaphore_t* p_sem, const DecxWaitSettings_t* p_settings, int32_t desired_cnt);

/**
 * @brief Post on a semaphore
 * @param p_sem Pointer to the semaphore
 * @param notify_cnt At which count the semaphore should be notified. If it's counting semaphore, this value should be the same
 *                   as "desired_cnt" where the thread calling DecxCore_SemaphoreWait(); If it's counting semaphore, this value
 *                   will not be used
 * @param 0 for OK; otherwise NG
 */
_DECX_API_ int32_t DecxCore_BinarySemaphorePost(DecxBinarySemaphore_t* p_sem);
_DECX_API_ int32_t DecxCore_CountingSemaphorePost(DecxCountingSemaphore_t* p_sem, int32_t notify_cnt);


_DECX_API_ int32_t DecxCore_BinarySemaphoreReset(DecxBinarySemaphore_t* p_sem);
_DECX_API_ int32_t DecxCore_CountingSemaphoreReset(DecxCountingSemaphore_t* p_sem);

#ifdef __cplusplus
}
#endif

#endif