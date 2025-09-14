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

#ifndef _DECX_LOCK_IMPL_H_
#define _DECX_LOCK_IMPL_H_

#include "Concurrent/lock.h"


namespace decx
{
namespace core
{
    class LockImpl;


    enum class LockStatus_e : int32_t
    {
        LockStatus_Released     = 0,
        LockStatus_Acquired     = 1,
    };


    enum class DecxWaitTolerance_e : uint8_t
    {
        WaitTol_GreaterEqual    = 0,
        WaitTol_LessEqual       = 1,
        WaitTol_Equal           = 2,
    };


    template <DecxWaitTolerance_e Tol> DecxWaitResult_e WaitSpin(std::atomic<int32_t>& predicate, const int32_t desired, 
        const uint64_t timeout_msec, const uint32_t max_spin_num, const uint32_t spin_factor_exp);
    

    template <DecxWaitTolerance_e Tol> DecxWaitResult_e WaitRelaxed(std::atomic<int32_t>& predicate, const int32_t desired, 
        const uint64_t timeout_msec);


    template <DecxWaitTolerance_e Tol> DecxWaitResult_e WaitHybrid(std::atomic<int32_t>& predicate, const int32_t desired, 
        const uint64_t timeout_msec, const uint32_t max_spin_num, const uint32_t spin_factor_exp);
}
}

class _DECX_API_ decx::core::LockImpl
{
private:
    std::atomic<int32_t> _flag;
    uint8_t _init;

public:
    LockImpl();

    
    DecxWaitResult_e Acquire(const DecxWaitSettings_t* p_settings);


    int32_t Release();


    ~LockImpl();
};

#endif
