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


#include "config.h"
#include <Handle/decx_handle.h>


_DECX_API_ void de::InitCPUInfo()
{
    decx::cpI.is_init = true;
    _decx_get_CPU_info(&decx::cpI._hardware_info);
    
    decx::cpI.cpu_concurrency = decx::cpI._hardware_info._hardware_concurrency;
}


_DECX_API_ void de::cpu::DecxSetThreadingNum(const uint64_t _thread_num)
{
    decx::cpI.cpu_concurrency = decx::utils::clamp_min<uint64_t>(_thread_num, 1);
    DecxResetLastHandle();
    if (_thread_num > decx::cpI._hardware_info._hardware_concurrency) {
        DecxAssignLastHandle(DecxErrorTypes_e::DECX_WARN_CPU_HYPER_THREADING, CPU_HYPER_THREADING);
    }
    
    return;
}


_DECX_API_ uint8_t DecxGetIsCPUInit()
{
    return decx::cpI.is_init;
}


_DECX_API_ uint64_t DecxGetPermitConcurrency()
{
    return decx::cpI.cpu_concurrency;
}

#if defined(__x86_64__) || defined(__i386__)
_DECX_API_ uint64_t DecxGetL1DataCacheSize_PerCore()
{
    return _decx_get_L1_cache_size_per_phy_core(_decx_NOT_AMD_CPU(&decx::cpI._hardware_info));
}


_DECX_API_ uint64_t DecxGetL2CacheSize_PerCore()
{
    return _decx_get_L2_cache_size_per_phy_core(_decx_NOT_AMD_CPU(&decx::cpI._hardware_info));
}


_DECX_API_ uint64_t DecxGetL3CacheSize()
{
    return _decxDecxGetL3CacheSize(_decx_NOT_AMD_CPU(&decx::cpI._hardware_info));
}
#endif


_DECX_API_ uint64_t DecxGetHWConcurrency()
{
    return decx::cpI._hardware_info._hardware_concurrency;
}