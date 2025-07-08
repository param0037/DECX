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

#ifndef _CONFIG_H_
#define _CONFIG_H_


#include "../../../common/basic.h"
#include "decx_CPUID.h"


#ifdef _DECX_CORE_CUDA_
#include "../cudaStream_management/cudaStream_queue.h"
#include "../cudaStream_management/cudaEvent_queue.h"
#endif


#ifdef _DECX_CUDA_PARTS_
namespace decx
{
    typedef struct cudaProp_t
    {
        cudaDeviceProp prop;
        int CURRENT_DEVICE;
        bool is_init;

        cudaProp_t() { this->is_init = false; }
    }cudaProp;
}


namespace de
{
    namespace cuda {
        _DECX_API_ void DECX_CUDA_exit();
    }
}
#endif

#ifdef _DECX_CORE_CPU_
namespace decx
{
    // Realized by DECX_allocations
    typedef struct cpuInfo_t
    {
        uint64_t cpu_concurrency;
        decx_CPUINFO _hardware_info;
        bool is_init;

        cpuInfo_t(){
            this->is_init = false;
            this->cpu_concurrency = 0;
        }
    }cpuInfo;
}
#endif


#ifdef __cplusplus
extern "C" {
#endif
#ifdef _DECX_CPU_PARTS_
    _DECX_API_ uint8_t DecxGetIsCPUInit();


    _DECX_API_ uint64_t DecxGetPermitConcurrency();


    _DECX_API_ uint64_t DecxGetL1DataCacheSize_PerCore();


    _DECX_API_ uint64_t DecxGetL2CacheSize_PerCore();


    _DECX_API_ uint64_t DecxGetL3CacheSize();


    _DECX_API_ uint64_t DecxGetHWConcurrency();
    
#endif
#ifdef _DECX_CUDA_PARTS_
    _DECX_API_ uint8_t DecxGetIsCUDAInit();


    _DECX_API_ cudaDeviceProp& DecxGetCUDAProp();
#endif
#ifdef __cplusplus
}
#endif


namespace decx
{
#ifdef _DECX_CUDA_PARTS_
    extern decx::cudaProp cuP;
#endif

#if defined(_DECX_CORE_CPU_)
    extern decx::cpuInfo cpI;
#endif
}


namespace de
{
#ifdef _DECX_CUDA_PARTS_
    _DECX_API_ void InitCuda();
#endif

#ifdef _DECX_CPU_PARTS_
    _DECX_API_ void InitCPUInfo();

    namespace cpu {
        _DECX_API_ void DecxSetThreadingNum(const uint64_t _thread_num);
    }
#endif
}

#endif
