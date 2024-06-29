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


#include "../basic.h"
#include "decx_CPUID.h"


#ifdef _DECX_CORE_CUDA_
#include "../cudaStream_management/cudaStream_queue.h"
#include "../cudaStream_management/cudaEvent_queue.h"
#endif


#ifdef _DECX_CUDA_PARTS_
namespace decx
{
    typedef struct cudaProp
    {
        cudaDeviceProp prop;
        int CURRENT_DEVICE;
        bool is_init;

        cudaProp() { this->is_init = false; }
    };
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
    typedef struct cpuInfo
    {
        size_t cpu_concurrency;
        decx_CPUINFO _hardware_info;
        bool is_init;

        cpuInfo();
    };
}
#endif


namespace decx
{
#ifdef _DECX_CPU_PARTS_
    namespace cpu {
        _DECX_API_ bool _is_CPU_init();


        _DECX_API_ uint64_t _get_permitted_concurrency();


        _DECX_API_ uint64_t _get_hardware_concurrency();
    }
#endif
#ifdef _DECX_CUDA_PARTS_
    namespace cuda
    {
        _DECX_API_ bool _is_CUDA_init();


        _DECX_API_ cudaDeviceProp& _get_cuda_prop();
    }
#endif
}


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
        _DECX_API_ void DecxSetThreadingNum(const size_t _thread_num);
    }
#endif
}

#endif
