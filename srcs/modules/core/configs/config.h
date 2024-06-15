/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
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
