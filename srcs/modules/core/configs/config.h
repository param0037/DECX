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


#ifdef _DECX_CORE_CUDA_
#include "../cudaStream_management/cudaStream_queue.h"
#include "../cudaStream_management/cudaEvent_queue.h"
#endif



namespace decx
{
#if defined(_DECX_CORE_CUDA_) || defined(_DECX_CORE_CPU_)
    struct logging_config
    {
        bool _enable_log_print;
        bool _ignore_successful_print;
        bool _ignore_warnings;

        logging_config();
    };
#endif
}


#ifdef _DECX_CORE_CUDA_
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
        size_t _hardware_concurrency;
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
#ifdef _DECX_CUDA_CODES_
    extern decx::cudaProp cuP;
#endif

#if defined(_DECX_CORE_CPU_)
    extern decx::cpuInfo cpI;
    extern decx::logging_config LogConf;
#endif
}


namespace de
{
#ifdef _DECX_CUDA_CODES_
    _DECX_API_ void InitCuda();
#endif

#ifdef _DECX_CPU_CODES_
    _DECX_API_ void InitCPUInfo();

    namespace cpu {
        _DECX_API_ de::DH DecxSetThreadingNum(const size_t _thread_num);
    }
#endif

    // Realized by DECX_allocations
    _DECX_API_ void DecxEnableLogPrint();


    // Realized by DECX_allocations
    _DECX_API_ void DecxDisableLogPrint();


    _DECX_API_ void DecxEnableWarningPrint();


    _DECX_API_ void DecxDisableWarningPrint();


    _DECX_API_ void DecxEnableSuccessfulPrint();


    _DECX_API_ void DecxDisableSuccessfulPrint();
}



#endif