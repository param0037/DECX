/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/



#ifndef _CONFIG_H_
#define _CONFIG_H_

#include "../../core/basic.h"
#include "../allocators.h"

#ifdef _DECX_CUDA_CODES_
#include "../cudaStream_management/cudaStream_queue.h"
#endif


#ifdef _DECX_CUDA_CODES_
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

#ifdef _DECX_CPU_CODES_
namespace decx
{
    typedef struct cpuInfo
    {
        size_t cpu_concurrency;
        bool is_init;

        cpuInfo() {
            is_init = false;
        }
    };
}
#endif


namespace decx
{
#ifdef _DECX_CUDA_CODES_
#ifdef Windows
    extern decx::cudaProp cuP;
#endif
#ifdef Linux
    extern decx::cudaProp cuP;
#endif
#endif

#ifdef _DECX_CPU_CODES_
extern decx::cpuInfo cpI;
#endif
}


namespace de
{
#ifdef _DECX_CUDA_CODES_
    _DECX_API_ void InitCuda();
#endif

#ifdef _DECX_CPU_CODES_
    _DECX_API_ void InitCPUInfo();
#endif
}



#endif