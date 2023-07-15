/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _STORE_TYPES_H_
#define _STORE_TYPES_H_

namespace de {
    /*
    * @brief If 'Page_Locked; flag is used, system will call cudaHostAlloc(..., cudaHostAllocDefault)
    * So make sure the users won't use this flag to create any memory block if they just load DECX_cpu.dll(.so)
    * But if users are using DECX_CUDA.dll(.so), both flags work. But Page_Locked flag will accelerate DMA
    * process (transfering data between host and device)
    */
    enum DATA_STORE_TYPE
    {
        Page_Locked     = 0x00,         // call(ed) cudaHostAlloc(..., cudaHostAllocDefault)
        Page_Default    = 0x01,         // call(ed) decx::alloc::aligned_malloc_Hv() -> std::malloc()
        Device_Memory   = 0x02          // call(ed) decx::alloc::aligned_malloc_D() -> cudaMalloc()
    };
}

#endif