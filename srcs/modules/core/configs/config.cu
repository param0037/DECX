/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#include "config.h"


_DECX_API_ void de::InitCuda()
{
    checkCudaErrors(cudaGetDevice(&decx::cuP.CURRENT_DEVICE));
    checkCudaErrors(cudaGetDeviceProperties(&decx::cuP.prop, decx::cuP.CURRENT_DEVICE));
    decx::cuP.is_init = true;
}



_DECX_API_ void de::cuda::DECX_CUDA_exit()
{
    //decx::alloc::release_all_tmp();

    decx::CStream.release();
}


_DECX_API_ bool decx::cuda::_is_CUDA_init()
{
    return decx::cuP.is_init;
}


_DECX_API_ cudaDeviceProp& decx::cuda::_get_cuda_prop()
{
    return decx::cuP.prop;
}