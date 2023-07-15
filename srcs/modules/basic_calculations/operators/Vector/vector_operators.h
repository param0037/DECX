/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _VECTOR_OPERATORS_H_
#define _VECTOR_OPERATORS_H_


#include "../../../classes/Vector.h"
#include "../../../classes/GPU_Vector.h"
#include "../../../core/basic.h"
#ifdef _DECX_ASYNC_CODES_
#include "../../../../Async Engine/DecxStream/DecxStream.h"
#endif


namespace de
{
    namespace cuda
    {
        _DECX_API_  de::DH Add(de::GPU_Vector& A, de::GPU_Vector& B, de::GPU_Vector& dst);



        _DECX_API_  de::DH Add(de::GPU_Vector& src, void* __x, de::GPU_Vector& dst);
    }
}


namespace decx
{
    namespace cuda {
        _DECX_API_ void cuda_Add_Raw_API(decx::_GPU_Vector* A, decx::_GPU_Vector* B, decx::_GPU_Vector* C, de::DH* handle);


        _DECX_API_ void cuda_AddC_Raw_API(decx::_GPU_Vector* src, void* __x, decx::_GPU_Vector* dst, de::DH* handle);
    }
}



#endif