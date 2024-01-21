/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _CUDA_GEMM_H_
#define _CUDA_GEMM_H_


#include "../../../../classes/Matrix.h"
#include "../../../../classes/Vector.h"
#include "../../../../classes/GPU_Matrix.h"
#include "../../../../classes/GPU_Vector.h"


namespace de
{
    namespace cuda
    {
        _DECX_API_ de::DH GEMM(de::Matrix& A, de::Matrix& B, de::Matrix& dst);


        _DECX_API_ de::DH GEMM(de::Matrix& A, de::Matrix& B, de::Matrix& C, de::Matrix& dst);


        _DECX_API_ de::DH GEMM(de::Vector& A, de::Matrix& B, de::Vector& dst, const uint32_t _fp16_accu);
        _DECX_API_ de::DH GEMM(de::Matrix& A, de::Vector& B, de::Vector& dst, const uint32_t _fp16_accu);


        // --------------------------------------------- on GPU -----------------------------------------------------------

        _DECX_API_ de::DH GEMM(de::GPU_Matrix& A, de::GPU_Matrix& B, de::GPU_Matrix& dst, const int flag);


        _DECX_API_ de::DH GEMM(de::GPU_Matrix& A, de::GPU_Matrix& B, de::GPU_Matrix& C, de::GPU_Matrix& dst, const int flag);
    }
}



namespace decx {
    namespace cuda {
        _DECX_API_ void GEMM_AB_Raw_API(decx::_Matrix* A, decx::_Matrix* B, decx::_Matrix* dst, de::DH* handle);


        _DECX_API_ void GEMM_ABC_Raw_API(decx::_Matrix* A, decx::_Matrix* B, decx::_Matrix* C, decx::_Matrix* dst, de::DH* handle);


        _DECX_API_ void dev_GEMM_AB_Raw_API(decx::_GPU_Matrix* A, decx::_GPU_Matrix* B, decx::_GPU_Matrix* dst, const int flag, de::DH* handle);


        _DECX_API_ void dev_GEMM_ABC_Raw_API(decx::_GPU_Matrix* A, decx::_GPU_Matrix* B, decx::_GPU_Matrix* C, decx::_GPU_Matrix* dst, const int flag, de::DH* handle);
    }
}


#endif