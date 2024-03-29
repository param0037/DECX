/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _CPU_GEMM_H_
#define _CPU_GEMM_H_


#include "../../../classes/Matrix.h"
#include "../../../../Async Engine/DecxStream/DecxStream.h"



namespace de
{
    namespace cpu {
        _DECX_API_ de::DH GEMM(de::Matrix& A, de::Matrix& B, de::Matrix& dst);


        _DECX_API_ de::DH GEMM(de::Matrix& A, de::Matrix& B, de::Matrix& C, de::Matrix& dst);


        _DECX_API_ void GEMM_Async(de::Matrix& A, de::Matrix& B, de::Matrix& dst, de::DecxStream& S);


        _DECX_API_ void GEMM_Async(de::Matrix& A, de::Matrix& B, de::Matrix& C, de::Matrix& dst, de::DecxStream& S);
    }
}


namespace decx
{
    namespace cpu {
        _DECX_API_ void GEMM_AB_Raw_API(decx::_Matrix* A, decx::_Matrix* B, decx::_Matrix* dst, de::DH* handle);


        _DECX_API_ void GEMM_ABC_Raw_API(decx::_Matrix* A, decx::_Matrix* B, decx::_Matrix* C, decx::_Matrix* dst, de::DH* handle);
    }
}



#endif