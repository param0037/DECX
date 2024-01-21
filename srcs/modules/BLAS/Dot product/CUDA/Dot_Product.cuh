/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _DP1D_CUH_
#define _DP1D_CUH_


#include "../../../classes/Vector.h"
#include "../../../classes/Matrix.h"
#include "../../../classes/GPU_Vector.h"
#include "../../../../Async Engine/DecxStream/DecxStream.h"
#include "../../../classes/DecxNumber.h"
#include "../../basic_process/type_statistics/Matrix_reduce.h"


namespace de
{
    namespace cuda
    {
        _DECX_API_ de::DH Dot_product(de::Vector& A, de::Vector& B, de::DecxNumber& res, const uint32_t _fp16_accu);

        _DECX_API_ de::DH Dot_product(de::GPU_Vector& A, de::GPU_Vector& B, de::DecxNumber& res, const uint32_t _fp16_accu);

        _DECX_API_ de::DH Dot_product_Async(de::Vector& A, de::Vector& B, de::DecxNumber* res, const uint32_t _fp16_accu, de::DecxStream& S);
    }

    namespace cuda
    {
        _DECX_API_ de::DH Dot_product(de::Matrix& A, de::Matrix& B, de::Vector& dst, const de::REDUCE_METHOD _rd_method, const uint32_t _fp16_accu);
    }
}



#endif