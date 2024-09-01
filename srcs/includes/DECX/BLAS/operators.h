/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _OPERATORS_H_
#define _OPERATORS_H_

#include "../classes/Matrix.h"
#include "../classes/Vector.h"
#include "../classes/Tensor.h"
#include "../classes/GPU_Matrix.h"
#include "../classes/GPU_Vector.h"
#include "../classes/GPU_Tensor.h"
#include "../classes/class_utils.h"
#include "../classes/DecxNumber.h"


namespace de
{
    enum DecxArithmetic{
        ADD = 0x00,
        MUL = 0x01,
        MIN = 0x02,
        MAX = 0x03,
        SIN = 0x04,
        COS = 0x05,

        SUB = 0x06,
        DIV = 0x07,
        
        OP_INV = 0x40
    };
}


namespace de
{
namespace blas{
namespace cpu{
    _DECX_API_ void Arithmetic(de::InputVector A, de::InputVector B, de::OutputVector dst, const int32_t arith_flag);
    _DECX_API_ void Arithmetic(de::InputVector src, de::InputNumber constant, de::OutputVector dst, const int32_t arith_flag);
    _DECX_API_ void Arithmetic(de::InputVector src, de::OutputVector dst, const int32_t arith_flag);
}

namespace cpu{
    _DECX_API_ void Arithmetic(de::InputMatrix A, de::InputMatrix B, de::OutputMatrix dst, const int32_t arith_flag);
    _DECX_API_ void Arithmetic(de::InputMatrix src, de::InputNumber constant, de::OutputMatrix dst, const int32_t arith_flag);
    _DECX_API_ void Arithmetic(de::InputMatrix src, de::OutputMatrix dst, const int32_t arith_flag);
}
}

}
#endif