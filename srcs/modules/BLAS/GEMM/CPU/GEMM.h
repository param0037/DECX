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


namespace de
{
    namespace cpu {
        _DECX_API_ void GEMM(de::Matrix& A, de::Matrix& B, de::Matrix& dst);


        _DECX_API_ void GEMM(de::Matrix& A, de::Matrix& B, de::Matrix& C, de::Matrix& dst);
    }
}



#endif
