/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/


#ifndef _MATRIXARRAY_FILL_H_
#define _MATRIXARRAY_FILL_H_

#include "constant_fill_exec_fp32.h"
#include "constant_fill_exec_int32.h"
#include "../../../../classes/MatrixArray.h"


namespace de
{
    namespace cpu
    {
        _DECX_API_ de::DH Fill(de::MatrixArray<float>& src, const float _value);
    }
}


de::DH de::cpu::Fill(de::MatrixArray<float>& src, const float _value)
{
    de::DH handle;

    return handle;
}



#endif