/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/


#ifndef _MATRIX_FILL_H_
#define _MATRIX_FILL_H_

#include "constant_fill_exec_fp32.h"
#include "constant_fill_exec_int32.h"
#include "../../../../classes/Matrix.h"

namespace de
{
    namespace cpu
    {
        _DECX_API_ de::DH Fill(de::Matrix<float>& src, const float _value);


        _DECX_API_ de::DH Fill(de::Matrix<int>& src, const int _value);


        _DECX_API_ de::DH Fill(de::Matrix<double>& src, const double _value);
    }
}


de::DH de::cpu::Fill(de::Matrix<float>& src, const float _value)
{
    de::DH handle;

    decx::_Matrix<float>* _src = dynamic_cast<decx::_Matrix<float>*>(&src);

    decx::err::Success(&handle);

    if (!decx::cpI.is_init) {
        decx::err::CPU_Not_init(&handle);
        return handle;
    }

    if (_src->Mat.ptr == NULL) {
        decx::err::ClassNotInit(&handle);
        return handle;
    }

    decx::fill_2D_fp32(_src->Mat.ptr, _value, _src->width, make_uint2(_src->pitch, _src->height));

    return handle;
}


de::DH de::cpu::Fill(de::Matrix<int>& src, const int _value)
{
    de::DH handle;

    decx::_Matrix<int>* _src = dynamic_cast<decx::_Matrix<int>*>(&src);

    decx::err::Success(&handle);

    if (!decx::cpI.is_init) {
        decx::err::CPU_Not_init(&handle);
        return handle;
    }

    if (_src->Mat.ptr == NULL) {
        decx::err::ClassNotInit(&handle);
        return handle;
    }

    decx::fill_2D_int32(_src->Mat.ptr, _value, _src->width, make_uint2(_src->pitch, _src->height));

    return handle;
}



#endif