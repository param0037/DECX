/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/


#ifndef _VECTOR_FILL_H_
#define _VECTOR_FILL_H_

#include "constant_fill_exec_fp32.h"
#include "constant_fill_exec_int32.h"
#include "../../../../classes/Vector.h"

namespace de
{
    namespace cpu
    {
        _DECX_API_ de::DH Fill(de::Vector<float>& src, const float _value);


        _DECX_API_ de::DH Fill(de::Vector<int>& src, const int _value);
    }
}


de::DH de::cpu::Fill(de::Vector<float>& src, const float _value)
{
    de::DH handle;

    decx::_Vector<float>* _src = dynamic_cast<decx::_Vector<float>*>(&src);

    decx::err::Success(&handle);

    if (!decx::cpI.is_init) {
        decx::err::CPU_Not_init(&handle);
        return handle;
    }

    if (_src->Vec.ptr == NULL) {
        decx::err::ClassNotInit(&handle);
        return handle;
    }

    decx::fill_1D_fp32(_src->Vec.ptr, _value, _src->length);

    return handle;
}



de::DH de::cpu::Fill(de::Vector<int>& src, const int _value)
{
    de::DH handle;

    decx::_Vector<int>* _src = dynamic_cast<decx::_Vector<int>*>(&src);

    decx::err::Success(&handle);

    if (!decx::cpI.is_init) {
        decx::err::CPU_Not_init(&handle);
        return handle;
    }

    if (_src->Vec.ptr == NULL) {
        decx::err::ClassNotInit(&handle);
        return handle;
    }

    decx::fill_1D_int32(_src->Vec.ptr, _value, _src->length);

    return handle;
}



#endif