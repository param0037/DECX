/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/

#include "module_fp32.h"
#include "../../handles/decx_handles.h"


_DECX_API_ de::DH 
de::signal::CPU::Module(de::Matrix& src, de::Matrix& dst)
{
    de::DH handle;

    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);
    decx::_Matrix* _dst = dynamic_cast<decx::_Matrix*>(&dst);

    if (_src->type != decx::_DATA_TYPES_FLAGS_::_COMPLEX_F32_ ||
        _dst->type != decx::_DATA_TYPES_FLAGS_::_FP32_) {
        decx::err::TypeError_NotMatch(&handle);
        return handle;
    }

    decx::signal::_module_fp32_caller((de::CPf*)_src->Mat.ptr, (float*)_dst->Mat.ptr, _src->_element_num);

    decx::err::Success(&handle);
    return handle;
}