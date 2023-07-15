/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _MATRIX_RECT_H_
#define _MATRIX_RECT_H_

#include "../../../classes/Matrix.h"
#include "rect_copy2D_exec.h"


namespace de
{
    namespace cpu
    {
        template <typename T>
        _DECX_API_ de::DH Rect(de::Matrix<T>& src, de::Matrix<T>& dst, const de::Point2D start, const de::Point2D end);
    }
}

template <typename T>
de::DH de::cpu::Rect(de::Matrix<T>& src, de::Matrix<T>& dst, const de::Point2D start, const de::Point2D end)
{
    de::DH handle;

    if (!decx::cpI.is_init) {
        decx::err::CPU_Not_init(&handle);
        Print_Error_Message(4, CPU_NOT_INIT);
        return handle;
    }

    decx::_Matrix<T>* _src = dynamic_cast<decx::_Matrix<T>*>(&src);
    decx::_Matrix<T>* _dst = dynamic_cast<decx::_Matrix<T>*>(&dst);

    // ~.x -> width; ~.y -> height
    uint2 cpy_dim = make_uint2(decx::utils::clamp_max<uint>(end.y, _src->width) - start.y,
                               decx::utils::clamp_max<uint>(end.x, _src->height) - start.x);
    
    decx::_cpy2D_anybit_caller<T>(DECX_PTR_SHF_XY_SAME_TYPE<T>(_src->Mat.ptr, start.x, start.y, _src->pitch), 
        _dst->Mat.ptr, cpy_dim);

    decx::err::Success(&handle);
    return handle;
}


#endif