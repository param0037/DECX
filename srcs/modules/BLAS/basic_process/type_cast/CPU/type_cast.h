/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _MATRIX_TYPE_CAST_H_
#define _MATRIX_TYPE_CAST_H_


#include "../../../../classes/Matrix.h"
#include "../../../../classes/MatrixArray.h"
#include "../../../../classes/Vector.h"
#include "../../../../classes/Tensor.h"
#include "_mm256_fp32_fp64.h"
#include "_mm256_fp32_int32.h"
#include "_mm256_uint8_int32.h"
#include "_mm256_fp32_uint8.h"


namespace decx
{
    namespace type_cast
    {
        namespace cpu {
            template <bool _print>
            _DECX_API_ void _type_cast1D_organiser(void* src, void* dst, const size_t proc_len, const int cvt_method, de::DH* handle);


            template <bool _print>
            _DECX_API_ void _type_cast2D_organiser(void* src, void* dst, const ulong2 proc_dims, const uint32_t Wsrc,
                const uint32_t Wdst, const int cvt_method, de::DH* handle);
        }
    }
}



namespace de {
    namespace cpu 
    {
        _DECX_API_ de::DH TypeCast(de::Vector& src, de::Vector& dst, const int cvt_method);


        _DECX_API_ de::DH TypeCast(de::Matrix& src, de::Matrix& dst, const int cvt_method);


        _DECX_API_ de::DH TypeCast(de::MatrixArray& src, de::MatrixArray& dst, const int cvt_method);


        _DECX_API_ de::DH TypeCast(de::Tensor& src, de::Tensor& dst, const int cvt_method);
    }
}


#endif