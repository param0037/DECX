/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _TYPE_INFO_H_
#define _TYPE_INFO_H_

#include"classes_util.h"
#include "../Vectorial/vector4.h"

namespace decx
{
    enum _DATA_TYPES_FLAGS_ 
    {
        _VOID_          = 0,
        _INT32_         = 1,
        _FP32_          = 2,
        _FP64_          = 3,
        _FP16_          = 4,
        _COMPLEX_F32_   = 5,
        _UINT8_         = 6,
        _UCHAR3_        = 7,
        _UCHAR4_        = 8,
        _VECTOR3F_      = 9,
        _VECTOR4F_      = 10,
        _UINT64_        = 11
    };
}

#define _SIZE_INT32_ sizeof(int)
#define _SIZE_FLOAT32_ sizeof(float)
#define _SIZE_UINT64_ sizeof(uint64_t)
#define _SIZE_FLOAT64_ sizeof(double)
#define _SIZE_FLOAT16_ sizeof(de::Half)
#define _SIZE_COMPLEX_F32_ sizeof(de::CPf)
#define _SIZE_UINT8_ sizeof(uchar)
#define _SIZE_UCHAR3_ sizeof(uchar4)
#define _SIZE_UCHAR4_ sizeof(uchar4)
#define _SIZE_VECTOR4_FP32_ sizeof(de::Vector4f)
#define _SIZE_VECTOR3_FP32_ sizeof(de::Vector3f)


namespace decx
{
    namespace core {
        static int _size_mapping(const int type);
    }
}


static int decx::core::_size_mapping(const int type) {
    int __byte = 0;
    switch (type)
    {
    case decx::_DATA_TYPES_FLAGS_::_VOID_:
        __byte = 0;                             break;
    case decx::_DATA_TYPES_FLAGS_::_INT32_:
        __byte = _SIZE_INT32_;                  break;
    case decx::_DATA_TYPES_FLAGS_::_FP32_:
        __byte = _SIZE_FLOAT32_;                break;
    case decx::_DATA_TYPES_FLAGS_::_UINT64_:
        __byte = _SIZE_UINT64_;                 break;
    case decx::_DATA_TYPES_FLAGS_::_FP64_:
        __byte = _SIZE_FLOAT64_;                break;
    case decx::_DATA_TYPES_FLAGS_::_FP16_:
        __byte = _SIZE_FLOAT16_;                break;
    case decx::_DATA_TYPES_FLAGS_::_COMPLEX_F32_:
        __byte = _SIZE_COMPLEX_F32_;            break;
    case decx::_DATA_TYPES_FLAGS_::_UINT8_:
        __byte = _SIZE_UINT8_;                  break;
    case decx::_DATA_TYPES_FLAGS_::_UCHAR3_:
        __byte = _SIZE_UCHAR3_;                 break;
    case decx::_DATA_TYPES_FLAGS_::_UCHAR4_:
        __byte = _SIZE_UCHAR4_;                 break;
    case decx::_DATA_TYPES_FLAGS_::_VECTOR3F_:
        __byte = _SIZE_VECTOR3_FP32_;           break;
    case decx::_DATA_TYPES_FLAGS_::_VECTOR4F_:
        __byte = _SIZE_VECTOR4_FP32_;           break;
    default:
        break;
    }
    return __byte;
}


#endif