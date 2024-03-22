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
#include "../BLAS/Vectorial/vector4.h"

namespace de
{
    enum _DATA_TYPES_FLAGS_ 
    {
        _VOID_              = 0,
        _INT32_             = 1,
        _FP32_              = 2,
        _FP64_              = 3,
        _FP16_              = 4,
        _COMPLEX_F32_       = 5,
        _UINT8_             = 6,
        _UCHAR3_            = 7,
        _UCHAR4_            = 8,
        _VECTOR3F_          = 9,
        _VECTOR4F_          = 10,
        _UINT64_            = 11
    };


    enum _DATA_FORMATS_
    {
        _NA_                = 0,
        _COLOR_RGB_         = 1,
        _COLOR_BGR_         = 2,
        _COLOR_RGBA_        = 3,
        _COLOR_YUV_         = 4,
        _COLOR_HSV_         = 5,
        _CPLX_CARTESIAN_    = 6,
        _CPLX_POLAR_        = 7
    };
}

#define _SIZE_INT32_        sizeof(int)
#define _SIZE_FLOAT32_      sizeof(float)
#define _SIZE_UINT64_       sizeof(uint64_t)
#define _SIZE_FLOAT64_      sizeof(double)
#define _SIZE_FLOAT16_      sizeof(de::Half)
#define _SIZE_COMPLEX_F32_  sizeof(de::CPf)
#define _SIZE_UINT8_        sizeof(uchar)
#define _SIZE_UCHAR3_       sizeof(uchar4)
#define _SIZE_UCHAR4_       sizeof(uchar4)
#define _SIZE_VECTOR4_FP32_ sizeof(de::Vector4f)
#define _SIZE_VECTOR3_FP32_ sizeof(de::Vector3f)


// C defs
#if _C_EXPORT_ENABLED_
// Data Types
#define DECX_TYPE_VOID      0
#define DECX_TYPE_INT32     1
#define DECX_TYPE_FP32      2
#define DECX_TYPE_FP64      3
#define DECX_TYPE_FP16      4
#define DECX_TYPE_CPLXF32   5
#define DECX_TYPE_UINT8     6
#define DECX_TYPE_UC3       7
#define DECX_TYPE_UC4       8
#define DECX_TYPE_VEC3F     9
#define DECX_TYPE_VEC4F     10
#define DECX_TYPE_UINT64    11

// Data Formats
#define DECX_FORMAT_NA                  0
#define DECX_FORMAT_COLOR_RGB           1
#define DECX_FORMAT_COLOR_BGR           2
#define DECX_FORMAT_COLOR_RGBA          3
#define DECX_FORMAT_COLOR_YUV           4
#define DECX_FORMAT_COLOR_HSV           5
#define DECX_FORMAT_CPLX_CARTESIAN      6
#define DECX_FORMAT_CPLX_POLAR          7

// End of C defs
#endif

namespace decx
{
    namespace core {
        static uint8_t _size_mapping(const de::_DATA_TYPES_FLAGS_ type);
    }
}


static uint8_t decx::core::_size_mapping(const de::_DATA_TYPES_FLAGS_ type) {
    uint8_t __byte = 0;
    switch (type)
    {
    case de::_DATA_TYPES_FLAGS_::_VOID_:
        __byte = 0;                             break;
    case de::_DATA_TYPES_FLAGS_::_INT32_:
        __byte = _SIZE_INT32_;                  break;
    case de::_DATA_TYPES_FLAGS_::_FP32_:
        __byte = _SIZE_FLOAT32_;                break;
    case de::_DATA_TYPES_FLAGS_::_UINT64_:
        __byte = _SIZE_UINT64_;                 break;
    case de::_DATA_TYPES_FLAGS_::_FP64_:
        __byte = _SIZE_FLOAT64_;                break;
    case de::_DATA_TYPES_FLAGS_::_FP16_:
        __byte = _SIZE_FLOAT16_;                break;
    case de::_DATA_TYPES_FLAGS_::_COMPLEX_F32_:
        __byte = _SIZE_COMPLEX_F32_;            break;
    case de::_DATA_TYPES_FLAGS_::_UINT8_:
        __byte = _SIZE_UINT8_;                  break;
    case de::_DATA_TYPES_FLAGS_::_UCHAR3_:
        __byte = _SIZE_UCHAR3_;                 break;
    case de::_DATA_TYPES_FLAGS_::_UCHAR4_:
        __byte = _SIZE_UCHAR4_;                 break;
    case de::_DATA_TYPES_FLAGS_::_VECTOR3F_:
        __byte = _SIZE_VECTOR3_FP32_;           break;
    case de::_DATA_TYPES_FLAGS_::_VECTOR4F_:
        __byte = _SIZE_VECTOR4_FP32_;           break;
    default:
        break;
    }
    return __byte;
}


#endif