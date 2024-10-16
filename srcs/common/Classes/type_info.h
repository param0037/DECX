/**
*   ----------------------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ----------------------------------------------------------------------------------
* 
* This is a part of the open source project named "DECX", a high-performance scientific
* computational library. This project follows the MIT License. For more information 
* please visit https://github.com/param0037/DECX.
* 
* Copyright (c) 2021 Wayne Anderson
* 
* Permission is hereby granted, free of charge, to any person obtaining a copy of this 
* software and associated documentation files (the "Software"), to deal in the Software 
* without restriction, including without limitation the rights to use, copy, modify, 
* merge, publish, distribute, sublicense, and/or sell copies of the Software, and to 
* permit persons to whom the Software is furnished to do so, subject to the following 
* conditions:
* 
* The above copyright notice and this permission notice shall be included in all copies 
* or substantial portions of the Software.
* 
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
* INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR 
* PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE 
* FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR 
* OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
* DEALINGS IN THE SOFTWARE.
*/

#ifndef _TYPE_INFO_H_
#define _TYPE_INFO_H_

#include "classes_util.h"
#include "../../modules/BLAS/Vectorial/vector4.h"

namespace de
{
    enum _DATA_TYPES_FLAGS_ 
    {
        _VOID_              = 0,

        // FP32 and CPLXF : 0bxx01
        _FP32_              = 0b0001,
        _COMPLEX_F32_       = 0b0101,

        _FP16_              = 0b0011,

        // FP64 and CPLXD : 0bxx10
        _FP64_              = 0b0010,
        _COMPLEX_F64_       = 0b0110,

        // Pixel related : 0bxx00
        _UINT8_             = 0b1000,
        _UCHAR4_            = 0b1100,

        _INT32_             = 0b0100,
        _UINT64_            = 0b0111,

        _VECTOR3F_          = 0b1110,
        _VECTOR4F_          = 0b1111
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
        _CPLX_POLAR_        = 7,
        _VECTOR4_           = 8,
        _QUATERNION_        = 9,
        _VECTOR2_           = 10
    };
}

#define _SIZE_INT32_        sizeof(int)
#define _SIZE_FLOAT32_      sizeof(float)
#define _SIZE_UINT64_       sizeof(uint64_t)
#define _SIZE_FLOAT64_      sizeof(double)
#define _SIZE_FLOAT16_      sizeof(de::Half)
#define _SIZE_COMPLEX_F32_  sizeof(de::CPf)
#define _SIZE_UINT8_        sizeof(uchar)
//#define _SIZE_UCHAR3_       sizeof(uchar4)
#define _SIZE_UCHAR4_       sizeof(uchar4)
#define _SIZE_VECTOR4_FP32_ sizeof(de::Vector4f)
#define _SIZE_VECTOR3_FP32_ sizeof(de::Vector3f)


// C defs
#if _C_EXPORT_ENABLED_
// Data Types
#define DECX_TYPE_VOID      0
#define DECX_TYPE_INT32     0b0100
#define DECX_TYPE_FP32      0b0001
#define DECX_TYPE_FP64      0b0010
#define DECX_TYPE_FP16      0b0011
#define DECX_TYPE_CPLXF32   0b0101
#define DECX_TYPE_UINT8     0b1000
//#define DECX_TYPE_UC3       7
#define DECX_TYPE_UC4       0b1100
#define DECX_TYPE_VEC3F     0b1110
#define DECX_TYPE_VEC4F     0b1111
#define DECX_TYPE_UINT64    0b0111

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
    case de::_DATA_TYPES_FLAGS_::_COMPLEX_F64_:
        __byte = _SIZE_VECTOR4_FP32_;           break;
    case de::_DATA_TYPES_FLAGS_::_UINT8_:
        __byte = _SIZE_UINT8_;                  break;
    /*case de::_DATA_TYPES_FLAGS_::_UCHAR3_:
        __byte = _SIZE_UCHAR3_;                 break;*/
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