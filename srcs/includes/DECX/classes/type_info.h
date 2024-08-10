/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/


#ifndef _TYPE_INFO_H_
#define _TYPE_INFO_H_

#include"class_utils.h"

#ifdef __cplusplus
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
        _NA_            = 0,
        _RGB_           = 1,
        _YUV444_        = 2,
        _HSV_           = 3,
        _CPLX_CARTESIAN_ = 4,
        _CPLX_POLAR_    = 5
    };
}
#endif

#ifdef _C_CONTEXT_
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

#endif


#endif
