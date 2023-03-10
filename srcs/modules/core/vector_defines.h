/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/

#ifndef _VECTOR_DEFINES_H_
#define _VECTOR_DEFINES_H_

#include "include.h"
#include "utils/decx_utils_macros.h"

// vectors for CPU codes
#if (!defined(_DECX_CUDA_CODES_) && !defined(_DECX_ALLOC_CODES_)) && !defined(_DECX_CLASSES_CODES_)


struct __align__(8) int2
{
    int x, y;
};


struct __align__(8) uint2
{
    uint x, y;
};


struct __align__(16) uint3
{
    uint x, y, z;
};


struct __align__(16) uint4
{
    uint x, y, z, w;
};



struct __align__(8) float2
{
    float x, y;
};


struct __align__(16) float4
{
    float x, y, z, w;
};


struct __align__(16) double2
{
    double x, y;
};


struct __align__(16) int4
{
    int x, y, z, w;
};


struct __align__(16) int3
{
    int x, y, z;
};


struct __align__(4) uchar4
{
    uchar x, y, z, w;
};


struct uchar3
{
    uchar x, y, z;
};


struct __align__(16) ulong2
{
#ifdef Windows
    unsigned __int64 x, y;
#endif

#ifdef Linux
    unsigned long long x, y;
#endif
};



static inline int2 make_int2(const int x, const int y) {
    int2 ans;
    ans.x = x;
    ans.y = y;
    return ans;
}



static inline uint2 make_uint2(const uint x, const uint y) {
    uint2 ans;
    ans.x = x;
    ans.y = y;
    return ans;
}



static inline ulong2 make_ulong2(const size_t x, const size_t y) {
    ulong2 ans;
    ans.x = x;
    ans.y = y;
    return ans;
}




static inline int4 make_int4(const int x, const int y, const int z, const int w) {
    int4 ans;
    ans.x = x;
    ans.y = y;
    ans.z = z;
    ans.w = w;
    return ans;
}


static inline float2 make_float2(const float x, const float y)
{
    float2 ans;
    ans.x = x;
    ans.y = y;
    return ans;
}


static inline int3 make_int3(const int x, const int y, const int z) {
    int3 ans;
    ans.x = x;
    ans.y = y;
    ans.z = z;
    return ans;
}

#endif


static inline uchar4 make_uchar4_from_fp32(const float x, const float y, const float z, const float w)
{
    uchar4 ans;
    ans.x = static_cast<uchar>(x);
    ans.y = static_cast<uchar>(y);
    ans.z = static_cast<uchar>(z);
    ans.w = static_cast<uchar>(w);
    return ans;
}




#endif