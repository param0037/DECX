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


#ifndef _VECTOR_DEFINES_H_
#define _VECTOR_DEFINES_H_

#include "include.h"
#include "decx_utils_macros.h"

// vectors for CPU codes
#ifndef _DECX_CUDA_PARTS_

//extern "C"
//{
typedef struct __align__(8) int2_t
{
    int x, y;
}int2;


typedef struct __align__(8) uint2_t
{
    uint x, y;
}uint2;


typedef struct __align__(16) uint3_t
{
    uint x, y, z;
}uint3;


typedef struct __align__(16) uint4_t
{
    uint x, y, z, w;
}uint4;



typedef struct __align__(8) float2_t
{
    float x, y;
}float2;


typedef struct __align__(16) float4_t
{
    float x, y, z, w;
}float4;


typedef struct __align__(16) double2_t
{
    double x, y;
}double2;


typedef struct __align__(16) int4_t
{
    int x, y, z, w;
}int4;


typedef struct __align__(16) int3_t
{
    int x, y, z;
}int3;


typedef struct __align__(4) uchar4_t
{
    uchar x, y, z, w;
}uchar4;


typedef struct uchar3_t
{
    uchar x, y, z;
}uchar3;


typedef struct __align__(16) ulong2_t
{
#ifdef Windows
    unsigned __int64 x, y;
#endif

#ifdef Linux
    unsigned long long x, y;
#endif
}ulong2;



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


static inline uint3 make_uint3(const uint x, const uint y, const uint z) {
    uint3 ans;
    ans.x = x;
    ans.y = y;
    ans.z = z;
    return ans;
}



static inline uint4 make_uint4(const uint x, const uint y, const uint z, const uint w)
{
    uint4 ans;
    ans.x = x;
    ans.y = y;
    ans.z = z;
    ans.w = w;
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


static inline double2 make_double2(const double x, const double y)
{
    double2 ans;
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
    ans.x = (uchar)x;
    ans.y = (uchar)y;
    ans.z = (uchar)z;
    ans.w = (uchar)w;
    return ans;
}

//}


#endif