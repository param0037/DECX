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


#ifndef _DECX_UTILS_MACROS_H_
#define _DECX_UTILS_MACROS_H_


#include "compile_params.h"


#define GetLarger(__a, __b) ((__a) > (__b) ? (__a) : (__b))
#define GetSmaller(__a, __b) ((__a) < (__b) ? (__a) : (__b))
#ifdef _DECX_CUDA_PARTS_
#if __ABOVE_SM_53
#define GetLarger_fp16(__a, __b) (__hgt((__a), (__b)) ? (__a) : (__b))
#define GetSmaller_fp16(__a, __b) (__hle((__a), (__b)) ? (__a) : (__b))
#endif
#endif


#ifndef FORCEINLINE
#define FORCEINLINE inline
#endif


#define SWAP(A, B, tmp) {   \
    (tmp) = (A);            \
    (A) = (B);              \
    (B) = (tmp);            \
}


#ifndef max
#define max(a, b) ((a) > (b) ? (a) : (b))
#endif

#ifndef min
#define min(a, b) ((a) < (b) ? (a) : (b))
#endif


#ifdef _MSC_VER
#define _DECX_API_ __declspec(dllexport)
#endif

#ifdef Linux
#define _DECX_API_ __attribute__((visibility("default")))
#endif



// the hardware infomation
// the most blocks can execute concurrently in one SM
#define most_bl_per_sm 8



#ifndef __align__
#ifdef Windows
#define __align__(n) __declspec(align(n))
#endif

#ifdef Linux
#define __align__(n) __attribute__((aligned(n)))
#endif
#endif


typedef unsigned char uchar;
typedef unsigned int uint;
typedef unsigned short ushort;
typedef unsigned long long ull64;


#endif