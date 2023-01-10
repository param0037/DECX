/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/

#ifndef _DECX_UTILS_MACROS_H_
#define _DECX_UTILS_MACROS_H_


#define GetLarger(__a, __b) ((__a) > (__b) ? (__a) : (__b))
#define GetSmaller(__a, __b) ((__a) < (__b) ? (__a) : (__b))
#ifdef _DECX_CUDA_CODES_
#if __ABOVE_SM_53
#define GetLarger_fp16(__a, __b) (__hgt((__a), (__b)) ? (__a) : (__b))
#define GetSmaller_fp16(__a, __b) (__hle((__a), (__b)) ? (__a) : (__b))
#endif
#endif


#ifndef FORCEINLINE
#define FORCEINLINE inline
#endif


#define SWAP(A, B, tmp) {    \
(tmp) = (A);    \
(A) = (B);    \
(B) = (tmp);    \
}




#define GetValue(src, _row, _col, cols) (src)[(_row) * (cols) + (_col)]

#ifndef GNU_CPUcodes
#define _MAX_TPB_(_prop, _device) cudaGetDeviceProperties(&_prop, _device);
#endif

//#define __SPACE__ sizeof(T)




#ifdef Windows
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


#endif