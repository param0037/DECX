/**
*    ---------------------------------------------------------------------
*    Author : Wayne
*   Date   : 2021.04.16
*    ---------------------------------------------------------------------
*    This is a part of the open source program named "DECX", copyright c Wayne,
*    2021.04.16
*/

#pragma once

#include "include.h"



#ifdef __CUDACC__
#define _DEVICE_COMPILE_ 1
#define _HOST_COMPILE 0
#else
#define _DEVICE_COMPILE_ 0
#define _HOST_COMPILE 1
#endif



// define for using __half and the corresponding functions


#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 350
#define __ABOVE_SM_35 1
#endif

#if __CUDA_ARCH__ >= 700
#define __ABOVE_SM_70 1
#endif

#if __CUDA_ARCH__ >= 530
#define __ABOVE_SM_53 1
#endif
#endif



#if defined(WIN64) || defined(_WIN64) || defined(_WIN64_) || defined(WIN32)
#define Windows
#endif



#if defined(__linux__) || defined(__GNUC__)
#define Linux
#endif