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


#ifndef _COMPILE_PARAMS_H_
#define _COMPILE_PARAMS_H_

#include "include.h"


#ifdef __CUDACC__
#define _DEVICE_COMPILE_ 1
#define _HOST_COMPILE 0
#else
#define _DEVICE_COMPILE_ 0
#define _HOST_COMPILE 1
#endif

#ifdef _DECX_CUDA_PARTS_

/*
* This macro is about adaptations for optimization using CUDA fp16 data on host codes
* CUDA fp16 is a type of data which consists of 16 bits. Theoretically, half of the
* bandwith usage can be saved comapred to fp32.
* 
* If this switch is on (set to 1):
* Where there is ensured that overflow is avoided and precision loss is neglected, fp16
* is used as intermediate variables and results to save the bandwith. 
* Warning : If you hope to turn it on, please make sure that the computability of the target
* platform and __CUDA_ARCH__ value is higher than 530, such that fp16 is supported.
*/
#define _CUDA_FP16_OPTIMIZATION_ 1

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


#ifdef _MSC_VER
#define __STDCALL__ __stdcall
#define __VECTORCALL__ __vectorcall
#endif
#ifdef __GNUC__
#define __STDCALL__ __attribute__((stdcall))
#define __VECTORCALL__ __attribute__((vectorcall))
#endif


#if defined(__linux__) || defined(__GNUC__)
#define Linux
#endif


#endif