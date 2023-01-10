/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/


#ifndef _INCLUDE_H_
#define _INCLUDE_H_


// STL
#include <iostream>
#include <vector>
#include <thread>
#include <cmath>
#include <initializer_list>

#include <functional>
#include <condition_variable>
#include <future>

#ifdef Windows
#include <windows.h>
#endif


#if defined _DECX_CUDA_CODES_ || defined _DECX_ALLOC_CODES_ || defined(_DECX_CLASSES_CODES_)
// CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <cudart_platform.h>
#include <device_launch_parameters.h>
#include <driver_types.h>
#include <cuda_fp16.h>
#include <thrust/extrema.h>

#else

#include <cstring>
#include <stdlib.h>
#include <malloc.h>

#endif


//#include "../../../bin/x64/"

// Windows
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#ifndef CUDA_EXTRA_HOST
#include <Windows.h>



#endif
#endif

// Linux
#if defined(__Linux__) || defined(__GNUC__)
// Linux APIs
//#include <...>
#endif

// SSE
#include <immintrin.h>

#if defined(_DECX_CUDA_CODES_) || defined(_DECX_CLASSES_CODES_)
// CUDA thrust
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/replace.h>
#endif

#endif