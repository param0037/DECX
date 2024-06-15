/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _INCLUDE_H_
#define _INCLUDE_H_

#include "configuration.h"
#include "compile_params.h"

#ifdef __cplusplus
// STL
#include <vector>
#include <thread>
#include <cmath>
#include <initializer_list>

#include <functional>
#include <condition_variable>
#include <future>


#ifdef _DECX_CUDA_PARTS_
// CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <cudart_platform.h>
#include <device_launch_parameters.h>
#include <driver_types.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
//#include <thrust/extrema.h>
#endif

#else

#include <stdlib.h>
#include <malloc.h>
#include <time.h>

#endif

// Windows

// Linux
#if defined(__Linux__) || defined(__GNUC__)
// Linux APIs
//#include <...>
#endif

// SIMD instructions (SSE, AVX2, AVX512, NEON)
#if defined(__x86_64__)
/**
 * Develop in Visual studio, these x86 SIMD files are accessible
 * even using a <> to include.
*/
#include <immintrin.h>
#include <mmintrin.h>
#ifndef __GNUC__
#include <intrin.h>
#endif
#include <xmmintrin.h>
#elif defined(__aarch64__)
/**
 * Develop in x86 platform but cross compile to arm. Hence, arm_neon.h is
 * invisible since it's in NDK include library. The error signs on the 
 * editor are quite annoying. So, include the arm_neon.h downloaded in
 * the project directory.
*/
#include "../../extern/arm/arm_neon.h"
#endif

//#ifdef _DECX_CUDA_PARTS_
//// CUDA thrust
//#include <thrust/host_vector.h>
//#include <thrust/device_vector.h>
//#include <thrust/copy.h>
//#include <thrust/fill.h>
//#include <thrust/sequence.h>
//#include <thrust/transform.h>
//#include <thrust/functional.h>
//#include <thrust/replace.h>
//#endif

#endif