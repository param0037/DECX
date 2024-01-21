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

// STL
//#include <iostream>
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

#else

#include <cstring>
#include <stdlib.h>
#include <malloc.h>
#include <time.h>

#endif


//#include "../../../bin/x64/"

// Windows

// Linux
#if defined(__Linux__) || defined(__GNUC__)
// Linux APIs
//#include <...>
#endif

// SSE
#include <immintrin.h>
#include <mmintrin.h>
#ifndef __GNUC__
#include <intrin.h>
#endif
#include <xmmintrin.h>

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