/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/

#ifndef _BASIC_H_
#define _BASIC_H_

#include "configuration.h"
#include "include.h"
#include "compile_params.h"
#include "error.h"
#include "../handles/decx_handles.h"
#include "vector_defines.h"



#if defined(_DECX_CUDA_CODES_) || defined(_DECX_CLASSES_CODES_)
#define CONSTANT_MEM_SIZE 0x10000 / 8
/**
* extern constant memory can be only compiled when generate relocatable device code
* is enabled
* 1. using Visual Studio : properties -> CUDA C/C++ -> generate relocatable device code -> Yes
* 2. using nvcc command line : -rdc=true
*/
#ifndef _DECX_CLASSES_CODES_
namespace decx {
    extern __constant__ uchar Const_Mem[];
}
#endif

#endif



//template <typename T>
//struct T2
//{
//    T x, y;
//};


// thread cores
#ifdef Windows
#define _NO_ALIAS_ __declspec(noalias)
#define _NO_THROW_ __declspec(nothrow)
#endif
#ifdef Linux
#define _NO_ALIAS_ __attribute__((noalias))
#define _NO_THROW_ __attribute__((nothrow))
#endif


#endif