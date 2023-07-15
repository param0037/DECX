/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _BASIC_H_
#define _BASIC_H_


#include "include.h"
#include "error.h"
#include "../handles/decx_handles.h"
#include "vector_defines.h"


#if defined(_DECX_CORE_CUDA_)
#define CONSTANT_MEM_SIZE 0x10000 / 8
/**
* extern constant memory can be only compiled when generate relocatable device code
* is enabled
* 1. using Visual Studio : properties -> CUDA C/C++ -> generate relocatable device code -> Yes
* 2. using nvcc command line : -rdc=true
*/
namespace decx {
    //extern __constant__ uchar Const_Mem[];
}

#endif



// thread cores
#ifdef _MSC_VER
#define _NO_ALIAS_ __declspec(noalias)
#define _NO_THROW_ __declspec(nothrow)
#endif
#ifdef __GNUC__
#define _NO_ALIAS_ __attribute__((noalias))
#define _NO_THROW_ __attribute__((nothrow))
#endif


#endif