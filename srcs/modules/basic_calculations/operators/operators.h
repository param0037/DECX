/**
*    ---------------------------------------------------------------------
*    Author : Wayne Anderson
*    Date   : 2021.04.16
*    ---------------------------------------------------------------------
*    This is a part of the open source program named "DECX", copyright c Wayne,
*    2021.04.16
*/

#ifndef _OPERATORS_H_
#define _OPERATORS_H_


#ifdef _DECX_CUDA_CODES_
// addition
#include "Matrix/cuda_add.h"
#include "Vector/cuda_add.h"
#include "Tensor/cuda_add.h"

// subtraction
#include "Matrix/cuda_subtract.h"
#include "Vector/cuda_subtract.h"
#include "Tensor/cuda_subtract.h"

// multiply
#include "Matrix/cuda_multiply.h"
#include "Vector/cuda_multiply.h"
#include "Tensor/cuda_multiply.h"

// divide
#include "Matrix/cuda_divide.h"
#include "Vector/cuda_divide.h"
#include "Tensor/cuda_divide.h"

// fma
#include "Matrix/cuda_fma.h"
#include "Vector/cuda_fma.h"
#include "Tensor/cuda_fma.h"

// fms
#include "Matrix/cuda_fms.h"
#include "Vector/cuda_fms.h"
#include "Tensor/cuda_fms.h"

#endif

#ifdef _DECX_CPU_CODES_
// addition
#include "Matrix/cpu_add.h"

// subtract
#include "Matrix/cpu_subtract.h"


// multiply
#include "Matrix/cpu_multiply.h"


// divide
#include "Matrix/cpu_divide.h"


// fma
#include "Matrix/cpu_fma.h"


#endif


#endif