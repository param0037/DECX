/*
*	DECX GPU BLAS(Basic Linear Algebra Subprogram) 
*	---------------------------------------------------------------------
*	@copyright				Wayne Anderson
*	@author					Wayne Anderson
*	@release-date			2021.5.23
*	---------------------------------------------------------------------
* 
*	  This Program should be compiled at C++ 11 or versions above
*	This Program supports from cuda 2.0 to cuda 10.2. However, the 
*	new features of Nvidia like RT core are not included yet. And it
*	supports only single GPU.
*/

#ifndef _DECX_H_
#define _DECX_H_

#define BasicClasses
#define BasicProc
#define DECX_VISION



#ifdef BasicClasses
#include "DECX/classes/Matrix.h"
#include "DECX/classes/Vector.h"
#include "DECX/classes/GPU_Matrix.h"
#include "DECX/classes/GPU_Vector.h"
#include "DECX/classes/GPU_MatrixArray.h"
#include "DECX/classes/Tensor.h"
#include "DECX/classes/TensorArray.h"
#include "DECX/classes/GPU_Tensor.h"
#include "DECX/classes/GPU_TensorArray.h"
#include "DECX/classes/decx_memcpy.h"
#include "DECX/classes/DecxNumber.h"
#endif


#ifdef BasicProc
#include "DECX/BLAS/BLAS.h"
#include "DECX/BLAS/mathematic.h"
#include "DECX/BLAS/operators.h"
#include "DECX/basic_proc/VectorProc.h"
#include "DECX/basic_proc/MatrixProc.h"
#include "DECX/basic_proc/type_cast.h"
#endif


#include "DECX/DSP/fft/FFT.h"
#include "DECX/DSP/filters/filters.h"
#include "DECX/DSP/generators.h"


#ifdef DECX_VISION
#include "DECX/vision/IO_GUI.h"
#include "DECX/vision/NLM.h"
#include "DECX/vision/imgproc.h"
#endif



#include "DECX/Async/DecxStream.h"




#endif