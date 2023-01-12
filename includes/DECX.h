/*
*	DECX GPU BLAS(Basic Linear Algebra Subprogram) 
*	---------------------------------------------------------------------
*	@copyright				Wayne Wang
*	@author					Wayne Wang
*	@release-date			2021.5.23
*	---------------------------------------------------------------------
* 
*	  This Program should be compiled at C++ 11 or versions above
*	This Program supports from cuda 2.0 to cuda 10.2. However, the 
*	new features of Nvidia like RT core are not included yet. And it
*	supports only single GPU.
*	  
*	  此子程序必须在 c++ 11 及以上平台链接和编译。 此子程序支持的 CUDA 版本
*	从 2.0 到 10.2。但是，对于英伟达的新特性， 例如光线追踪核心的调用，尚未
*	支持。另外，此子程序只支持单 GPU 工作。
*/

#ifndef _DECX_H_
#define _DECX_H_

#define BasicClasses
#define BasicProc
#define DECX_VISION
#define DECX_DNN


enum DECX_THREADS_FLAGS
{
	MULTI_HOST_THREADS = 0,
	SINGLE_HOST_THREADS = 1
};


enum DECX_border_type
{
	de_border_constant = 0,
	de_border_mirror = 1,
	de_border_ignore = 2
};



#ifdef BasicClasses
#include "DECX/classes/matrix.h"
#include "DECX/classes/vector.h"
#include "DECX/classes/GPU_Matrix.h"
#include "DECX/classes/GPU_Vector.h"
#include "DECX/classes/GPU_MatrixArray.h"
#include "DECX/classes/Tensor.h"
#include "DECX/classes/TensorArray.h"
#endif


#ifdef BasicProc
#include "DECX/basic_calculations/GEMM.h"
#include "DECX/basic_calculations/convolution.h"
#include "DECX/basic_calculations/dot.h"
#include "DECX/basic_proc/calculus.h"
#include "DECX/basic_proc/type_cast.h"
#include "DECX/basic_proc/type_statistic.h"
#endif


#include "DECX/basic_calculations/Add.h"
#include "DECX/basic_calculations/Subtract.h"
#include "DECX/basic_calculations/Multiply.h"
#include "DECX/basic_calculations/Divide.h"
#include "DECX/basic_calculations/Fma.h"
#include "DECX/basic_calculations/Fms.h"



#include "DECX/signal/fft/fft2D.h"
#include "DECX/signal/fft/fft1D.h"
#include "DECX/signal/filters/filters.h"


#ifdef DECX_VISION
#include "DECX/vision/IO_GUI.h"
#include "DECX/vision/NLM.h"
#endif



#ifdef DECX_DNN

#endif


#endif
