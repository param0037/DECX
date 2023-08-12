/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _OPERATORS_H_
#define _OPERATORS_H_

#include "../classes/Matrix.h"
#include "../classes/Vector.h"
#include "../classes/Tensor.h"
#include "../classes/GPU_Matrix.h"
#include "../classes/GPU_Vector.h"
#include "../classes/GPU_Tensor.h"
#include "../classes/class_utils.h"
#include "../Async/DecxStream.h"


namespace de
{
	namespace cuda
	{
		_DECX_API_  de::DH Add(de::GPU_Vector& A, de::GPU_Vector& B, de::GPU_Vector& dst);


		_DECX_API_  de::DH Add(de::GPU_Vector& src, void* __x, de::GPU_Vector& dst);


		_DECX_API_ de::DH Sub(de::GPU_Vector& A, de::GPU_Vector& B, de::GPU_Vector& dst);


		_DECX_API_ de::DH Sub(de::GPU_Vector& src, void* __x, de::GPU_Vector& dst);


		_DECX_API_ de::DH Sub(void* __x, de::GPU_Vector& src, de::GPU_Vector& dst);


		_DECX_API_  de::DH Mul(de::GPU_Vector& A, de::GPU_Vector& B, de::GPU_Vector& dst);


		_DECX_API_  de::DH Mul(de::GPU_Vector& src, void* __x, de::GPU_Vector& dst);


		_DECX_API_ de::DH Div(de::GPU_Vector& A, de::GPU_Vector& B, de::GPU_Vector& dst);


		_DECX_API_ de::DH Div(de::GPU_Vector& src, void* __x, de::GPU_Vector& dst);


		_DECX_API_ de::DH Div(void* __x, de::GPU_Vector& src, de::GPU_Vector& dst);


		_DECX_API_  de::DH Fma(de::GPU_Vector& A, de::GPU_Vector& B, de::GPU_Vector& C, de::GPU_Vector& dst);


		_DECX_API_  de::DH Fma(de::GPU_Vector& src, void* __x, de::GPU_Vector& B, de::GPU_Vector& dst);


		_DECX_API_  de::DH Fms(de::GPU_Vector& A, de::GPU_Vector& B, de::GPU_Vector& C, de::GPU_Vector& dst);


		_DECX_API_  de::DH Fms(de::GPU_Vector& src, void* __x, de::GPU_Vector& B, de::GPU_Vector& dst);
	}

	namespace cpu
	{
		_DECX_API_ de::DH Add(de::Matrix& A, de::Matrix& B, de::Matrix& dst);


		_DECX_API_ de::DH Add(de::Matrix& src, void* __x, de::Matrix& dst);


		_DECX_API_ de::DH Add(de::Vector& A, de::Vector& B, de::Vector& dst);


		_DECX_API_ de::DH Add(de::Vector& src, void* __x, de::Vector& dst);


		_DECX_API_ de::DH Sub(de::Vector& A, de::Vector& B, de::Vector& dst);


		_DECX_API_ de::DH Sub(de::Vector& src, void* __x, de::Vector& dst);


		_DECX_API_ de::DH Sub(void* __x, de::Vector& src, de::Vector& dst);


		_DECX_API_ de::DH Sub(de::Matrix& A, de::Matrix& B, de::Matrix& dst);


		_DECX_API_ de::DH Sub(de::Matrix& src, void* __x, de::Matrix& dst);


		_DECX_API_ de::DH Sub(void* __x, de::Matrix& src, de::Matrix& dst);


		_DECX_API_ de::DH Mul(de::Matrix& A, de::Matrix& B, de::Matrix& dst);


		_DECX_API_ de::DH Mul(de::Matrix& src, void* __x, de::Matrix& dst);


		_DECX_API_ de::DH Mul(de::Vector& A, de::Vector& B, de::Vector& dst);


		_DECX_API_ de::DH Mul(de::Vector& src, void* __x, de::Vector& dst);


		_DECX_API_ de::DH Div(de::Vector& A, de::Vector& B, de::Vector& dst);


		_DECX_API_ de::DH Div(de::Vector& src, void* __x, de::Vector& dst);


		_DECX_API_ de::DH Div(void* __x, de::Vector& src, de::Vector& dst);


		_DECX_API_ de::DH Div(de::Matrix& A, de::Matrix& B, de::Matrix& dst);


		_DECX_API_ de::DH Div(de::Matrix& src, void* __x, de::Matrix& dst);


		_DECX_API_ de::DH Div(void* __x, de::Matrix& src, de::Matrix& dst);


		_DECX_API_ de::DH Fma(de::Matrix& A, de::Matrix& B, de::Matrix& C, de::Matrix& dst);


		_DECX_API_ de::DH Fma(de::Matrix& src, void* __x, de::Matrix& B, de::Matrix& dst);


		_DECX_API_ de::DH Fma(de::Matrix& A, de::Matrix& B, de::Matrix& C, de::Matrix& dst);


		_DECX_API_ de::DH Fma(de::Matrix& src, void* __x, de::Matrix& B, de::Matrix& dst);


		_DECX_API_ de::DH Fms(de::Matrix& A, de::Matrix& B, de::Matrix& C, de::Matrix& dst);


		_DECX_API_ de::DH Fms(de::Matrix& src, void* __x, de::Matrix& B, de::Matrix& dst);


		_DECX_API_ de::DH Fms(de::Matrix& A, de::Matrix& B, de::Matrix& C, de::Matrix& dst);


		_DECX_API_ de::DH Fms(de::Matrix& src, void* __x, de::Matrix& B, de::Matrix& dst);
	}
}



#endif