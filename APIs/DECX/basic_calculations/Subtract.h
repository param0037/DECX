/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _SUBTRACT_H_
#define _SUBTRACT_H_


#include "../classes/Matrix.h"
#include "../classes/Vector.h"
#include "../classes/Tensor.h"
#include "../classes/GPU_Matrix.h"
#include "../classes/GPU_Vector.h"
#include "../classes/GPU_Tensor.h"


namespace de
{
	namespace cuda
	{
		_DECX_API_ de::DH Sub(de::GPU_Vector& A, de::GPU_Vector& B, de::GPU_Vector& dst);


		_DECX_API_ de::DH Sub(de::GPU_Vector& src, void* __x, de::GPU_Vector& dst);


		_DECX_API_ de::DH Sub(void* __x, de::GPU_Vector& src, de::GPU_Vector& dst);
	}

	namespace cpu
	{
		_DECX_API_ de::DH Sub(de::Vector& A, de::Vector& B, de::Vector& dst);


		_DECX_API_ de::DH Sub(de::Vector& src, void* __x, de::Vector& dst);


		_DECX_API_ de::DH Sub(void* __x, de::Vector& src, de::Vector& dst);


		_DECX_API_ de::DH Sub(de::Matrix& A, de::Matrix& B, de::Matrix& dst);


		_DECX_API_ de::DH Sub(de::Matrix& src, void* __x, de::Matrix& dst);


		_DECX_API_ de::DH Sub(void* __x, de::Matrix& src, de::Matrix& dst);
	}
}

#endif