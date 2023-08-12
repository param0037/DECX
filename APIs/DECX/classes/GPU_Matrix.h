/**
*	---------------------------------------------------------------------
*	Author : Wayne Anderson
*   Date   : 2021.04.16
*	---------------------------------------------------------------------
*	This is a part of the open source program named "DECX", copyright c Wayne,
*	2021.04.16
*/

#pragma once

#ifndef _GPU_MATRIX_H_
#define _GPU_MATRIX_H_

#include "../basic.h"

namespace de
{
	class _DECX_API_ GPU_Matrix
	{
	public:
		GPU_Matrix() {}


		virtual uint Width() = 0;


		virtual uint Height() = 0;


		virtual void release() = 0;


		virtual int Type() = 0;


		virtual de::GPU_Matrix& SoftCopy(de::GPU_Matrix& src) = 0;


		~GPU_Matrix() {}
	};


	_DECX_API_ de::GPU_Matrix& CreateGPUMatrixRef();


	_DECX_API_ de::GPU_Matrix* CreateGPUMatrixPtr();


	_DECX_API_ de::GPU_Matrix& CreateGPUMatrixRef(const int _type, const uint width, const uint height);


	_DECX_API_ de::GPU_Matrix* CreateGPUMatrixPtr(const int _type, const uint width, const uint height);
}

#endif
