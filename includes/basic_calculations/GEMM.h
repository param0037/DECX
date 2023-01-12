/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _GEMM_H_
#define _FEMM_H_

#include "../classes/GPU_Matrix.h"
#include "../classes/Matrix.h"
#include "../classes/MatrixArray.h"

namespace de
{
	namespace cuda
	{
		_DECX_API_ de::DH GEMM(de::Matrix& A, de::Matrix& B, de::Matrix& dst);


		_DECX_API_ de::DH GEMM(de::Matrix& A, de::Matrix& B, de::Matrix& C, de::Matrix& dst);


		_DECX_API_ de::DH GEMM(de::GPU_Matrix& A, de::GPU_Matrix& B, de::GPU_Matrix& dst);


		_DECX_API_ de::DH GEMM(de::GPU_Matrix& A, de::GPU_Matrix& B, de::GPU_Matrix& dst, const int flag);


		_DECX_API_ de::DH GEMM(de::GPU_Matrix& A, de::GPU_Matrix& B, de::GPU_Matrix& C, de::GPU_Matrix& dst);


		_DECX_API_ de::DH GEMM(de::GPU_Matrix& A, de::GPU_Matrix& B, de::GPU_Matrix& C, de::GPU_Matrix& dst, const int flag);


		_DECX_API_ de::DH GEMM3(de::MatrixArray& A, de::MatrixArray& B, de::MatrixArray& dst, const int flag);


		_DECX_API_ de::DH GEMM3(de::MatrixArray& A, de::MatrixArray& B, de::MatrixArray& C, de::MatrixArray& dst, const int flag);
	}

	namespace cpu
	{
		_DECX_API_ de::DH GEMM(de::Matrix& A, de::Matrix& B, de::Matrix& dst);


		_DECX_API_ de::DH GEMM(de::Matrix& A, de::Matrix& B, de::Matrix& dst);


		_DECX_API_ de::DH GEMM(de::Matrix& A, de::Matrix& B, de::Matrix& C, de::Matrix& dst);


		_DECX_API_ de::DH GEMM(de::Matrix& A, de::Matrix& B, de::Matrix& C, de::Matrix& dst);
	}
}

#endif