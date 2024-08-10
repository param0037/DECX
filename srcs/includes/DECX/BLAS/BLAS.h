/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _BLAS_H_
#define _BLAS_H_

#include "../classes/Matrix.h"
#include "../classes/Vector.h"
#include "../classes/Tensor.h"
#include "../classes/GPU_Matrix.h"
#include "../classes/GPU_Vector.h"
#include "../classes/GPU_Tensor.h"
#include "../classes/class_utils.h"
#include "../Async/DecxStream.h"
#include "../basic_proc/VectorProc.h"


namespace de
{
	enum GEMM_properties
	{
		HALF_GEMM_DIRECT	= 0,
		HALF_GEMM_ACCURATE	= 1
	};
}


namespace de
{
namespace cuda {
	_DECX_API_ de::DH Transpose(de::GPU_Matrix& src, de::GPU_Matrix& dst);


	_DECX_API_ de::DH Transpose(de::GPU_Vector& src, de::GPU_Vector& dst);
}
}

namespace de
{
	enum REDUCE_METHOD
	{
		_REDUCE2D_FULL_ = 0,
		_REDUCE2D_H_ = 1,
		_REDUCE2D_V_ = 2,
	};


namespace blas {
	namespace cpu {
		_DECX_API_ void GEMM(de::Matrix& A, de::Matrix& B, de::Matrix& dst);


		_DECX_API_ void GEMM(de::Matrix& A, de::Matrix& B, de::Matrix& C, de::Matrix& dst);


		_DECX_API_ void Transpose(de::Matrix& src, de::Matrix& dst);

	}

	namespace cuda
	{
		_DECX_API_ void GEMM(de::Matrix& A, de::Matrix& B, de::Matrix& dst);
		_DECX_API_ void GEMM(de::Matrix& A, de::Matrix& B, de::Matrix& C, de::Matrix& dst);


		// _DECX_API_ void GEMM(de::GPU_Matrix& A, de::GPU_Matrix& B, de::GPU_Matrix& dst, const int flag);
		_DECX_API_ void GEMM(de::GPU_Matrix& A, de::GPU_Matrix& B, de::GPU_Matrix& dst);
		_DECX_API_ void GEMM(de::GPU_Matrix& A, de::GPU_Matrix& B, de::GPU_Matrix& C, de::GPU_Matrix& dst,
            const de::Number alpha, const de::Number beta);


		_DECX_API_ void GEMM(de::Vector& A, de::Matrix& B, de::Vector& dst, const uint32_t _fp16_accu = 0);
		_DECX_API_ void GEMM(de::Matrix& A, de::Vector& B, de::Vector& dst, const uint32_t _fp16_accu = 0);


		_DECX_API_ void Dot_product(de::Vector& A, de::Vector& B, de::Number& res, const uint32_t _fp16_accu);


		_DECX_API_ void Dot_product(de::GPU_Vector& A, de::GPU_Vector& B, de::Number& res, const uint32_t _fp16_accu);


		_DECX_API_ void Dot_product(de::Matrix& A, de::Matrix& B, de::Vector& dst, const de::REDUCE_METHOD _rd_method, const uint32_t _fp16_accu);
	}
}
}



namespace de
{
	namespace nn {
		namespace cuda
		{
			_DECX_API_ de::DH Conv2D(de::GPU_Tensor& src, de::GPU_TensorArray& kernel, de::GPU_Tensor& dst,
				const de::Point2D strides = { 1, 1 }, const de::extend_label extend = de::_EXTEND_NONE_);
		}
	}
}


#endif