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
	namespace cuda
	{
		_DECX_API_ de::DH GEMM(de::Matrix& A, de::Matrix& B, de::Matrix& dst);


		_DECX_API_ de::DH GEMM(de::Matrix& A, de::Matrix& B, de::Matrix& C, de::Matrix& dst);


		_DECX_API_ de::DH GEMM(de::GPU_Matrix& A, de::GPU_Matrix& B, de::GPU_Matrix& dst, const int flag);


		_DECX_API_ de::DH GEMM(de::GPU_Matrix& A, de::GPU_Matrix& B, de::GPU_Matrix& C, de::GPU_Matrix& dst);


		_DECX_API_ de::DH GEMM(de::GPU_Matrix& A, de::GPU_Matrix& B, de::GPU_Matrix& C, de::GPU_Matrix& dst, const int flag);


		_DECX_API_ void GEMM_Async(de::Matrix& A, de::Matrix& B, de::Matrix& dst, de::DecxStream& S);


		_DECX_API_ void GEMM_Async(de::Matrix& A, de::Matrix& B, de::Matrix& C, de::Matrix& dst, de::DecxStream& S);


		_DECX_API_ void GEMM_Async(de::GPU_Matrix& A, de::GPU_Matrix& B, de::GPU_Matrix& dst, const int flag, de::DecxStream& S);


		_DECX_API_ void GEMM_Async(de::GPU_Matrix& A, de::GPU_Matrix& B, de::GPU_Matrix& C, de::GPU_Matrix& dst, const int flag, de::DecxStream& S);
	}

	namespace cpu
	{
		_DECX_API_ de::DH GEMM(de::Matrix& A, de::Matrix& B, de::Matrix& dst);


		_DECX_API_ de::DH GEMM(de::Matrix& A, de::Matrix& B, de::Matrix& dst);


		_DECX_API_ de::DH GEMM(de::Matrix& A, de::Matrix& B, de::Matrix& C, de::Matrix& dst);


		_DECX_API_ de::DH GEMM(de::Matrix& A, de::Matrix& B, de::Matrix& C, de::Matrix& dst);


		_DECX_API_ void GEMM_Async(de::Matrix& A, de::Matrix& B, de::Matrix& dst, de::DecxStream& S);


		_DECX_API_ void GEMM_Async(de::Matrix& A, de::Matrix& B, de::Matrix& C, de::Matrix& dst, de::DecxStream& S);
	}
}



namespace de
{
	enum conv_property
	{
		de_conv_no_compensate = 0,
		de_conv_zero_compensate = 1,

		half_conv_ordinary = 2,
		half_conv_accurate = 3
	};

	namespace cuda
	{
		_DECX_API_ de::DH Filter2D(de::Matrix& src, de::Matrix& kernel, de::Matrix& dst,
			const int conv_flag, const int accu_flag, const de::_DATA_TYPES_FLAGS_ output_type);


		_DECX_API_ de::DH Filter2D(de::GPU_Matrix& src, de::GPU_Matrix& kernel, de::GPU_Matrix& dst,
			const int conv_flag, const int accu_flag, const de::_DATA_TYPES_FLAGS_ output_type);


		_DECX_API_ de::DH
			Filter2D_multi_channel(de::MatrixArray& src, de::Matrix& kernel, de::MatrixArray& dst,
				const int conv_flag, const int accu_flag, const de::_DATA_TYPES_FLAGS_ output_type);

		_DECX_API_ de::DH
			Filter2D_multi_channel(de::GPU_MatrixArray& src, de::GPU_Matrix& kernel, de::GPU_MatrixArray& dst,
				const int conv_flag, const int accu_flag, const de::_DATA_TYPES_FLAGS_ output_type);


		_DECX_API_ de::DH
			Filter2D_multi_channel(de::MatrixArray& src, de::MatrixArray& kernel, de::MatrixArray& dst,
				const int conv_flag, const int accu_flag, const de::_DATA_TYPES_FLAGS_ output_type);

		_DECX_API_ de::DH
			Filter2D_multi_channel(de::GPU_MatrixArray& src, de::GPU_MatrixArray& kernel, de::GPU_MatrixArray& dst,
				const int conv_flag, const int accu_flag, const de::_DATA_TYPES_FLAGS_ output_type);


		_DECX_API_ de::DH Conv2D(de::GPU_Tensor& src, de::GPU_TensorArray& kernel,
			de::GPU_Tensor& dst, const de::Point2D strideXY, const int flag, const int accu_flag);


		_DECX_API_ void Filter2D_Async(de::Matrix& src, de::Matrix& kernel, de::Matrix& dst,
			const int conv_flag, const int accu_flag, const de::_DATA_TYPES_FLAGS_ output_type, de::DecxStream& S);


		_DECX_API_ void Filter2D_Async(de::GPU_Matrix& src, de::GPU_Matrix& kernel, de::GPU_Matrix& dst,
			const int conv_flag, const int accu_flag, const de::_DATA_TYPES_FLAGS_ output_type, de::DecxStream& S);
	}


	namespace cpu
	{
		_DECX_API_ de::DH Filter2D(de::Matrix& src, de::Matrix& kernel, de::Matrix& dst, const int flag, const de::_DATA_TYPES_FLAGS_ _output_type);



		_DECX_API_ de::DH Filter2D_single_channel(de::MatrixArray& src, de::Matrix& kernel, de::MatrixArray& dst, const int flag);



		_DECX_API_ de::DH Filter2D_multi_channel(de::MatrixArray& src, de::MatrixArray& kernel, de::MatrixArray& dst, const int flag);



		_DECX_API_ de::DH Conv2D(de::Tensor& src, de::TensorArray& kernel, de::Tensor& dst, const de::Point2D strideXY, const int conv_flag);


		_DECX_API_ void Filter2D_Async(de::Matrix& src, de::Matrix& kernel, de::Matrix& dst,
			const int conv_flag, const de::_DATA_TYPES_FLAGS_ output_type, de::DecxStream& S);
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