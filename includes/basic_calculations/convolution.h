/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _CONVOLUTION_H_
#define _CONVOLUTION_H_

#include "../basic.h"
#include "../classes/Matrix.h"
#include "../classes/GPU_Matrix.h"
#include "../classes/MatrixArray.h"
#include "../classes/GPU_MatrixArray.h"
#include "../classes/TensorArray.h"


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
		_DECX_API_ de::DH Conv2(de::Matrix& src, de::Matrix& kernel, de::Matrix& dst, const uint conv_flag, const int accu_flag);


		_DECX_API_ de::DH Conv2(de::GPU_Matrix& src, de::GPU_Matrix& kernel, de::GPU_Matrix& dst, const uint conv_flag, const int accu_flag);


		_DECX_API_ de::DH Conv2_single_kernel(
			de::GPU_MatrixArray& src, de::GPU_Matrix& kernel, de::GPU_MatrixArray& dst, const int conv_flag, const int accu_flag);


		_DECX_API_ de::DH Conv2_multi_kernel(
			de::GPU_MatrixArray& src, de::GPU_MatrixArray& kernel, de::GPU_MatrixArray& dst, const int conv_flag, const int accu_flag);


		_DECX_API_ de::DH Conv2_single_kernel(
			de::MatrixArray& src, de::Matrix& kernel, de::MatrixArray& dst, const int conv_flag, const int accu_flag);


		_DECX_API_ de::DH Conv2_multi_kernel(
			de::MatrixArray& src, de::MatrixArray& kernel, de::MatrixArray& dst, const int conv_flag, const int accu_flag);
	}


	namespace cpu
	{
		_DECX_API_ de::DH Conv2(de::Matrix& src, de::Matrix& kernel, de::Matrix& dst, const int flag, const int _output_type);


		//_DECX_API_ de::DH Conv2(de::Matrix& src, de::Matrix& kernel, de::Matrix& dst, const int flag);


		_DECX_API_ de::DH Conv2_single_channel(de::MatrixArray& src, de::Matrix& kernel, de::MatrixArray& dst, const int flag);



		_DECX_API_ de::DH Conv2_multi_channel(de::MatrixArray& src, de::MatrixArray& kernel, de::MatrixArray& dst, const int flag);
	}
}

#endif
