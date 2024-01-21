/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _FFT_H_
#define _FFT_H_

#include "../../basic.h"
#include "../../classes/Vector.h"
#include "../../classes/GPU_Vector.h"
#include "../../classes/GPU_Matrix.h"
#include "../../classes/class_utils.h"



namespace de
{
	namespace signal
	{
		enum FFT_flags {
			FFT_R2C = 0,
			FFT_C2C = 1,
			IFFT_C2C = 2,
			IFFT_C2R = 3
		};

		namespace cuda {
			_DECX_API_ de::DH FFT1D(de::Vector& src, de::Vector& dst, const int FFT_flag);


			_DECX_API_ de::DH FFT1D(de::GPU_Vector& src, de::GPU_Vector& dst, const int FFT_flag);


			_DECX_API_ de::DH IFFT1D(de::Vector& src, de::Vector& dst, const int FFT_flag);


			_DECX_API_ de::DH IFFT1D(de::GPU_Vector& src, de::GPU_Vector& dst, const int FFT_flag);
		}

		namespace cpu
		{
			_DECX_API_ de::DH FFT1D(de::Vector& src, de::Vector& dst, const int FFT_flag);


			_DECX_API_ de::DH IFFT1D(de::Vector& src, de::Vector& dst, const int FFT_flag);
		}
	}
}



namespace de
{
	namespace dsp
	{
		namespace cpu
		{
			_DECX_API_ de::DH FFT(de::Vector& src, de::Vector& dst);


			_DECX_API_ de::DH IFFT(de::Vector& src, de::Vector& dst);
		}
	}
}



namespace de
{
	namespace signal {
		namespace cpu {
			_DECX_API_ de::DH Module(de::Matrix& src, de::Matrix& dst);


			_DECX_API_ de::DH Angle(de::Matrix& src, de::Matrix& dst);
		}
	}
}


namespace de
{
	namespace signal
	{
		namespace cuda {
			_DECX_API_ de::DH FFT2D(de::Matrix& src, de::Matrix& dst, const int FFT_flag);


			_DECX_API_ de::DH FFT2D(de::GPU_Matrix& src, de::GPU_Matrix& dst, const int FFT_flag);


			_DECX_API_ de::DH IFFT2D(de::Matrix& src, de::Matrix& dst, const int FFT_flag);


			_DECX_API_ de::DH IFFT2D(de::GPU_Matrix& src, de::GPU_Matrix& dst, const int FFT_flag);
		}


		namespace cpu {
			_DECX_API_ de::DH FFT2D(de::Matrix& src, de::Matrix& dst, const int FFT_flag);


			_DECX_API_ de::DH IFFT2D(de::Matrix& src, de::Matrix& dst, const int FFT_flag);
		}
	}
}


#endif